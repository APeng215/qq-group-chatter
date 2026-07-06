from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from time import time
from typing import Any

from qq_group_chatter.models import (
    ChatMessage,
    ConversationArchiveRecord,
    ConversationContext,
)
from qq_group_chatter.observability import record_error


class ConversationArchiveError(RuntimeError):
    pass


ARCHIVE_CONTEXT_BEFORE_MESSAGES = 2
ARCHIVE_PROMPT_MAX_MESSAGES = 15


class ConversationArchiveService:
    def __init__(
        self,
        *,
        mem0_client: Any,
        enabled: bool = True,
        top_k: int = 5,
        candidate_k: int = 20,
        semantic_weight: float = 0.85,
        recency_weight: float = 0.15,
        time_decay_days: float = 90.0,
        max_messages_per_conversation: int = 5000,
    ):
        self._mem0 = mem0_client
        self._enabled = enabled
        self._top_k = max(1, int(top_k))
        self._candidate_k = max(self._top_k, int(candidate_k))
        self._semantic_weight = float(semantic_weight)
        self._recency_weight = float(recency_weight)
        self._time_decay_seconds = max(1.0, float(time_decay_days) * 24 * 60 * 60)
        self._max_messages_per_conversation = int(max_messages_per_conversation)
        self._queue: asyncio.Queue[ChatMessage | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if not self._enabled:
            return
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_worker())

    async def stop(self) -> None:
        if self._worker is not None:
            await self._queue.put(None)
            await self._worker
            self._worker = None
        await self._close_mem0()

    async def join(self) -> None:
        await self._queue.join()

    async def enqueue_message(self, message: ChatMessage) -> None:
        if not self._enabled:
            return
        if self._worker is None:
            await self._write_message(message)
            return
        await self._queue.put(message)

    async def search(
        self,
        user_message: str,
        context: ConversationContext,
        limit: int | None = None,
        *,
        now: float | None = None,
    ) -> list[ConversationArchiveRecord]:
        if not self._enabled:
            return []
        resolved_limit = max(1, int(limit or self._top_k))
        try:
            raw = await asyncio.to_thread(
                self._mem0.search,
                user_message,
                filters={
                    "user_id": _archive_user_id(context.conversation_id),
                    "conversation_id": context.conversation_id,
                    "archive_type": "conversation_message",
                },
                top_k=max(self._candidate_k, resolved_limit),
            )
        except Exception as exc:
            record_error("conversation_archive_search", exc)
            raise ConversationArchiveError("Failed to search conversation archive.") from exc
        records = _normalize_archive_records(raw)
        records = _exclude_current_message(records, context)
        ranked = sorted(
            records,
            key=lambda record: _rerank_score(
                record,
                now=time() if now is None else now,
                semantic_weight=self._semantic_weight,
                recency_weight=self._recency_weight,
                time_decay_seconds=self._time_decay_seconds,
            ),
            reverse=True,
        )
        semantic_hits = [
            replace(record, is_semantic_hit=True) for record in ranked[:resolved_limit]
        ]
        return await self._expand_with_previous_messages(semantic_hits, context)

    async def _expand_with_previous_messages(
        self,
        semantic_hits: list[ConversationArchiveRecord],
        context: ConversationContext,
    ) -> list[ConversationArchiveRecord]:
        if not semantic_hits:
            return []
        try:
            raw = await asyncio.to_thread(
                self._mem0.get_all,
                filters={
                    "user_id": _archive_user_id(context.conversation_id),
                    "conversation_id": context.conversation_id,
                    "archive_type": "conversation_message",
                },
                top_k=max(10000, self._max_messages_per_conversation * 2),
            )
        except Exception as exc:
            record_error("conversation_archive_context_get_all", exc)
            return sorted(semantic_hits, key=_archive_prompt_sort_key)

        history = _exclude_current_message(_normalize_archive_records(raw), context)
        if not history:
            return sorted(semantic_hits, key=_archive_prompt_sort_key)

        history = sorted(history, key=_archive_prompt_sort_key)
        expanded: dict[tuple[str, float, str], ConversationArchiveRecord] = {}
        for hit in semantic_hits:
            hit_index = _find_history_record_index(history, hit)
            if hit_index is None:
                for key, record in _records_for_prompt([hit]):
                    expanded[key] = record
                continue
            window_start = max(0, hit_index - ARCHIVE_CONTEXT_BEFORE_MESSAGES)
            window = history[window_start : hit_index + 1]
            for key, record in _records_for_prompt(window):
                existing = expanded.get(key)
                if existing is None or record.is_semantic_hit:
                    expanded[key] = replace(
                        record,
                        is_semantic_hit=record.is_semantic_hit
                        or _is_same_archive_record(record, hit),
                    )
            hit_key = _archive_record_key(hit)
            if hit_key not in expanded:
                expanded[hit_key] = hit
            elif hit.is_semantic_hit:
                expanded[hit_key] = replace(expanded[hit_key], is_semantic_hit=True)
        return sorted(expanded.values(), key=_archive_prompt_sort_key)[
            :ARCHIVE_PROMPT_MAX_MESSAGES
        ]

    async def _run_worker(self) -> None:
        while True:
            message = await self._queue.get()
            try:
                if message is None:
                    return
                try:
                    await self._write_message(message)
                except Exception as exc:
                    record_error("conversation_archive_add", exc)
            finally:
                self._queue.task_done()

    async def _write_message(self, message: ChatMessage) -> None:
        conversation_type = _conversation_type_from_id(message.conversation_id)
        await asyncio.to_thread(
            self._mem0.add,
            [{"role": message.role, "content": message.content}],
            user_id=_archive_user_id(message.conversation_id),
            metadata={
                "archive_type": "conversation_message",
                "conversation_id": message.conversation_id,
                "conversation_type": conversation_type,
                "source_user_id": message.user_id,
                "source_nickname": message.nickname,
                "role": message.role,
                "message_id": message.message_id,
                "timestamp": message.timestamp,
            },
            infer=False,
        )
        await self._prune_old_messages(message.conversation_id)

    async def _prune_old_messages(self, conversation_id: str) -> None:
        if self._max_messages_per_conversation <= 0:
            return
        raw = await asyncio.to_thread(
            self._mem0.get_all,
            filters={
                "user_id": _archive_user_id(conversation_id),
                "conversation_id": conversation_id,
                "archive_type": "conversation_message",
            },
            top_k=max(10000, self._max_messages_per_conversation * 2),
        )
        records = _normalize_archive_memory_records(raw)
        overflow = len(records) - self._max_messages_per_conversation
        if overflow <= 0:
            return
        for record in sorted(records, key=_archive_record_sort_key)[:overflow]:
            memory_id = record.get("id")
            if memory_id is not None:
                await asyncio.to_thread(self._mem0.delete, str(memory_id))

    async def _close_mem0(self) -> None:
        await _call_close(getattr(self._mem0, "close", None))
        vector_client = getattr(getattr(self._mem0, "vector_store", None), "client", None)
        await _call_close(getattr(vector_client, "close", None))


def _archive_user_id(conversation_id: str) -> str:
    return f"qq_archive:{conversation_id}"


def _conversation_type_from_id(conversation_id: str) -> str:
    if conversation_id.startswith("qq_group:"):
        return "group"
    if conversation_id.startswith("qq_private:"):
        return "private"
    return "unknown"


def _normalize_archive_records(raw: Any) -> list[ConversationArchiveRecord]:
    items = raw.get("results", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    records = []
    for item in items:
        if not isinstance(item, dict):
            continue
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        metadata: dict[str, object] = {}
        if isinstance(payload.get("metadata"), dict):
            metadata.update(payload["metadata"])
        if isinstance(item.get("metadata"), dict):
            metadata.update(item["metadata"])
        content = str(
            item.get("memory")
            or item.get("content")
            or payload.get("memory")
            or payload.get("content")
            or ""
        ).strip()
        if not content:
            continue
        role = _role_text(metadata.get("role"))
        user_id = _optional_text(metadata.get("source_user_id"))
        nickname = _optional_text(metadata.get("source_nickname"))
        message_id = _optional_text(metadata.get("message_id"))
        if _looks_like_legacy_assistant_record(metadata, user_id, nickname, message_id):
            role = "assistant"
            nickname = "神奈"
        records.append(
            ConversationArchiveRecord(
                content=content,
                role=role,
                user_id=user_id,
                nickname=nickname,
                message_id=message_id,
                timestamp=_float_or_zero(metadata.get("timestamp")),
                score=_optional_float(item.get("score")),
            )
        )
    return records


def _records_for_prompt(
    records: list[ConversationArchiveRecord],
) -> list[tuple[tuple[str, float, str], ConversationArchiveRecord]]:
    return [(_archive_record_key(record), record) for record in records]


def _archive_record_key(record: ConversationArchiveRecord) -> tuple[str, float, str]:
    if record.message_id:
        return ("id", 0.0, record.message_id)
    return ("time", record.timestamp, record.content)


def _find_history_record_index(
    history: list[ConversationArchiveRecord],
    hit: ConversationArchiveRecord,
) -> int | None:
    for index, record in enumerate(history):
        if _is_same_archive_record(record, hit):
            return index
    return None


def _is_same_archive_record(
    left: ConversationArchiveRecord,
    right: ConversationArchiveRecord,
) -> bool:
    if left.message_id and right.message_id:
        return left.message_id == right.message_id
    return left.timestamp == right.timestamp and left.content == right.content


def _archive_prompt_sort_key(record: ConversationArchiveRecord) -> tuple[int, float, str]:
    if record.timestamp > 0:
        return (1, record.timestamp, record.message_id or record.content)
    return (0, 0.0, record.message_id or record.content)


def _normalize_archive_memory_records(raw: Any) -> list[dict[str, Any]]:
    items = raw.get("results", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _archive_record_sort_key(record: dict[str, Any]) -> tuple[float, str]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    if not metadata and isinstance(payload.get("metadata"), dict):
        metadata = payload["metadata"]
    timestamp = _float_or_zero(metadata.get("timestamp"))
    memory_id = str(record.get("id") or "")
    return (timestamp, memory_id)


def _exclude_current_message(
    records: list[ConversationArchiveRecord],
    context: ConversationContext,
) -> list[ConversationArchiveRecord]:
    return [
        record
        for record in records
        if record.message_id != context.message_id
        and (record.message_id is not None or record.timestamp < context.timestamp)
    ]


def _rerank_score(
    record: ConversationArchiveRecord,
    *,
    now: float,
    semantic_weight: float,
    recency_weight: float,
    time_decay_seconds: float,
) -> float:
    semantic = record.score if record.score is not None else 0.0
    age = max(0.0, now - record.timestamp)
    recency = math.exp(-age / time_decay_seconds)
    return semantic_weight * semantic + recency_weight * recency


def _role_text(value: object) -> str:
    text = str(value or "").strip()
    return "assistant" if text == "assistant" else "user"


def _looks_like_legacy_assistant_record(
    metadata: dict[str, object],
    user_id: str | None,
    nickname: str | None,
    message_id: str | None,
) -> bool:
    return (
        metadata.get("role") is None
        and user_id is None
        and nickname is None
        and message_id is None
    )


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


async def _call_close(close: Any) -> None:
    if close is None:
        return
    try:
        result = close()
        if hasattr(result, "__await__"):
            await result
    except Exception as exc:
        record_error("conversation_archive_close", exc)
