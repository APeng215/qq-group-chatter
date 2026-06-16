from __future__ import annotations

import asyncio
import re
from difflib import SequenceMatcher
from typing import Any

from qq_group_chatter.models import (
    ConversationContext,
    LongTermMemoryBundle,
    LongTermMemoryIngestionJob,
    LongTermMemoryOperation,
    LongTermMemoryRecord,
    conversation_memory_id,
    user_memory_id,
)
from qq_group_chatter.observability import (
    MEM0_ADD_TOTAL,
    MEM0_ADD_LATENCY_SECONDS,
    MEM0_SEARCH_LATENCY_SECONDS,
    MEMORY_CANDIDATES_TOTAL,
    MEMORY_DUPLICATE_SKIPS_TOTAL,
    MEMORY_INGESTION_QUEUE_SIZE,
    conversation_log_fields,
    observe_duration,
    record_error,
)
from qq_group_chatter.services.long_term_memory_planner import LongTermMemoryPlanner


class LongTermMemorySearchError(RuntimeError):
    pass


SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{11}\b"),
    re.compile(r"(?i)(password|passwd|token|api[\s_-]?key|secret|bearer)\s*[:=：是为]"),
    re.compile(r"(密码|口令|令牌|密钥|秘钥|接口密钥)\s*[:=：是为]"),
    re.compile(r"(手机号|手机号码|电话|联系电话|联系方式)\s*[:=：是为]?"),
    re.compile(r"(住址|地址|家庭地址|公司地址)\s*[:=：是为]?"),
    re.compile(r"[\u4e00-\u9fff]+住在[\u4e00-\u9fff0-9\s]+(?:省|市|区|县|镇|乡|村|路|街|号)"),
    re.compile(r"(?i)\b(?:sk|ak)-[a-z0-9][a-z0-9_-]{6,}\b"),
]


class LongTermMemoryService:
    def __init__(
        self,
        *,
        mem0_client: Any,
        planner: LongTermMemoryPlanner,
        min_confidence: float = 0.8,
        duplicate_threshold: float = 0.88,
        max_operations_per_message: int = 2,
        max_records_per_scope: int = 50,
    ):
        self._mem0 = mem0_client
        self._planner = planner
        self._min_confidence = min_confidence
        self._duplicate_threshold = duplicate_threshold
        self._max_operations_per_message = max_operations_per_message
        self._max_records_per_scope = max_records_per_scope
        self._queue: asyncio.Queue[LongTermMemoryIngestionJob | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker is None or self._worker.done():
            await self._health_check()
            self._worker = asyncio.create_task(self._run_worker())

    async def stop(self) -> None:
        if self._worker is not None:
            await self._queue.put(None)
            await self._worker
            self._worker = None
            MEMORY_INGESTION_QUEUE_SIZE.set(self._queue.qsize())
        await self._close_mem0()

    async def join(self) -> None:
        await self._queue.join()

    async def enqueue_ingestion(self, job: LongTermMemoryIngestionJob) -> None:
        await self._queue.put(job)
        MEMORY_INGESTION_QUEUE_SIZE.set(self._queue.qsize())

    async def search(self, user_message: str, context: ConversationContext) -> LongTermMemoryBundle:
        user_memories, conversation_memories, global_memories = await asyncio.gather(
            self._search_scope(user_message, user_memory_id(context), "user"),
            self._search_scope(
                user_message,
                conversation_memory_id(context),
                "conversation",
            ),
            self._search_conversation_global(user_message, context),
        )
        global_memories = _dedupe_global_memories(
            _filter_conversation_global_memories(global_memories, context),
            user_memories,
            conversation_memories,
        )
        return LongTermMemoryBundle(
            user_memories=user_memories,
            conversation_memories=conversation_memories,
            global_memories=global_memories,
        )

    async def _run_worker(self) -> None:
        while True:
            job = await self._queue.get()
            MEMORY_INGESTION_QUEUE_SIZE.set(self._queue.qsize())
            try:
                if job is None:
                    return
                await self._process_job(job)
            except Exception as exc:
                record_error("long_term_memory_worker", exc)
            finally:
                self._queue.task_done()
                MEMORY_INGESTION_QUEUE_SIZE.set(self._queue.qsize())

    async def _close_mem0(self) -> None:
        await self._call_close(getattr(self._mem0, "close", None), stage="mem0_close")
        vector_client = getattr(getattr(self._mem0, "vector_store", None), "client", None)
        await self._call_close(getattr(vector_client, "close", None), stage="mem0_vector_store_close")

    async def _call_close(self, close: Any, *, stage: str) -> None:
        if close is None:
            return
        try:
            result = close()
            if hasattr(result, "__await__"):
                await result
        except Exception as exc:
            record_error(stage, exc)

    async def _health_check(self) -> None:
        await self._search_scope(
            "__qq_group_chatter_startup_health_check__",
            "__qq_group_chatter_startup_health_check__",
            "startup",
            limit=1,
        )

    async def _process_job(self, job: LongTermMemoryIngestionJob) -> None:
        existing_memories = job.existing_memories
        if existing_memories is None:
            existing_memories = await self.search(job.user_message, job.context)
        try:
            planned_operations = await self._planner.plan(
                user_message=job.user_message,
                context=job.context,
                user_memories=existing_memories.user_memories,
                conversation_memories=existing_memories.conversation_memories,
                global_memories=existing_memories.global_memories,
            )
        except Exception as exc:
            record_error("long_term_memory_planner", exc)
            return

        writable_count = 0
        known_records = LongTermMemoryBundle(
            user_memories=list(existing_memories.user_memories),
            conversation_memories=list(existing_memories.conversation_memories),
            global_memories=list(existing_memories.global_memories),
        )
        for operation in planned_operations:
            if not self._is_valid_operation(operation):
                await self._ingest_operation(operation, job, known_records)
                continue
            if operation.action in {"add", "update", "delete"}:
                if writable_count >= self._max_operations_per_message:
                    continue
                writable_count += 1
            ingested = await self._ingest_operation(operation, job, known_records)
            if ingested is not None:
                self._remember_ingested_record(known_records, ingested)

    async def _ingest_operation(
        self,
        operation: LongTermMemoryOperation,
        job: LongTermMemoryIngestionJob,
        existing_memories: LongTermMemoryBundle,
    ) -> LongTermMemoryRecord | None:
        if not self._is_valid_operation(operation):
            MEMORY_CANDIDATES_TOTAL.labels(
                scope=operation.scope,
                kind=operation.kind,
                result="validation_skip",
            ).inc()
            return None

        if operation.action == "skip":
            MEMORY_CANDIDATES_TOTAL.labels(
                scope=operation.scope,
                kind=operation.kind,
                result="planner_skip",
            ).inc()
            return None

        target_id = self._target_memory_id(operation, job.context)
        existing_records = self._existing_records_for_scope(operation.scope, existing_memories) or []
        existing = [record.content for record in existing_records]
        if operation.action == "add" and self._is_duplicate(operation.content, existing):
            MEMORY_DUPLICATE_SKIPS_TOTAL.labels(scope=operation.scope).inc()
            MEMORY_CANDIDATES_TOTAL.labels(
                scope=operation.scope,
                kind=operation.kind,
                result="duplicate_skip",
            ).inc()
            return None

        if operation.action == "delete":
            target_record = _find_record(existing_records, operation.target_id)
            if target_record is not None and target_record.id is not None:
                await self._delete_memory(target_record.id, operation.scope)
                MEMORY_CANDIDATES_TOTAL.labels(
                    scope=operation.scope,
                    kind=operation.kind,
                    result="delete",
                ).inc()
            else:
                MEMORY_CANDIDATES_TOTAL.labels(
                    scope=operation.scope,
                    kind=operation.kind,
                    result="validation_skip",
                ).inc()
            return None

        if operation.action == "update":
            target_record = _find_record(existing_records, operation.target_id)
            if target_record is not None and target_record.id is not None:
                await self._update_memory(target_record, operation.content, operation, job)
                await self._prune_old_memories(target_id, operation.scope)
            else:
                MEMORY_CANDIDATES_TOTAL.labels(
                    scope=operation.scope,
                    kind=operation.kind,
                    result="validation_skip",
                ).inc()
            return None

        await self._add_memory(operation.content, target_id, operation, job)
        await self._prune_old_memories(target_id, operation.scope)
        return LongTermMemoryRecord(id=None, content=operation.content, metadata={"scope": operation.scope})

    async def _add_memory(
        self,
        content: str,
        target_id: str,
        operation: LongTermMemoryOperation,
        job: LongTermMemoryIngestionJob,
    ) -> None:
        metadata = _new_memory_metadata(operation, job)
        try:
            with observe_duration(
                metric=MEM0_ADD_LATENCY_SECONDS,
                labels={"scope": operation.scope},
                log_name="mem0_add",
                log_fields={
                    "memory_scope": operation.scope,
                    "memory_kind": operation.kind,
                    **conversation_log_fields(job.context),
                },
            ):
                await asyncio.to_thread(
                    self._mem0.add,
                    [{"role": "user", "content": content}],
                    user_id=target_id,
                    metadata=metadata,
                    infer=False,
                )
            MEM0_ADD_TOTAL.labels(scope=operation.scope, result="success").inc()
            MEMORY_CANDIDATES_TOTAL.labels(
                scope=operation.scope,
                kind=operation.kind,
                result="add",
            ).inc()
        except Exception as exc:
            MEM0_ADD_TOTAL.labels(scope=operation.scope, result="error").inc()
            record_error("mem0_add", exc)

    async def _update_memory(
        self,
        record: LongTermMemoryRecord,
        content: str,
        operation: LongTermMemoryOperation,
        job: LongTermMemoryIngestionJob,
    ) -> None:
        metadata = {
            **_memory_update_metadata(record.metadata),
            "source": "qq",
            "conversation_id": job.context.conversation_id,
            "conversation_type": job.context.conversation_type,
            "message_id": job.context.message_id,
            "source_user_id": job.context.user_id,
            "source_nickname": _display_nickname(job.context.nickname),
            "scope": operation.scope,
            "kind": operation.kind,
            "last_seen_at": job.context.timestamp,
            "last_seen_message_id": job.context.message_id,
        }
        try:
            await asyncio.to_thread(
                self._mem0.update,
                record.id,
                content,
                metadata=metadata,
            )
            MEMORY_CANDIDATES_TOTAL.labels(
                scope=operation.scope,
                kind=operation.kind,
                result="update",
            ).inc()
        except Exception as exc:
            record_error("mem0_update", exc)

    async def _delete_memory(self, memory_id: str, scope: str) -> None:
        try:
            await asyncio.to_thread(self._mem0.delete, memory_id)
        except Exception as exc:
            record_error("mem0_delete", exc)

    async def _prune_old_memories(self, target_id: str, scope: str) -> None:
        if self._max_records_per_scope <= 0:
            return
        try:
            raw = await asyncio.to_thread(
                self._mem0.get_all,
                filters={"user_id": target_id},
                top_k=1000,
            )
            records = normalize_mem0_records(raw)
        except Exception as exc:
            record_error("mem0_get_all", exc)
            return

        overflow = len(records) - self._max_records_per_scope
        if overflow <= 0:
            return
        sorted_records = sorted(records, key=_record_created_at)
        for record in sorted_records[:overflow]:
            if record.id is not None:
                await self._delete_memory(record.id, scope)

    async def _search_scope(
        self,
        query: str,
        memory_id: str,
        scope: str,
        limit: int = 5,
    ) -> list[LongTermMemoryRecord]:
        try:
            with observe_duration(
                metric=MEM0_SEARCH_LATENCY_SECONDS,
                labels={"scope": scope},
                log_name="mem0_search",
                log_fields={"memory_scope": scope},
            ):
                raw = await asyncio.to_thread(
                    self._mem0.search,
                    query,
                    filters={"user_id": memory_id},
                    top_k=limit,
                )
            return normalize_mem0_records(raw)
        except Exception as exc:
            record_error("mem0_search", exc)
            raise LongTermMemorySearchError(
                f"Failed to search {scope} long-term memory."
            ) from exc

    async def _search_conversation_global(
        self,
        query: str,
        context: ConversationContext,
        limit: int = 5,
    ) -> list[LongTermMemoryRecord]:
        try:
            with observe_duration(
                metric=MEM0_SEARCH_LATENCY_SECONDS,
                labels={"scope": "global"},
                log_name="mem0_search",
                log_fields={"memory_scope": "global"},
            ):
                raw = await asyncio.to_thread(
                    self._mem0.search,
                    query,
                    filters={
                        "user_id": "*",
                        "conversation_id": context.conversation_id,
                    },
                    top_k=limit,
                )
            return normalize_mem0_records(raw)
        except Exception as exc:
            record_error("mem0_search", exc)
            raise LongTermMemorySearchError(
                "Failed to search global long-term memory."
            ) from exc

    def _is_valid_operation(self, operation: LongTermMemoryOperation) -> bool:
        if operation.action not in {"add", "update", "delete", "skip"}:
            return False
        if operation.scope not in {"user", "conversation"}:
            return False
        if operation.confidence < self._min_confidence:
            return False
        if operation.action == "delete":
            return bool(operation.target_id)
        if not operation.content.strip():
            return False
        return not _contains_sensitive_content(operation.content)

    def _target_memory_id(
        self,
        operation: LongTermMemoryOperation,
        context: ConversationContext,
    ) -> str:
        if operation.scope == "user":
            return user_memory_id(context)
        return conversation_memory_id(context)

    def _existing_records_for_scope(
        self,
        scope: str,
        bundle: LongTermMemoryBundle | None,
    ) -> list[LongTermMemoryRecord] | None:
        if bundle is None:
            return None
        if scope == "user":
            return [
                *bundle.user_memories,
                *[
                    record
                    for record in bundle.global_memories
                    if record.metadata.get("scope") == "user"
                ],
            ]
        if scope == "conversation":
            return [
                *bundle.conversation_memories,
                *[
                    record
                    for record in bundle.global_memories
                    if record.metadata.get("scope") == "conversation"
                ],
            ]
        return []

    def _is_duplicate(self, content: str, existing_memories: list[str]) -> bool:
        normalized_content = _normalize_text(content)
        for item in existing_memories:
            normalized_item = _normalize_text(item)
            if not normalized_item:
                continue
            if normalized_content == normalized_item:
                return True
            if SequenceMatcher(None, normalized_content, normalized_item).ratio() >= self._duplicate_threshold:
                return True
        return False

    def _remember_ingested_record(
        self,
        bundle: LongTermMemoryBundle,
        record: LongTermMemoryRecord,
    ) -> None:
        if record.metadata.get("scope") == "user":
            bundle.user_memories.append(record)
        if record.metadata.get("scope") == "conversation":
            bundle.conversation_memories.append(record)


def normalize_mem0_records(raw: Any) -> list[LongTermMemoryRecord]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        if "results" in raw:
            raw = raw["results"]
        elif "memories" in raw:
            raw = raw["memories"]
        else:
            raw = [raw]
    records = []
    for item in raw:
        if isinstance(item, str):
            records.append(LongTermMemoryRecord(id=None, content=item, metadata={}))
        elif isinstance(item, dict):
            record = _normalize_mem0_record(item)
            if record is not None:
                records.append(record)
    return records


def normalize_mem0_memories(raw: Any) -> list[str]:
    return [record.content for record in normalize_mem0_records(raw)]


def _normalize_mem0_record(item: dict[str, Any]) -> LongTermMemoryRecord | None:
    payload = item.get("payload")
    if isinstance(payload, dict):
        value = (
            item.get("memory")
            or item.get("content")
            or item.get("text")
            or item.get("data")
            or payload.get("memory")
            or payload.get("content")
            or payload.get("text")
            or payload.get("data")
        )
    else:
        payload = {}
        value = item.get("memory") or item.get("content") or item.get("text") or item.get("data")
    if not value:
        return None

    metadata: dict[str, object] = {}
    item_metadata = item.get("metadata")
    payload_metadata = payload.get("metadata")
    if isinstance(item_metadata, dict):
        metadata.update(item_metadata)
    if isinstance(payload_metadata, dict):
        metadata.update(payload_metadata)

    content_keys = {"id", "memory", "content", "text", "data", "metadata", "payload"}
    for source in (payload, item):
        for key, extra_value in source.items():
            if key in content_keys:
                continue
            if value:
                metadata.setdefault(key, extra_value)

    record_id = item.get("id") or payload.get("id")
    return LongTermMemoryRecord(
        id=str(record_id) if record_id is not None else None,
        content=str(value),
        metadata=metadata,
    )


def _find_record(
    records: list[LongTermMemoryRecord],
    record_id: str | None,
) -> LongTermMemoryRecord | None:
    if record_id is None:
        return None
    for record in records:
        if record.id == record_id:
            return record
    return None


def _dedupe_global_memories(
    global_memories: list[LongTermMemoryRecord],
    *preferred_groups: list[LongTermMemoryRecord],
) -> list[LongTermMemoryRecord]:
    preferred_ids = {
        record.id
        for group in preferred_groups
        for record in group
        if record.id is not None
    }
    return [
        record
        for record in global_memories
        if record.id is None or record.id not in preferred_ids
    ]


def _filter_conversation_global_memories(
    records: list[LongTermMemoryRecord],
    context: ConversationContext,
) -> list[LongTermMemoryRecord]:
    return [
        record
        for record in records
        if record.metadata.get("conversation_id") == context.conversation_id
        and _is_current_conversation_owner(record, context)
    ]


def _is_current_conversation_owner(
    record: LongTermMemoryRecord,
    context: ConversationContext,
) -> bool:
    owner_id = record.metadata.get("user_id")
    if owner_id is None:
        return True
    owner = str(owner_id)
    return owner.startswith(f"qq_user:{context.conversation_id}:") or owner == conversation_memory_id(context)


def _new_memory_metadata(
    operation: LongTermMemoryOperation,
    job: LongTermMemoryIngestionJob,
) -> dict[str, object]:
    return {
        "source": "qq",
        "conversation_id": job.context.conversation_id,
        "conversation_type": job.context.conversation_type,
        "message_id": job.context.message_id,
        "source_user_id": job.context.user_id,
        "source_nickname": _display_nickname(job.context.nickname),
        "scope": operation.scope,
        "kind": operation.kind,
        "source_created_at": job.context.timestamp,
        "last_seen_at": job.context.timestamp,
    }


def _memory_update_metadata(metadata: dict[str, object]) -> dict[str, object]:
    cleaned = {key: value for key, value in metadata.items() if key not in {"created_at", "updated_at"}}
    if "source_created_at" not in cleaned and "created_at" in metadata:
        cleaned["source_created_at"] = metadata["created_at"]
    return cleaned


def _display_nickname(nickname: str | None) -> str:
    if nickname is None:
        return "未设置"
    text = str(nickname).strip()
    return text or "未设置"


def _record_created_at(record: LongTermMemoryRecord) -> float:
    value = record.metadata.get("source_created_at", record.metadata.get("created_at"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _contains_sensitive_content(value: str) -> bool:
    if any(pattern.search(value) for pattern in SENSITIVE_PATTERNS):
        return True
    digits = re.sub(r"[\s\-()（）]", "", value)
    return re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", digits) is not None
