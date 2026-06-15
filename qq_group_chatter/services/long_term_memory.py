from __future__ import annotations

import asyncio
import re
from difflib import SequenceMatcher
from typing import Any

from qq_group_chatter.models import (
    ConversationContext,
    LongTermMemoryBundle,
    LongTermMemoryCandidate,
    LongTermMemoryIngestionJob,
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
    observe_duration,
    record_error,
)
from qq_group_chatter.services.long_term_memory_extractor import LongTermMemoryExtractor


SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{11}\b"),
    re.compile(r"(?i)(password|passwd|token|api[_-]?key|secret)\s*[:=]"),
]


class LongTermMemoryService:
    def __init__(
        self,
        *,
        mem0_client: Any,
        extractor: LongTermMemoryExtractor,
        min_confidence: float = 0.8,
        duplicate_threshold: float = 0.88,
        max_candidates_per_message: int = 2,
    ):
        self._mem0 = mem0_client
        self._extractor = extractor
        self._min_confidence = min_confidence
        self._duplicate_threshold = duplicate_threshold
        self._max_candidates_per_message = max_candidates_per_message
        self._queue: asyncio.Queue[LongTermMemoryIngestionJob | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker is None or self._worker.done():
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
        user_memories, conversation_memories = await asyncio.gather(
            self._search_scope(user_message, user_memory_id(context), "user"),
            self._search_scope(
                user_message,
                conversation_memory_id(context),
                "conversation",
            ),
        )
        return LongTermMemoryBundle(
            user_memories=user_memories,
            conversation_memories=conversation_memories,
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

    async def _process_job(self, job: LongTermMemoryIngestionJob) -> None:
        candidates = await self._extractor.extract(
            user_message=job.user_message,
            context=job.context,
        )
        for candidate in candidates[: self._max_candidates_per_message]:
            await self._ingest_candidate(candidate, job)

    async def _ingest_candidate(
        self,
        candidate: LongTermMemoryCandidate,
        job: LongTermMemoryIngestionJob,
    ) -> None:
        if not self._is_valid_candidate(candidate):
            MEMORY_CANDIDATES_TOTAL.labels(
                scope=candidate.scope,
                kind=candidate.kind,
                result="validation_skip",
            ).inc()
            return

        target_id = self._target_memory_id(candidate, job.context)
        existing = await self._search_scope(candidate.content, target_id, candidate.scope, limit=3)
        if self._is_duplicate(candidate.content, existing):
            MEMORY_DUPLICATE_SKIPS_TOTAL.labels(scope=candidate.scope).inc()
            MEMORY_CANDIDATES_TOTAL.labels(
                scope=candidate.scope,
                kind=candidate.kind,
                result="duplicate_skip",
            ).inc()
            return

        metadata = {
            "source": "qq",
            "conversation_id": job.context.conversation_id,
            "conversation_type": job.context.conversation_type,
            "message_id": job.context.message_id,
            "scope": candidate.scope,
            "kind": candidate.kind,
        }
        try:
            with observe_duration(
                metric=MEM0_ADD_LATENCY_SECONDS,
                labels={"scope": candidate.scope},
                log_name="mem0_add",
                log_fields={
                    "conversation_id": job.context.conversation_id,
                    "message_id": job.context.message_id,
                    "memory_scope": candidate.scope,
                    "memory_kind": candidate.kind,
                },
            ):
                await asyncio.to_thread(
                    self._mem0.add,
                    [{"role": "user", "content": candidate.content}],
                    user_id=target_id,
                    metadata=metadata,
                    infer=False,
                )
            MEM0_ADD_TOTAL.labels(scope=candidate.scope, result="success").inc()
            MEMORY_CANDIDATES_TOTAL.labels(
                scope=candidate.scope,
                kind=candidate.kind,
                result="add",
            ).inc()
        except Exception as exc:
            MEM0_ADD_TOTAL.labels(scope=candidate.scope, result="error").inc()
            record_error("mem0_add", exc)

    async def _search_scope(
        self,
        query: str,
        memory_id: str,
        scope: str,
        limit: int = 5,
    ) -> list[str]:
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
            return normalize_mem0_memories(raw)
        except Exception as exc:
            record_error("mem0_search", exc)
            return []

    def _is_valid_candidate(self, candidate: LongTermMemoryCandidate) -> bool:
        if candidate.scope not in {"user", "conversation"}:
            return False
        if candidate.confidence < self._min_confidence:
            return False
        if not candidate.content.strip():
            return False
        return not any(pattern.search(candidate.content) for pattern in SENSITIVE_PATTERNS)

    def _target_memory_id(
        self,
        candidate: LongTermMemoryCandidate,
        context: ConversationContext,
    ) -> str:
        if candidate.scope == "user":
            return user_memory_id(context)
        return conversation_memory_id(context)

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


def normalize_mem0_memories(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        if "results" in raw:
            raw = raw["results"]
        elif "memories" in raw:
            raw = raw["memories"]
        else:
            raw = [raw]
    memories = []
    for item in raw:
        if isinstance(item, str):
            memories.append(item)
        elif isinstance(item, dict):
            value = item.get("memory") or item.get("content") or item.get("text")
            if value:
                memories.append(str(value))
    return memories


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()
