from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qq_group_chatter.agent.chat_agent import ChatAgent
from qq_group_chatter.agent.deepseek_llm import create_deepseek_chat_llm, _read_dotenv_key
from qq_group_chatter.llm_tracing import LLMTraceStore
from qq_group_chatter.orchestrator import ChatOrchestrator
from qq_group_chatter.services.conversation_archive import ConversationArchiveService
from qq_group_chatter.services.long_term_memory import LongTermMemoryService
from qq_group_chatter.services.long_term_memory_planner import LongTermMemoryPlanner
from qq_group_chatter.services.short_term_memory import ShortTermMemoryService
from qq_group_chatter.services.web_search import WebSearchService, create_default_web_search_service


class MemoryConfigurationError(RuntimeError):
    pass


class NoopMem0Client:
    enabled = False

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int | None = None):
        return []

    def add(
        self,
        messages,
        *,
        user_id: str,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
    ):
        return {"id": None}

    def update(self, memory_id: str, data: str, metadata: dict[str, Any] | None = None):
        return {"id": memory_id}

    def delete(self, memory_id: str):
        return {"id": memory_id}

    def get_all(self, *, filters: dict[str, Any] | None = None, top_k: int = 20):
        return {"results": []}


@dataclass
class ChatBotApplication:
    orchestrator: ChatOrchestrator
    long_term_memory: LongTermMemoryService
    conversation_archive: ConversationArchiveService | None = None
    web_search: WebSearchService | None = None
    llm_trace_store: LLMTraceStore | None = None

    async def start(self) -> None:
        await self.long_term_memory.start()
        if self.conversation_archive is not None:
            await self.conversation_archive.start()

    async def stop(self) -> None:
        if self.conversation_archive is not None:
            await self.conversation_archive.stop()
        await self.long_term_memory.stop()


def create_default_mem0_client(
    *,
    collection_prefix: str = "qq_group_chatter_memories",
    qdrant_subdir: str = "qdrant",
    history_db_name: str = "history.db",
) -> Any:
    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or _read_dotenv_key()
    if not deepseek_key:
        raise MemoryConfigurationError(
            "DEEPSEEK_API_KEY is required to enable Mem0 long-term memory."
        )

    mem0_dir = Path(os.getenv("MEM0_DIR", ".mem0")).resolve()
    os.environ.setdefault("MEM0_DIR", str(mem0_dir))
    os.environ.setdefault("MEM0_TELEMETRY", "false")

    try:
        from mem0 import Memory
    except ImportError as exc:
        raise MemoryConfigurationError(
            "The 'mem0ai' package is required to enable Mem0 long-term memory."
        ) from exc

    fastembed_model = (
        os.getenv("MEM0_FASTEMBED_MODEL")
        or _read_dotenv_key("MEM0_FASTEMBED_MODEL")
        or "BAAI/bge-small-zh-v1.5"
    )
    config = _build_mem0_config(
        deepseek_key=deepseek_key,
        fastembed_model=fastembed_model,
        mem0_dir=mem0_dir,
        collection_prefix=collection_prefix,
        qdrant_subdir=qdrant_subdir,
        history_db_name=history_db_name,
    )
    try:
        return Memory.from_config(config)
    except Exception as exc:
        raise MemoryConfigurationError(
            "Failed to initialize Mem0 long-term memory. "
            "Install configured dependencies and make sure the fastembed model is available."
        ) from exc


def _collection_suffix(model_name: str) -> str:
    suffix = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    return suffix or "default"


def _build_mem0_config(
    *,
    deepseek_key: str,
    fastembed_model: str,
    mem0_dir: str | Path,
    collection_prefix: str = "qq_group_chatter_memories",
    qdrant_subdir: str = "qdrant",
    history_db_name: str = "history.db",
) -> dict[str, Any]:
    resolved_mem0_dir = Path(mem0_dir).resolve()
    embedding_dims = _fastembed_model_dims(fastembed_model)
    collection_name = (
        f"{collection_prefix}_{_collection_suffix(fastembed_model)}_{embedding_dims}d"
    )
    return {
        "llm": {
            "provider": "deepseek",
            "config": {
                "api_key": deepseek_key,
                "model": "deepseek-v4-pro",
                "temperature": 0.0,
                "max_tokens": 1000,
                "deepseek_base_url": "https://api.deepseek.com",
            },
        },
        "embedder": {
            "provider": "fastembed",
            "config": {
                "model": fastembed_model,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "embedding_model_dims": embedding_dims,
                "path": str(resolved_mem0_dir / qdrant_subdir),
            },
        },
        "history_db_path": str(resolved_mem0_dir / history_db_name),
    }


def _fastembed_model_dims(model_name: str) -> int:
    if model_name == "BAAI/bge-small-zh-v1.5":
        return 512
    try:
        from fastembed import TextEmbedding

        return TextEmbedding.get_embedding_size(model_name)
    except Exception as exc:
        raise MemoryConfigurationError(
            f"Failed to resolve embedding dimensions for fastembed model '{model_name}'."
        ) from exc


def create_default_long_term_memory_service(
    *,
    mem0_client: Any | None = None,
    planner_llm: Any | None = None,
    llm_trace_store: LLMTraceStore | None = None,
) -> LongTermMemoryService:
    resolved_planner_llm = (
        planner_llm
        if planner_llm is not None
        else create_deepseek_chat_llm(
            model="deepseek-v4-pro",
            trace_store=llm_trace_store,
        )
    )
    return LongTermMemoryService(
        mem0_client=mem0_client or create_default_mem0_client(),
        planner=LongTermMemoryPlanner(llm=resolved_planner_llm),
        top_k=_read_int("LONG_TERM_MEMORY_TOP_K", 10),
        candidate_k=_read_int("LONG_TERM_MEMORY_CANDIDATE_K", 30),
        semantic_weight=_read_float("LONG_TERM_MEMORY_SEMANTIC_WEIGHT", 0.85),
        recency_weight=_read_float("LONG_TERM_MEMORY_RECENCY_WEIGHT", 0.15),
        time_decay_days=_read_float("LONG_TERM_MEMORY_TIME_DECAY_DAYS", 180.0),
    )


def create_default_conversation_archive_service(
    *,
    mem0_client: Any | None = None,
) -> ConversationArchiveService | None:
    if not _read_bool("CONVERSATION_ARCHIVE_ENABLED", True):
        return None
    return ConversationArchiveService(
        mem0_client=mem0_client
        or create_default_mem0_client(
            collection_prefix="qq_group_chatter_archive",
            qdrant_subdir="qdrant-archive",
            history_db_name="archive-history.db",
        ),
        enabled=True,
        top_k=_read_int("CONVERSATION_ARCHIVE_TOP_K", 5),
        candidate_k=_read_int("CONVERSATION_ARCHIVE_CANDIDATE_K", 20),
        semantic_weight=_read_float("CONVERSATION_ARCHIVE_SEMANTIC_WEIGHT", 0.85),
        recency_weight=_read_float("CONVERSATION_ARCHIVE_RECENCY_WEIGHT", 0.15),
        time_decay_days=_read_float("CONVERSATION_ARCHIVE_TIME_DECAY_DAYS", 90.0),
        max_messages_per_conversation=_read_int(
            "CONVERSATION_ARCHIVE_MAX_MESSAGES_PER_CONVERSATION",
            5000,
        ),
    )


def create_default_orchestrator(
    *,
    chat_llm: Any | None = None,
    planner_llm: Any | None = None,
    mem0_client: Any | None = None,
) -> ChatOrchestrator:
    """Create an orchestrator for tests or custom wiring.

    Production entrypoints should use create_default_application() so the
    long-term memory worker is started and stopped with the bot lifecycle.
    """
    llm_trace_store = create_default_llm_trace_store()
    resolved_chat_llm = (
        chat_llm
        if chat_llm is not None
        else create_deepseek_chat_llm(trace_store=llm_trace_store)
    )
    return ChatOrchestrator(
        short_term_memory=ShortTermMemoryService(
            max_messages_per_conversation=_read_int("SHORT_TERM_MEMORY_MAX_MESSAGES", 300)
        ),
        long_term_memory=create_default_long_term_memory_service(
            mem0_client=mem0_client,
            planner_llm=planner_llm,
            llm_trace_store=llm_trace_store,
        ),
        chat_agent=ChatAgent(llm=resolved_chat_llm),
    )


def create_default_application(
    *,
    chat_llm: Any | None = None,
    planner_llm: Any | None = None,
    mem0_client: Any | None = None,
    conversation_archive_mem0_client: Any | None = None,
) -> ChatBotApplication:
    llm_trace_store = create_default_llm_trace_store()
    resolved_chat_llm = (
        chat_llm
        if chat_llm is not None
        else create_deepseek_chat_llm(trace_store=llm_trace_store)
    )
    web_search = create_default_web_search_service()
    long_term_memory = create_default_long_term_memory_service(
        mem0_client=mem0_client,
        planner_llm=planner_llm,
        llm_trace_store=llm_trace_store,
    )
    conversation_archive = create_default_conversation_archive_service(
        mem0_client=conversation_archive_mem0_client or mem0_client
    )
    orchestrator = ChatOrchestrator(
        short_term_memory=create_default_short_term_memory_service(),
        long_term_memory=long_term_memory,
        conversation_archive=conversation_archive,
        chat_agent=ChatAgent(llm=resolved_chat_llm),
        web_search=web_search,
    )
    return ChatBotApplication(
        orchestrator=orchestrator,
        long_term_memory=long_term_memory,
        conversation_archive=conversation_archive,
        web_search=web_search,
        llm_trace_store=llm_trace_store,
    )


def create_default_short_term_memory_service() -> ShortTermMemoryService:
    return ShortTermMemoryService(
        max_messages_per_conversation=_read_int("SHORT_TERM_MEMORY_MAX_MESSAGES", 300),
        path=os.getenv("SHORT_TERM_MEMORY_PATH")
        or _read_dotenv_key("SHORT_TERM_MEMORY_PATH")
        or ".mem0/short-term-memory.jsonl",
    )


def create_default_llm_trace_store() -> LLMTraceStore:
    if not _read_bool("QQ_GROUP_CHATTER_LLM_TRACE_ENABLED", True):
        return LLMTraceStore.disabled()
    path = os.getenv("QQ_GROUP_CHATTER_LLM_TRACE_PATH") or _read_dotenv_key(
        "QQ_GROUP_CHATTER_LLM_TRACE_PATH"
    )
    return LLMTraceStore.enabled_store(
        path=path or "logs/llm-traces.jsonl",
        max_records=_read_int("QQ_GROUP_CHATTER_LLM_TRACE_MAX_RECORDS", 500),
    )


def _read_bool(env_name: str, default: bool) -> bool:
    raw = os.getenv(env_name) or _read_dotenv_key(env_name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _read_int(env_name: str, default: int) -> int:
    raw = os.getenv(env_name) or _read_dotenv_key(env_name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _read_float(env_name: str, default: float) -> float:
    raw = os.getenv(env_name) or _read_dotenv_key(env_name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default
