from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from qq_group_chatter.agent.chat_agent import ChatAgent
from qq_group_chatter.agent.deepseek_llm import create_deepseek_chat_llm, _read_dotenv_key
from qq_group_chatter.orchestrator import ChatOrchestrator
from qq_group_chatter.services.long_term_memory import LongTermMemoryService
from qq_group_chatter.services.long_term_memory_planner import LongTermMemoryPlanner
from qq_group_chatter.services.short_term_memory import ShortTermMemoryService


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

    async def start(self) -> None:
        await self.long_term_memory.start()

    async def stop(self) -> None:
        await self.long_term_memory.stop()


def create_default_mem0_client() -> Any:
    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or _read_dotenv_key()
    if not deepseek_key:
        raise MemoryConfigurationError(
            "DEEPSEEK_API_KEY is required to enable Mem0 long-term memory."
        )

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
    embedding_dims = _fastembed_model_dims(fastembed_model)
    collection_name = f"qq_group_chatter_memories_{_collection_suffix(fastembed_model)}_{embedding_dims}d"

    embedder_config: dict[str, Any] = {
        "provider": "fastembed",
        "config": {
            "model": fastembed_model,
        },
    }

    config = {
        "llm": {
            "provider": "deepseek",
            "config": {
                "api_key": deepseek_key,
                "model": "deepseek-v4-flash",
                "temperature": 0.0,
                "max_tokens": 1000,
                "deepseek_base_url": "https://api.deepseek.com",
            },
        },
        "embedder": embedder_config,
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "embedding_model_dims": embedding_dims,
                "path": ".mem0/qdrant",
            },
        },
    }
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
) -> LongTermMemoryService:
    resolved_planner_llm = (
        planner_llm
        if planner_llm is not None
        else create_deepseek_chat_llm(model="deepseek-v4-flash")
    )
    return LongTermMemoryService(
        mem0_client=mem0_client or create_default_mem0_client(),
        planner=LongTermMemoryPlanner(llm=resolved_planner_llm),
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
    resolved_chat_llm = chat_llm if chat_llm is not None else create_deepseek_chat_llm()
    return ChatOrchestrator(
        short_term_memory=ShortTermMemoryService(),
        long_term_memory=create_default_long_term_memory_service(
            mem0_client=mem0_client,
            planner_llm=planner_llm,
        ),
        chat_agent=ChatAgent(llm=resolved_chat_llm),
    )


def create_default_application(
    *,
    chat_llm: Any | None = None,
    planner_llm: Any | None = None,
    mem0_client: Any | None = None,
) -> ChatBotApplication:
    resolved_chat_llm = chat_llm if chat_llm is not None else create_deepseek_chat_llm()
    long_term_memory = create_default_long_term_memory_service(
        mem0_client=mem0_client,
        planner_llm=planner_llm,
    )
    orchestrator = ChatOrchestrator(
        short_term_memory=ShortTermMemoryService(),
        long_term_memory=long_term_memory,
        chat_agent=ChatAgent(llm=resolved_chat_llm),
    )
    return ChatBotApplication(
        orchestrator=orchestrator,
        long_term_memory=long_term_memory,
    )
