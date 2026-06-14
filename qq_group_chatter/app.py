from __future__ import annotations

from typing import Any

from qq_group_chatter.agent.chat_agent import ChatAgent
from qq_group_chatter.orchestrator import ChatOrchestrator
from qq_group_chatter.services.long_term_memory import LongTermMemoryService
from qq_group_chatter.services.long_term_memory_extractor import LongTermMemoryExtractor
from qq_group_chatter.services.short_term_memory import ShortTermMemoryService


class NoopMem0Client:
    def search(self, query: str, *, filters: dict[str, Any] | None = None, limit: int | None = None):
        return []

    def add(self, messages, *, user_id: str, metadata: dict[str, Any] | None = None):
        return {"id": None}


def create_default_long_term_memory_service(
    *,
    mem0_client: Any | None = None,
    extractor_llm: Any | None = None,
) -> LongTermMemoryService:
    return LongTermMemoryService(
        mem0_client=mem0_client or NoopMem0Client(),
        extractor=LongTermMemoryExtractor(llm=extractor_llm),
    )


def create_default_orchestrator(
    *,
    chat_llm: Any | None = None,
    extractor_llm: Any | None = None,
    mem0_client: Any | None = None,
) -> ChatOrchestrator:
    return ChatOrchestrator(
        short_term_memory=ShortTermMemoryService(),
        long_term_memory=create_default_long_term_memory_service(
            mem0_client=mem0_client,
            extractor_llm=extractor_llm,
        ),
        chat_agent=ChatAgent(llm=chat_llm),
    )
