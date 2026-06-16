from __future__ import annotations

from time import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from qq_group_chatter.agent.chat_agent import ChatDecision, ChatReplyDecision, WebSearchDecision
from qq_group_chatter.models import (
    ChatMessage,
    ConversationContext,
    LongTermMemoryIngestionJob,
    PendingAssistantReply,
)
from qq_group_chatter.observability import (
    MESSAGES_TOTAL,
    RESPONSE_LATENCY_SECONDS,
    conversation_log_fields,
    observe_duration,
    record_error,
)


class ShortTermMemory(Protocol):
    async def add_message(self, message: ChatMessage) -> None: ...

    async def get_recent(self, conversation_id: str, limit: int = 20) -> list[ChatMessage]: ...


class LongTermMemory(Protocol):
    async def enqueue_ingestion(self, job: LongTermMemoryIngestionJob) -> None: ...

    async def search(self, user_message: str, context: ConversationContext): ...


class ReplyAgent(Protocol):
    async def generate_reply(
        self,
        *,
        user_message: str,
        context: ConversationContext,
        short_term_messages: list[ChatMessage],
        long_term_memory,
    ) -> ChatDecision: ...

    async def generate_grounded_search_reply(
        self,
        *,
        user_message: str,
        search_query: str,
        search_sources: list[Any],
        context: ConversationContext,
        short_term_messages: list[ChatMessage],
        long_term_memory,
    ) -> str: ...


class WebSearch(Protocol):
    async def search_sources(self, query: str) -> list[Any]: ...


class ChatOrchestrator:
    def __init__(
        self,
        *,
        short_term_memory: ShortTermMemory,
        long_term_memory: LongTermMemory,
        chat_agent: ReplyAgent,
        web_search: WebSearch | None = None,
        short_term_limit: int = 20,
    ):
        self._short_term_memory = short_term_memory
        self._long_term_memory = long_term_memory
        self._chat_agent = chat_agent
        self._web_search = web_search
        self._short_term_limit = short_term_limit

    async def handle_message(
        self,
        *,
        context: ConversationContext,
        user_message: str,
        on_search_start: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> PendingAssistantReply | None:
        content = user_message.strip()
        if not content:
            MESSAGES_TOTAL.labels(
                conversation_type=context.conversation_type,
                result="ignored_empty",
            ).inc()
            return None

        with observe_duration(
            metric=RESPONSE_LATENCY_SECONDS,
            log_name="message_handled",
            log_fields=conversation_log_fields(context),
        ):
            try:
                await self._short_term_memory.add_message(
                    ChatMessage(
                        conversation_id=context.conversation_id,
                        role="user",
                        content=content,
                        user_id=context.user_id,
                        nickname=context.nickname,
                        message_id=context.message_id,
                        timestamp=context.timestamp,
                    )
                )
                short_term_messages = await self._short_term_memory.get_recent(
                    context.conversation_id,
                    limit=self._short_term_limit,
                )
                long_term_memory = await self._long_term_memory.search(content, context)
                await self._long_term_memory.enqueue_ingestion(
                    LongTermMemoryIngestionJob(
                        context=context,
                        user_message=content,
                        existing_memories=long_term_memory,
                    )
                )
                decision = await self._chat_agent.generate_reply(
                    user_message=content,
                    context=context,
                    short_term_messages=short_term_messages,
                    long_term_memory=long_term_memory,
                )
                reply = await self._resolve_decision(
                    decision,
                    user_message=content,
                    context=context,
                    short_term_messages=short_term_messages,
                    long_term_memory=long_term_memory,
                    on_search_start=on_search_start,
                )
                return PendingAssistantReply(
                    context=context,
                    content=reply,
                    timestamp=time(),
                )
            except Exception as exc:
                MESSAGES_TOTAL.labels(
                    conversation_type=context.conversation_type,
                    result="error",
                ).inc()
                record_error("chat_orchestrator", exc)
                raise

    async def _resolve_decision(
        self,
        decision: ChatDecision,
        *,
        user_message: str,
        context: ConversationContext,
        short_term_messages: list[ChatMessage],
        long_term_memory,
        on_search_start: Callable[[str], Awaitable[None] | None] | None,
    ) -> str:
        if isinstance(decision, ChatReplyDecision):
            return decision.content
        if not isinstance(decision, WebSearchDecision):
            return "我刚刚没能整理好回复，稍后再试。"
        if self._web_search is None:
            return "我现在没法联网搜索，稍后再试。"

        if on_search_start is not None:
            try:
                result = on_search_start(decision.notice)
                if _is_awaitable(result):
                    await result
            except Exception as exc:
                record_error("web_search_notice", exc)
        try:
            sources = await self._web_search.search_sources(decision.query)
            if not sources:
                return "我搜了一下，但没找到足够可靠的网页正文来确认。"
            return await self._chat_agent.generate_grounded_search_reply(
                user_message=user_message,
                search_query=decision.query,
                search_sources=sources,
                context=context,
                short_term_messages=short_term_messages,
                long_term_memory=long_term_memory,
            )
        except Exception as exc:
            record_error("web_search", exc)
            return "搜索失败，稍后再试。"

    async def record_assistant_reply(self, reply: PendingAssistantReply) -> None:
        await self._short_term_memory.add_message(
            ChatMessage(
                conversation_id=reply.context.conversation_id,
                role="assistant",
                content=reply.content,
                user_id=None,
                nickname=None,
                message_id=None,
                timestamp=reply.timestamp,
            )
        )
        MESSAGES_TOTAL.labels(
            conversation_type=reply.context.conversation_type,
            result="replied",
        ).inc()


def _is_awaitable(value: Any) -> bool:
    return hasattr(value, "__await__")
