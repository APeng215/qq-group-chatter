from __future__ import annotations

from time import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from qq_group_chatter.agent.chat_agent import ChatDecision, ChatReplyDecision, WebSearchDecision
from qq_group_chatter.models import (
    ChatMessage,
    ConversationContext,
    ErrorNoticeContext,
    LongTermMemoryBundle,
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
        memory_warning: ErrorNoticeContext | None = None,
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

    async def generate_error_notice(
        self,
        *,
        error_context: ErrorNoticeContext,
        context: ConversationContext,
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
        short_term_limit: int = 30,
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
                await self.record_user_message(context=context, user_message=content)
                short_term_messages = await self._short_term_memory.get_recent(
                    context.conversation_id,
                    limit=self._short_term_limit,
                )
                memory_warning = None
                try:
                    long_term_memory = await self._long_term_memory.search(content, context)
                except Exception as exc:
                    record_error("long_term_memory_search", exc)
                    memory_warning = ErrorNoticeContext(
                        stage="long_term_memory_search",
                        error_type=type(exc).__name__,
                        impact="本轮回复可能没有用上长期记忆。",
                    )
                    long_term_memory = LongTermMemoryBundle(
                        user_memories=[],
                        conversation_memories=[],
                    )
                decision = await self._chat_agent.generate_reply(
                    user_message=content,
                    context=context,
                    short_term_messages=short_term_messages,
                    long_term_memory=long_term_memory,
                    memory_warning=memory_warning,
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
                    user_message=content,
                    short_term_messages=short_term_messages,
                    long_term_memory=long_term_memory,
                    memory_warning=memory_warning,
                )
            except Exception as exc:
                MESSAGES_TOTAL.labels(
                    conversation_type=context.conversation_type,
                    result="error",
                ).inc()
                record_error("chat_orchestrator", exc)
                raise

    async def record_user_message(
        self,
        *,
        context: ConversationContext,
        user_message: str,
    ) -> None:
        content = user_message.strip()
        if not content:
            return
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

    async def record_assistant_reply(
        self,
        reply: PendingAssistantReply,
        on_memory_error_notice: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> str | None:
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
        if reply.user_message is None:
            return None
        if reply.long_term_memory is None:
            return None
        try:
            await self._long_term_memory.enqueue_ingestion(
                LongTermMemoryIngestionJob(
                    context=reply.context,
                    user_message=reply.user_message,
                    short_term_messages=reply.short_term_messages,
                    existing_memories=reply.long_term_memory,
                    assistant_reply=reply.content,
                    on_error_notice=(
                        None
                        if on_memory_error_notice is None
                        else lambda error_context: self._send_generated_error_notice(
                            error_context,
                            reply.context,
                            on_memory_error_notice,
                        )
                    ),
                )
            )
        except Exception as exc:
            record_error("long_term_memory_enqueue", exc)
            return await self._generate_error_notice(
                ErrorNoticeContext(
                    stage="long_term_memory_enqueue",
                    error_type=type(exc).__name__,
                    impact="刚刚这条消息可能没能写入长期记忆。",
                ),
                reply.context,
            )
        return None

    async def _send_generated_error_notice(
        self,
        error_context: ErrorNoticeContext,
        context: ConversationContext,
        sender: Callable[[str], Awaitable[None] | None],
    ) -> None:
        notice = await self._generate_error_notice(error_context, context)
        try:
            result = sender(notice)
            if _is_awaitable(result):
                await result
        except Exception as exc:
            record_error("memory_error_notice_send", exc)

    async def _generate_error_notice(
        self,
        error_context: ErrorNoticeContext,
        context: ConversationContext,
    ) -> str:
        try:
            notice = await self._chat_agent.generate_error_notice(
                error_context=error_context,
                context=context,
            )
        except Exception as exc:
            record_error("memory_error_notice_generation", exc)
            return _fallback_error_notice(error_context)
        text = str(notice).strip()
        return text or _fallback_error_notice(error_context)


def _is_awaitable(value: Any) -> bool:
    return hasattr(value, "__await__")


def _fallback_error_notice(error_context: ErrorNoticeContext) -> str:
    if error_context.stage == "long_term_memory_search":
        return "记忆好像出了点小问题，这次我可能没用上以前记住的内容。"
    return "记忆好像出了点小问题，刚刚这条我可能没能记下来。"
