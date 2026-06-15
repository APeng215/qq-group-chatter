from __future__ import annotations

from time import time
from typing import Protocol

from qq_group_chatter.models import (
    ChatMessage,
    ConversationContext,
    LongTermMemoryIngestionJob,
    PendingAssistantReply,
)
from qq_group_chatter.observability import (
    MESSAGES_TOTAL,
    RESPONSE_LATENCY_SECONDS,
    hash_identifier,
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
    ) -> str: ...


class ChatOrchestrator:
    def __init__(
        self,
        *,
        short_term_memory: ShortTermMemory,
        long_term_memory: LongTermMemory,
        chat_agent: ReplyAgent,
        short_term_limit: int = 20,
    ):
        self._short_term_memory = short_term_memory
        self._long_term_memory = long_term_memory
        self._chat_agent = chat_agent
        self._short_term_limit = short_term_limit

    async def handle_message(
        self,
        *,
        context: ConversationContext,
        user_message: str,
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
            log_fields={
                "conversation_id": context.conversation_id,
                "conversation_type": context.conversation_type,
                "user_id_hash": hash_identifier(context.user_id),
                "group_id_hash": hash_identifier(context.group_id),
                "message_id": context.message_id,
            },
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
                await self._long_term_memory.enqueue_ingestion(
                    LongTermMemoryIngestionJob(context=context, user_message=content)
                )
                short_term_messages = await self._short_term_memory.get_recent(
                    context.conversation_id,
                    limit=self._short_term_limit,
                )
                long_term_memory = await self._long_term_memory.search(content, context)
                reply = await self._chat_agent.generate_reply(
                    user_message=content,
                    context=context,
                    short_term_messages=short_term_messages,
                    long_term_memory=long_term_memory,
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
