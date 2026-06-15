from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ConversationType = Literal["group", "private"]
MessageRole = Literal["user", "assistant"]
MemoryScope = Literal["user", "conversation"]
MemoryKind = Literal[
    "identity",
    "preference",
    "constraint",
    "relationship",
    "conversation_rule",
    "other",
]


@dataclass(frozen=True)
class ConversationContext:
    conversation_id: str
    conversation_type: ConversationType
    user_id: str
    group_id: str | None
    message_id: str
    nickname: str | None
    timestamp: float


@dataclass(frozen=True)
class ChatMessage:
    conversation_id: str
    role: MessageRole
    content: str
    user_id: str | None
    nickname: str | None
    message_id: str | None
    timestamp: float


@dataclass(frozen=True)
class LongTermMemoryCandidate:
    scope: MemoryScope
    content: str
    confidence: float
    kind: MemoryKind


@dataclass(frozen=True)
class LongTermMemoryIngestionJob:
    context: ConversationContext
    user_message: str


@dataclass(frozen=True)
class PendingAssistantReply:
    context: ConversationContext
    content: str
    timestamp: float


@dataclass(frozen=True)
class LongTermMemoryBundle:
    user_memories: list[str]
    conversation_memories: list[str]

    def as_prompt_section(self) -> str:
        user_lines = "\n".join(f"- {item}" for item in self.user_memories) or "- \u65e0"
        conversation_lines = (
            "\n".join(f"- {item}" for item in self.conversation_memories) or "- \u65e0"
        )
        return (
            "\u76f8\u5173\u4e2a\u4eba\u957f\u671f\u8bb0\u5fc6\uff1a\n"
            f"{user_lines}\n\n"
            "\u76f8\u5173\u4f1a\u8bdd\u957f\u671f\u8bb0\u5fc6\uff1a\n"
            f"{conversation_lines}"
        )


def build_group_conversation_context(
    *,
    group_id: str | int,
    user_id: str | int,
    message_id: str | int,
    nickname: str | None,
    timestamp: float,
) -> ConversationContext:
    group = str(group_id)
    return ConversationContext(
        conversation_id=f"qq_group:{group}",
        conversation_type="group",
        user_id=str(user_id),
        group_id=group,
        message_id=str(message_id),
        nickname=nickname,
        timestamp=timestamp,
    )


def build_private_conversation_context(
    *,
    user_id: str | int,
    message_id: str | int,
    nickname: str | None,
    timestamp: float,
) -> ConversationContext:
    user = str(user_id)
    return ConversationContext(
        conversation_id=f"qq_private:{user}",
        conversation_type="private",
        user_id=user,
        group_id=None,
        message_id=str(message_id),
        nickname=nickname,
        timestamp=timestamp,
    )


def user_memory_id(context: ConversationContext) -> str:
    return f"qq_user:{context.user_id}"


def conversation_memory_id(context: ConversationContext) -> str:
    return f"qq_conversation:{context.conversation_id}"
