from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qq_group_chatter.prompt_loader import load_prompt


ConversationType = Literal["group", "private"]
MessageRole = Literal["user", "assistant"]
MemoryScope = Literal["user", "conversation"]
MemoryMergeAction = Literal["add", "update", "delete", "skip"]
MemoryKind = Literal[
    "identity",
    "preference",
    "constraint",
    "relationship",
    "conversation_rule",
    "other",
]

LONG_TERM_MEMORY_SECTION_TEMPLATE = load_prompt("long_term_memory_section.txt")


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
class LongTermMemoryRecord:
    id: str | None
    content: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class LongTermMemoryOperation:
    action: MemoryMergeAction
    scope: MemoryScope
    target_id: str | None
    content: str
    kind: MemoryKind
    confidence: float


@dataclass(frozen=True)
class LongTermMemoryBundle:
    user_memories: list[LongTermMemoryRecord]
    conversation_memories: list[LongTermMemoryRecord]

    def as_prompt_section(self) -> str:
        user_lines = "\n".join(
            f"- {record.content}" for record in self.user_memories
        ) or "- 无"
        conversation_lines = (
            "\n".join(f"- {record.content}" for record in self.conversation_memories)
            or "- 无"
        )
        return LONG_TERM_MEMORY_SECTION_TEMPLATE.format(
            user_memory_lines=user_lines,
            conversation_memory_lines=conversation_lines,
        )


@dataclass(frozen=True)
class LongTermMemoryIngestionJob:
    context: ConversationContext
    user_message: str
    existing_memories: LongTermMemoryBundle | None = None


@dataclass(frozen=True)
class PendingAssistantReply:
    context: ConversationContext
    content: str
    timestamp: float


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
