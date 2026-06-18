from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Awaitable, Callable
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
    is_addressed_to_bot: bool = False


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
    global_memories: list[LongTermMemoryRecord] = field(default_factory=list)

    def as_prompt_section(self, context: ConversationContext) -> str:
        user_lines = "\n".join(
            _format_memory_record(record) for record in self.user_memories
        ) or "- 无"
        conversation_lines = (
            "\n".join(_format_memory_record(record) for record in self.conversation_memories)
            or "- 无"
        )
        global_lines = (
            "\n".join(_format_global_memory_record(record) for record in self.global_memories)
            or "- 无"
        )
        return LONG_TERM_MEMORY_SECTION_TEMPLATE.format(
            current_user_qq=context.user_id,
            current_user_nickname=_display_nickname(context.nickname),
            user_memory_lines=user_lines,
            conversation_memory_lines=conversation_lines,
            global_memory_lines=global_lines,
        )


@dataclass(frozen=True)
class ErrorNoticeContext:
    stage: str
    error_type: str
    impact: str


@dataclass(frozen=True)
class LongTermMemoryIngestionJob:
    context: ConversationContext
    user_message: str
    short_term_messages: list[ChatMessage] = field(default_factory=list)
    existing_memories: LongTermMemoryBundle | None = None
    assistant_reply: str | None = None
    on_error_notice: Callable[[ErrorNoticeContext], Awaitable[None] | None] | None = None


@dataclass(frozen=True)
class PendingAssistantReply:
    context: ConversationContext
    content: str
    timestamp: float
    user_message: str | None = None
    short_term_messages: list[ChatMessage] = field(default_factory=list)
    long_term_memory: LongTermMemoryBundle | None = None
    memory_warning: ErrorNoticeContext | None = None


def build_group_conversation_context(
    *,
    group_id: str | int,
    user_id: str | int,
    message_id: str | int,
    nickname: str | None,
    timestamp: float,
    is_addressed_to_bot: bool = False,
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
        is_addressed_to_bot=is_addressed_to_bot,
    )


def build_private_conversation_context(
    *,
    user_id: str | int,
    message_id: str | int,
    nickname: str | None,
    timestamp: float,
    is_addressed_to_bot: bool = True,
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
        is_addressed_to_bot=is_addressed_to_bot,
    )


def user_memory_id(context: ConversationContext) -> str:
    return f"qq_user:{context.conversation_id}:{context.user_id}"


def conversation_memory_id(context: ConversationContext) -> str:
    return f"qq_conversation:{context.conversation_id}"


def _format_memory_record(record: LongTermMemoryRecord) -> str:
    return f"- {record.content}"


def _format_global_memory_record(record: LongTermMemoryRecord) -> str:
    metadata = record.metadata
    source = "；".join(
        (
            f"scope={_metadata_text(metadata.get('scope'))}",
            f"source_user_id={_metadata_text(metadata.get('source_user_id'))}",
            f"source_nickname={_metadata_text(metadata.get('source_nickname'))}",
            f"conversation_id={_metadata_text(metadata.get('conversation_id'))}",
            f"conversation_type={_metadata_text(metadata.get('conversation_type'))}",
            f"kind={_metadata_text(metadata.get('kind'))}",
        )
    )
    return f"- {record.content}\n  来源：{source}"


def _metadata_text(value: object) -> str:
    if value is None:
        return "未知"
    text = str(value).strip()
    return text or "未知"


def _display_nickname(nickname: str | None) -> str:
    if nickname is None:
        return "未设置"
    text = str(nickname).strip()
    return text or "未设置"
