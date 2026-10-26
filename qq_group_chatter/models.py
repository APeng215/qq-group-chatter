from __future__ import annotations

from dataclasses import dataclass, field
import re
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
class ImageAttachment:
    position: int
    source_url: str | None = field(default=None, repr=False)
    file: str | None = None
    resource_id: str | None = None
    status: str = "pending"
    media_type: str | None = None
    size_bytes: int = 0
    duration_ms: float = 0


@dataclass(frozen=True)
class VisualImage:
    resource_id: str
    media_type: str
    data: bytes = field(repr=False)


@dataclass
class ImageSnapshot:
    enabled: bool = False
    max_images: int = 2
    images: list[VisualImage] = field(default_factory=list)
    attachments: list[dict[str, object]] = field(default_factory=list)

    def render_content(self, content: str, message_id: str | None, *, archive: bool = False) -> str:
        """Render request-local labels without changing persisted message text."""
        entries = sorted(
            (a for a in self.attachments if message_id is not None and a.get("message_id") == message_id),
            key=lambda a: int(a.get("position", 0)),
        )
        labels = []
        for entry in entries:
            status = entry.get("status")
            index = entry.get("input_index")
            if self.enabled and status in {"sent", "sent_duplicate"} and isinstance(index, int) and 1 <= index <= min(len(self.images), self.max_images):
                labels.append(f"[图{index}]")
            else:
                reason = {"disabled": "功能关闭", "request_limit": "超限", "message_limit": "超限",
                          "unavailable": "缓存不可用", "missing_url": "缺少下载地址",
                          "timeout": "下载超时", "too_large": "文件超限",
                          "unsupported_format": "格式不支持"}.get(str(status), "下载失败")
                labels.append(f"[图片未读取：{reason}]")
        # The adapter emits standalone placeholders in attachment order. Only
        # consume as many as this message's actual attachments; preserve prose.
        remaining = iter(labels)
        content = re.sub(r"(?<!\S)\[图片\](?!\S)",
                         lambda match: next(remaining, match.group(0)), content) if entries else content
        if archive and not entries:
            content = content.replace("[图片]", "[历史图片未提供]")
        return " ".join([content, *remaining]).strip()


@dataclass(frozen=True)
class ConversationContext:
    conversation_id: str
    conversation_type: ConversationType
    user_id: str
    group_id: str | None
    message_id: str
    reply_to_message_id: str | None
    nickname: str | None
    timestamp: float
    is_addressed_to_bot: bool = False
    bot_user_id: str | None = None
    bot_nickname: str | None = "神奈"
    round_id: str | None = None


@dataclass(frozen=True)
class ChatMessage:
    conversation_id: str
    role: MessageRole
    content: str
    user_id: str | None
    nickname: str | None
    message_id: str | None
    timestamp: float
    attachments: list[ImageAttachment] = field(default_factory=list)


@dataclass(frozen=True)
class ConversationArchiveRecord:
    content: str
    role: MessageRole
    user_id: str | None
    nickname: str | None
    message_id: str | None
    timestamp: float
    score: float | None = None
    is_semantic_hit: bool = False


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
    target_user_id: str | None = None
    target_nickname: str | None = None


@dataclass(frozen=True)
class LongTermMemoryUsageUpdate:
    scope: MemoryScope
    target_id: str
    confidence: float


@dataclass(frozen=True)
class LongTermMemoryPlanResult:
    operations: list[LongTermMemoryOperation]
    usage_updates: list[LongTermMemoryUsageUpdate] = field(default_factory=list)

    def __iter__(self):
        return iter(self.operations)

    def __len__(self) -> int:
        return len(self.operations)

    def __getitem__(self, index):
        return self.operations[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, list):
            return self.operations == other
        if isinstance(other, LongTermMemoryPlanResult):
            return (
                self.operations == other.operations
                and self.usage_updates == other.usage_updates
            )
        return super().__eq__(other)


@dataclass(frozen=True)
class LongTermMemoryBundle:
    user_memories: list[LongTermMemoryRecord]
    conversation_memories: list[LongTermMemoryRecord]
    global_memories: list[LongTermMemoryRecord] = field(default_factory=list)

    def as_prompt_section(self, context: ConversationContext) -> str:
        sections: list[str] = []
        if self.user_memories:
            sections.append(
                "相关个人长期记忆"
                f"（当前发言者 QQ号：{context.user_id}，昵称：{_display_nickname(context.nickname)}）：\n"
                + "\n".join(_format_memory_record(record) for record in self.user_memories)
            )
        if self.conversation_memories:
            sections.append(
                "相关会话长期记忆：\n"
                + "\n".join(
                    _format_memory_record(record) for record in self.conversation_memories
                )
            )
        if self.global_memories:
            sections.append(
                "当前 conversation 内相关长期记忆"
                "（只作为当前会话背景；不要把其他人的信息误当成当前发言者自己的信息）：\n"
                + "\n".join(
                    _format_global_memory_record(record) for record in self.global_memories
                )
            )
        return "\n\n".join(sections) if sections else "长期记忆：无"


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
    conversation_archive: list[ConversationArchiveRecord] = field(default_factory=list)
    existing_memories: LongTermMemoryBundle | None = None
    assistant_reply: str | None = None
    image_snapshot: ImageSnapshot | None = None
    on_error_notice: Callable[[ErrorNoticeContext], Awaitable[None] | None] | None = None


@dataclass(frozen=True)
class PendingAssistantReply:
    context: ConversationContext
    content: str
    timestamp: float
    user_message: str | None = None
    short_term_messages: list[ChatMessage] = field(default_factory=list)
    conversation_archive: list[ConversationArchiveRecord] = field(default_factory=list)
    long_term_memory: LongTermMemoryBundle | None = None
    memory_warning: ErrorNoticeContext | None = None
    image_snapshot: ImageSnapshot | None = None


def build_group_conversation_context(
    *,
    group_id: str | int,
    user_id: str | int,
    message_id: str | int,
    nickname: str | None,
    timestamp: float,
    is_addressed_to_bot: bool = False,
    reply_to_message_id: str | int | None = None,
    bot_user_id: str | int | None = None,
    bot_nickname: str | None = "神奈",
) -> ConversationContext:
    group = str(group_id)
    return ConversationContext(
        conversation_id=f"qq_group:{group}",
        conversation_type="group",
        user_id=str(user_id),
        group_id=group,
        message_id=str(message_id),
        reply_to_message_id=_optional_text(reply_to_message_id),
        nickname=nickname,
        timestamp=timestamp,
        is_addressed_to_bot=is_addressed_to_bot,
        bot_user_id=_optional_text(bot_user_id),
        bot_nickname=_optional_text(bot_nickname),
    )


def build_private_conversation_context(
    *,
    user_id: str | int,
    message_id: str | int,
    nickname: str | None,
    timestamp: float,
    is_addressed_to_bot: bool = True,
    reply_to_message_id: str | int | None = None,
    bot_user_id: str | int | None = None,
    bot_nickname: str | None = "神奈",
) -> ConversationContext:
    user = str(user_id)
    return ConversationContext(
        conversation_id=f"qq_private:{user}",
        conversation_type="private",
        user_id=user,
        group_id=None,
        message_id=str(message_id),
        reply_to_message_id=_optional_text(reply_to_message_id),
        nickname=nickname,
        timestamp=timestamp,
        is_addressed_to_bot=is_addressed_to_bot,
        bot_user_id=_optional_text(bot_user_id),
        bot_nickname=_optional_text(bot_nickname),
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


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _display_nickname(nickname: str | None) -> str:
    if nickname is None:
        return "未设置"
    text = str(nickname).strip()
    return text or "未设置"
