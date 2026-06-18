from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal
from typing import Any

from qq_group_chatter.models import (
    ChatMessage,
    ConversationArchiveRecord,
    ConversationContext,
    ErrorNoticeContext,
    LongTermMemoryBundle,
)
from qq_group_chatter.observability import (
    LLM_LATENCY_SECONDS,
    conversation_log_fields,
    observe_duration,
)
from qq_group_chatter.agent.identity import BOT_IDENTITY_PROMPT
from qq_group_chatter.prompt_loader import load_prompt
from qq_group_chatter.time_utils import current_time_text, format_time_text


CHAT_CONTEXT_RULES = load_prompt("chat_context_rules.txt")
DEFAULT_CHAT_SYSTEM_PREFIX = load_prompt("deepseek_system.txt").format(
    bot_identity_prompt=BOT_IDENTITY_PROMPT,
)
CHAT_AGENT_PROMPT_TEMPLATE = load_prompt("chat_agent.txt")
CHAT_AGENT_SYSTEM_PROMPT = "\n".join(
    (
        DEFAULT_CHAT_SYSTEM_PREFIX,
        load_prompt("chat_agent_system.txt").format(
            chat_context_rules=CHAT_CONTEXT_RULES,
        ),
    )
)
CHAT_SEARCH_GROUNDED_PROMPT_TEMPLATE = load_prompt("chat_search_grounded.txt")
CHAT_SEARCH_GROUNDED_SYSTEM_PROMPT = "\n".join(
    (
        DEFAULT_CHAT_SYSTEM_PREFIX,
        load_prompt("chat_search_grounded_system.txt").format(
            chat_context_rules=CHAT_CONTEXT_RULES,
        ),
    )
)
ERROR_NOTICE_PROMPT_TEMPLATE = load_prompt("memory_error_notice.txt")


@dataclass(frozen=True)
class ChatReplyDecision:
    content: str


@dataclass(frozen=True)
class WebSearchDecision:
    notice: str
    query: str


ChatDecision = ChatReplyDecision | WebSearchDecision


def parse_chat_decision(raw: str) -> ChatDecision | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    action = data.get("action")
    if action == "reply":
        if set(data) != {"action", "content"}:
            return None
        content = _clean_field(data.get("content"))
        if not content:
            return None
        return ChatReplyDecision(content=content)
    if action == "web_search":
        if set(data) != {"action", "notice", "query"}:
            return None
        notice = _clean_field(data.get("notice"))
        query = _clean_field(data.get("query"))
        if not notice or not query:
            return None
        if _looks_like_template_leak(notice) or _looks_like_template_leak(query):
            return None
        return WebSearchDecision(notice=notice, query=query)
    return None


def _clean_field(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip()


def _looks_like_template_leak(value: str) -> bool:
    if "<" in value or ">" in value:
        return True
    return any(
        marker in value
        for marker in (
            "神奈要先发给对方的等待提示",
            "适合搜索的查询词",
        )
    )


class ChatAgent:
    def __init__(self, llm: Any | None = None):
        self._llm = llm

    async def generate_reply(
        self,
        *,
        user_message: str,
        context: ConversationContext,
        short_term_messages: list[ChatMessage],
        long_term_memory: LongTermMemoryBundle,
        conversation_archive: list[ConversationArchiveRecord] | None = None,
        memory_warning: ErrorNoticeContext | None = None,
    ) -> ChatDecision:
        prompt = self._build_prompt(
            user_message=user_message,
            context=context,
            short_term_messages=short_term_messages,
            long_term_memory=long_term_memory,
            conversation_archive=conversation_archive or [],
            memory_warning=memory_warning,
        )
        if self._llm is None:
            return ChatReplyDecision(content="\u6211\u73b0\u5728\u8fd8\u6ca1\u6709\u914d\u7f6e\u804a\u5929\u6a21\u578b\u3002")
        with observe_duration(
            metric=LLM_LATENCY_SECONDS,
            labels={"component": "chat_agent"},
            log_name="llm_call",
            log_fields={
                "component": "chat_agent",
                **conversation_log_fields(context),
            },
        ):
            raw = await self._call_llm(
                prompt,
                response_format={"type": "json_object"},
                system_prompt=CHAT_AGENT_SYSTEM_PROMPT,
                trace_context={
                    "component": "chat_agent",
                    "operation": "decision",
                },
            )
        decision = parse_chat_decision(self._content(raw))
        if decision is None:
            return ChatReplyDecision(content="我刚刚没能整理好回复，稍后再试。")
        return decision

    async def generate_grounded_search_reply(
        self,
        *,
        user_message: str,
        search_query: str,
        search_sources: list[Any],
        context: ConversationContext,
        short_term_messages: list[ChatMessage],
        long_term_memory: LongTermMemoryBundle,
        conversation_archive: list[ConversationArchiveRecord] | None = None,
    ) -> str:
        prompt = self._build_grounded_search_prompt(
            user_message=user_message,
            search_query=search_query,
            search_sources=search_sources,
            context=context,
            short_term_messages=short_term_messages,
            long_term_memory=long_term_memory,
            conversation_archive=conversation_archive or [],
        )
        if self._llm is None:
            return "我搜到了一些资料，但现在还没有配置聊天模型来整理成回复。"
        with observe_duration(
            metric=LLM_LATENCY_SECONDS,
            labels={"component": "chat_agent"},
            log_name="llm_call",
            log_fields={
                "component": "chat_agent",
                **conversation_log_fields(context),
            },
        ):
            raw = await self._call_llm(
                prompt,
                system_prompt=CHAT_SEARCH_GROUNDED_SYSTEM_PROMPT,
                trace_context={
                    "component": "chat_agent",
                    "operation": "grounded_search_reply",
                },
            )
        return self._content(raw)

    async def generate_error_notice(
        self,
        *,
        error_context: ErrorNoticeContext,
        context: ConversationContext,
    ) -> str:
        if self._llm is None:
            return _fallback_error_notice(error_context)
        prompt = ERROR_NOTICE_PROMPT_TEMPLATE.format(
            conversation_type=context.conversation_type,
            current_speaker=_format_current_speaker(context),
            error_stage=error_context.stage,
            error_type=error_context.error_type,
            error_impact=error_context.impact,
        )
        with observe_duration(
            metric=LLM_LATENCY_SECONDS,
            labels={"component": "chat_agent"},
            log_name="llm_call",
            log_fields={
                "component": "chat_agent",
                **conversation_log_fields(context),
            },
        ):
            raw = await self._call_llm(
                prompt,
                system_prompt=CHAT_AGENT_SYSTEM_PROMPT,
                trace_context={
                    "component": "chat_agent",
                    "operation": "memory_error_notice",
                },
            )
        return self._content(raw).strip() or _fallback_error_notice(error_context)

    async def _call_llm(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        trace_context: dict[str, str] | None = None,
    ) -> Any:
        if hasattr(self._llm, "ainvoke"):
            if response_format is not None:
                try:
                    return await self._llm.ainvoke(
                        prompt,
                        response_format=response_format,
                        system_prompt=system_prompt,
                        trace_context=trace_context,
                    )
                except TypeError:
                    pass
            try:
                return await self._llm.ainvoke(
                    prompt,
                    system_prompt=system_prompt,
                    trace_context=trace_context,
                )
            except TypeError:
                return await self._llm.ainvoke(prompt)
        if hasattr(self._llm, "invoke"):
            if response_format is not None:
                try:
                    return self._llm.invoke(
                        prompt,
                        response_format=response_format,
                        system_prompt=system_prompt,
                        trace_context=trace_context,
                    )
                except TypeError:
                    pass
            try:
                return self._llm.invoke(
                    prompt,
                    system_prompt=system_prompt,
                    trace_context=trace_context,
                )
            except TypeError:
                return self._llm.invoke(prompt)
        if callable(self._llm):
            if response_format is not None:
                try:
                    result = self._llm(
                        prompt,
                        response_format=response_format,
                        system_prompt=system_prompt,
                        trace_context=trace_context,
                    )
                except TypeError:
                    result = self._llm(prompt)
            else:
                try:
                    result = self._llm(
                        prompt,
                        system_prompt=system_prompt,
                        trace_context=trace_context,
                    )
                except TypeError:
                    result = self._llm(prompt)
            if hasattr(result, "__await__"):
                return await result
            return result
        raise TypeError("llm must be callable or expose invoke/ainvoke")

    def _build_prompt(
        self,
        *,
        user_message: str,
        context: ConversationContext,
        short_term_messages: list[ChatMessage],
        long_term_memory: LongTermMemoryBundle,
        conversation_archive: list[ConversationArchiveRecord] | None = None,
        memory_warning: ErrorNoticeContext | None = None,
    ) -> str:
        history_messages = _history_without_current_message(short_term_messages, context)
        history = _format_short_term_history(history_messages)
        prompt = CHAT_AGENT_PROMPT_TEMPLATE.format(
            current_time=current_time_text(),
            conversation_type=context.conversation_type,
            long_term_memory_section=long_term_memory.as_prompt_section(context),
            conversation_archive_section=_format_conversation_archive(
                conversation_archive or []
            ),
            memory_warning=_format_memory_warning(memory_warning),
            short_term_history=history or "\u65e0",
            current_speaker=_format_current_speaker(context),
            addressed_to_bot=_format_addressed_to_bot(context),
            quoted_message=_format_quoted_message(context, short_term_messages),
            user_message=_format_current_user_message(context, user_message),
        )
        return _compact_blank_lines(prompt)

    def _build_grounded_search_prompt(
        self,
        *,
        user_message: str,
        search_query: str,
        search_sources: list[Any],
        context: ConversationContext,
        short_term_messages: list[ChatMessage],
        long_term_memory: LongTermMemoryBundle,
        conversation_archive: list[ConversationArchiveRecord] | None = None,
    ) -> str:
        history_messages = _history_without_current_message(short_term_messages, context)
        history = _format_short_term_history(history_messages)
        prompt = CHAT_SEARCH_GROUNDED_PROMPT_TEMPLATE.format(
            current_time=current_time_text(),
            conversation_type=context.conversation_type,
            long_term_memory_section=long_term_memory.as_prompt_section(context),
            conversation_archive_section=_format_conversation_archive(
                conversation_archive or []
            ),
            short_term_history=history or "\u65e0",
            current_speaker=_format_current_speaker(context),
            addressed_to_bot=_format_addressed_to_bot(context),
            quoted_message=_format_quoted_message(context, short_term_messages),
            user_message=_format_current_user_message(context, user_message),
            search_query=search_query,
            search_sources=_format_search_sources(search_sources),
        )
        return _compact_blank_lines(prompt)

    def _content(self, raw: Any) -> str:
        if hasattr(raw, "content"):
            return str(raw.content)
        return str(raw)


def _format_short_term_history(messages: list[ChatMessage]) -> str:
    lines = []
    for item in messages:
        speaker = _format_message_speaker(item)
        timestamp = format_time_text(item.timestamp)
        if timestamp is None:
            lines.append(f"{speaker} {item.content}")
        else:
            lines.append(f"[{timestamp}] {speaker} {item.content}")
    return "\n".join(lines)


def _format_conversation_archive(records: list[ConversationArchiveRecord]) -> str:
    if not records:
        return ""
    lines = [
        "相关历史对话（语义召回，仅表示过去说过，不代表当前事实仍成立）："
    ]
    for record in records:
        speaker = _format_archive_speaker(record)
        timestamp = format_time_text(record.timestamp)
        if timestamp is None:
            lines.append(f"- {speaker} {record.content}")
        else:
            lines.append(f"- [{timestamp}] {speaker} {record.content}")
    return "\n".join(lines)


def _format_archive_speaker(record: ConversationArchiveRecord) -> str:
    if record.role == "assistant":
        return "[神奈]"
    return _format_user_identity(record.user_id or "未知", record.nickname)


def _history_without_current_message(
    messages: list[ChatMessage],
    context: ConversationContext,
) -> list[ChatMessage]:
    return [message for message in messages if message.message_id != context.message_id]


def _format_current_speaker(context: ConversationContext) -> str:
    return f"- QQ号：{context.user_id}\n- 昵称：{_display_nickname(context.nickname)}"


def _format_addressed_to_bot(context: ConversationContext) -> str:
    if context.is_addressed_to_bot:
        return "已明确指向神奈"
    return "未明确指向神奈，仅作为会话背景"


def _format_quoted_message(
    context: ConversationContext,
    short_term_messages: list[ChatMessage],
) -> str:
    if not context.reply_to_message_id:
        return ""
    quoted = _find_message_by_id(short_term_messages, context.reply_to_message_id)
    if quoted is None:
        return (
            "当前消息引用：\n"
            f"- 引用消息 ID：{context.reply_to_message_id}\n"
            "- 引用原文：短期上下文中未找到"
        )
    speaker = _format_message_speaker(quoted)
    return (
        "当前消息引用：\n"
        f"- 引用消息 ID：{context.reply_to_message_id}\n"
        f"- 引用原文：{speaker} {quoted.content}"
    )


def _find_message_by_id(
    messages: list[ChatMessage],
    message_id: str,
) -> ChatMessage | None:
    for message in reversed(messages):
        if message.message_id == message_id:
            return message
    return None


def _format_current_user_message(context: ConversationContext, user_message: str) -> str:
    return f"{_format_user_identity(context.user_id, context.nickname)} {user_message}"


def _format_message_speaker(message: ChatMessage) -> str:
    if message.role == "assistant":
        return "[神奈]"
    return _format_user_identity(message.user_id or "未知", message.nickname)


def _format_user_identity(user_id: str, nickname: str | None) -> str:
    return f"[QQ:{user_id} 昵称:{_display_nickname(nickname)}]"


def _display_nickname(nickname: str | None) -> str:
    if nickname is None:
        return "未设置"
    text = str(nickname).strip()
    return text or "未设置"


def _format_memory_warning(memory_warning: ErrorNoticeContext | None) -> str:
    if memory_warning is None:
        return ""
    return (
        "记忆状态提示（请在回复中自然、简短地说明影响；不要暴露内部错误细节）：\n"
        f"- stage: {memory_warning.stage}\n"
        f"- error_type: {memory_warning.error_type}\n"
        f"- impact: {memory_warning.impact}\n"
        "- 需要用神奈的口吻自然提醒，不要暴露内部堆栈、密钥或实现细节。"
    )


def _compact_blank_lines(value: str) -> str:
    lines = value.splitlines()
    compacted: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and compacted and compacted[-1].startswith("当前消息指向："):
            continue
        if is_blank and previous_blank:
            continue
        compacted.append(line)
        previous_blank = is_blank
    return "\n".join(compacted).strip()


def _fallback_error_notice(error_context: ErrorNoticeContext) -> str:
    if error_context.stage == "long_term_memory_search":
        return "记忆好像出了点小问题，这次我可能没用上以前记住的内容。"
    return "记忆好像出了点小问题，刚刚这条我可能没能记下来。"


def _format_search_sources(search_sources: list[Any]) -> str:
    blocks = []
    for index, source in enumerate(search_sources, start=1):
        title = str(getattr(source, "title", "") or "无标题")
        content = str(getattr(source, "content", "") or "无摘要")
        raw_content = str(getattr(source, "raw_content", "") or "")
        blocks.append(
            f"[来源 {index}]\n"
            f"标题: {title}\n"
            f"摘要: {content}\n"
            f"原网页正文:\n{raw_content}"
        )
    return "\n\n".join(blocks) or "无"
