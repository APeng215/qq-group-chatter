from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qq_group_chatter.models import (
    ChatMessage,
    ConversationContext,
    LongTermMemoryBundle,
)
from qq_group_chatter.agent.identity import BOT_IDENTITY_PROMPT
from qq_group_chatter.observability import (
    LLM_LATENCY_SECONDS,
    conversation_log_fields,
    observe_duration,
)
from qq_group_chatter.prompt_loader import load_prompt


CHAT_AGENT_PROMPT_TEMPLATE = load_prompt("chat_agent.txt")
CHAT_SEARCH_GROUNDED_PROMPT_TEMPLATE = load_prompt("chat_search_grounded.txt")
WEB_SEARCH_REQUEST_MARKER = "__NEED_WEB_SEARCH__"


@dataclass(frozen=True)
class WebSearchRequest:
    notice: str
    query: str


def parse_web_search_request(reply: str) -> WebSearchRequest | None:
    lines = reply.strip().splitlines()
    if len(lines) != 3:
        return None
    if lines[0].strip() != WEB_SEARCH_REQUEST_MARKER:
        return None

    notice_prefix = "提示:"
    query_prefix = "查询:"
    notice_line = lines[1].strip()
    query_line = lines[2].strip()
    if not notice_line.startswith(notice_prefix):
        return None
    if not query_line.startswith(query_prefix):
        return None

    notice = notice_line[len(notice_prefix) :].strip()
    query = query_line[len(query_prefix) :].strip()
    if not notice or not query:
        return None
    return WebSearchRequest(notice=notice, query=query)


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
    ) -> str:
        prompt = self._build_prompt(
            user_message=user_message,
            context=context,
            short_term_messages=short_term_messages,
            long_term_memory=long_term_memory,
        )
        if self._llm is None:
            return "\u6211\u73b0\u5728\u8fd8\u6ca1\u6709\u914d\u7f6e\u804a\u5929\u6a21\u578b\u3002"
        with observe_duration(
            metric=LLM_LATENCY_SECONDS,
            labels={"component": "chat_agent"},
            log_name="llm_call",
            log_fields={
                "component": "chat_agent",
                **conversation_log_fields(context),
            },
        ):
            raw = await self._call_llm(prompt)
        return self._content(raw)

    async def generate_grounded_search_reply(
        self,
        *,
        user_message: str,
        search_query: str,
        search_sources: list[Any],
        context: ConversationContext,
        short_term_messages: list[ChatMessage],
        long_term_memory: LongTermMemoryBundle,
    ) -> str:
        prompt = self._build_grounded_search_prompt(
            user_message=user_message,
            search_query=search_query,
            search_sources=search_sources,
            context=context,
            short_term_messages=short_term_messages,
            long_term_memory=long_term_memory,
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
            raw = await self._call_llm(prompt)
        return self._content(raw)

    async def _call_llm(self, prompt: str) -> Any:
        if hasattr(self._llm, "ainvoke"):
            return await self._llm.ainvoke(prompt)
        if hasattr(self._llm, "invoke"):
            return self._llm.invoke(prompt)
        if callable(self._llm):
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
    ) -> str:
        history = "\n".join(
            f"{item.nickname or item.role}: {item.content}" for item in short_term_messages
        )
        return CHAT_AGENT_PROMPT_TEMPLATE.format(
            bot_identity_prompt=BOT_IDENTITY_PROMPT,
            conversation_type=context.conversation_type,
            long_term_memory_section=long_term_memory.as_prompt_section(),
            short_term_history=history or "\u65e0",
            user_message=user_message,
        )

    def _build_grounded_search_prompt(
        self,
        *,
        user_message: str,
        search_query: str,
        search_sources: list[Any],
        context: ConversationContext,
        short_term_messages: list[ChatMessage],
        long_term_memory: LongTermMemoryBundle,
    ) -> str:
        history = "\n".join(
            f"{item.nickname or item.role}: {item.content}" for item in short_term_messages
        )
        return CHAT_SEARCH_GROUNDED_PROMPT_TEMPLATE.format(
            bot_identity_prompt=BOT_IDENTITY_PROMPT,
            conversation_type=context.conversation_type,
            long_term_memory_section=long_term_memory.as_prompt_section(),
            short_term_history=history or "\u65e0",
            user_message=user_message,
            search_query=search_query,
            search_sources=_format_search_sources(search_sources),
        )

    def _content(self, raw: Any) -> str:
        if hasattr(raw, "content"):
            return str(raw.content)
        return str(raw)


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
