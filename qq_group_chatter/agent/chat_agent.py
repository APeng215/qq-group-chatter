from __future__ import annotations

from typing import Any

from qq_group_chatter.models import (
    ChatMessage,
    ConversationContext,
    LongTermMemoryBundle,
)
from qq_group_chatter.observability import LLM_LATENCY_SECONDS, observe_duration


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
            return "我现在还没有配置聊天模型。"
        with observe_duration(
            metric=LLM_LATENCY_SECONDS,
            labels={"component": "chat_agent"},
            log_name="llm_call",
            log_fields={
                "component": "chat_agent",
                "conversation_id": context.conversation_id,
                "conversation_type": context.conversation_type,
                "message_id": context.message_id,
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
        return (
            "你是 QQ 聊天机器人。根据当前会话上下文和长期记忆自然回复。\n\n"
            f"conversation_type: {context.conversation_type}\n"
            f"{long_term_memory.as_prompt_section()}\n\n"
            "短期会话上下文：\n"
            f"{history or '无'}\n\n"
            f"当前用户消息：{user_message}"
        )

    def _content(self, raw: Any) -> str:
        if hasattr(raw, "content"):
            return str(raw.content)
        return str(raw)

