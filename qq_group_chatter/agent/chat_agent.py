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
            return "\u6211\u73b0\u5728\u8fd8\u6ca1\u6709\u914d\u7f6e\u804a\u5929\u6a21\u578b\u3002"
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
            "\u4f60\u662f QQ \u804a\u5929\u673a\u5668\u4eba\u3002"
            "\u6839\u636e\u5f53\u524d\u4f1a\u8bdd\u4e0a\u4e0b\u6587\u548c"
            "\u957f\u671f\u8bb0\u5fc6\u81ea\u7136\u56de\u590d\u3002\n\n"
            f"conversation_type: {context.conversation_type}\n"
            f"{long_term_memory.as_prompt_section()}\n\n"
            "\u77ed\u671f\u4f1a\u8bdd\u4e0a\u4e0b\u6587\uff1a\n"
            f"{history or '\u65e0'}\n\n"
            f"\u5f53\u524d\u7528\u6237\u6d88\u606f\uff1a{user_message}"
        )

    def _content(self, raw: Any) -> str:
        if hasattr(raw, "content"):
            return str(raw.content)
        return str(raw)

