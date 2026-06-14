from __future__ import annotations

import json
from typing import Any

from qq_group_chatter.models import (
    ConversationContext,
    LongTermMemoryCandidate,
    MemoryKind,
    MemoryScope,
)
from qq_group_chatter.observability import LLM_LATENCY_SECONDS, observe_duration


VALID_SCOPES: set[str] = {"user", "conversation"}
VALID_KINDS: set[str] = {
    "identity",
    "preference",
    "constraint",
    "relationship",
    "conversation_rule",
    "other",
}


class LongTermMemoryExtractor:
    def __init__(self, llm: Any | None = None):
        self._llm = llm

    async def extract(
        self,
        *,
        user_message: str,
        context: ConversationContext,
    ) -> list[LongTermMemoryCandidate]:
        if self._llm is None:
            return []

        prompt = self._build_prompt(user_message=user_message, context=context)
        with observe_duration(
            metric=LLM_LATENCY_SECONDS,
            labels={"component": "memory_extractor"},
            log_name="llm_call",
            log_fields={
                "component": "memory_extractor",
                "conversation_id": context.conversation_id,
                "conversation_type": context.conversation_type,
                "message_id": context.message_id,
            },
        ):
            raw = await self._call_llm(prompt)
        return self._parse_candidates(raw)

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

    def _build_prompt(self, *, user_message: str, context: ConversationContext) -> str:
        return (
            "你是长期记忆提取器。只从用户消息中提取稳定、未来有用的长期记忆。\n"
            "不要提取临时情绪、一次性事件、手机号、密码、token、地址等敏感信息。\n"
            "最多输出 2 条。scope 只能是 user 或 conversation。\n"
            "返回 JSON：{\"memories\":[{\"scope\":\"user\",\"content\":\"...\",\"confidence\":0.9,\"kind\":\"preference\"}]}\n\n"
            f"conversation_type: {context.conversation_type}\n"
            f"user_message: {user_message}"
        )

    def _parse_candidates(self, raw: Any) -> list[LongTermMemoryCandidate]:
        if hasattr(raw, "content"):
            raw = raw.content
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            data = json.loads(raw)
        elif isinstance(raw, dict):
            data = raw
        else:
            raise TypeError(f"unsupported extractor response type: {type(raw)!r}")

        candidates = []
        for item in data.get("memories", []):
            scope = item.get("scope")
            kind = item.get("kind", "other")
            content = str(item.get("content", "")).strip()
            confidence = float(item.get("confidence", 0.0))
            if scope in VALID_SCOPES and kind in VALID_KINDS and content:
                candidates.append(
                    LongTermMemoryCandidate(
                        scope=scope,  # type: ignore[arg-type]
                        content=content,
                        confidence=confidence,
                        kind=kind,  # type: ignore[arg-type]
                    )
                )
        return candidates[:2]

