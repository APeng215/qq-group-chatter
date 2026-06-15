from __future__ import annotations

import json
from typing import Any

from qq_group_chatter.models import ConversationContext, LongTermMemoryCandidate
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
            "\u4f60\u662f\u957f\u671f\u8bb0\u5fc6\u63d0\u53d6\u5668\u3002"
            "\u53ea\u4ece\u7528\u6237\u6d88\u606f\u4e2d\u63d0\u53d6"
            "\u7a33\u5b9a\u3001\u672a\u6765\u6709\u7528\u7684\u957f\u671f\u8bb0\u5fc6\u3002\n"
            "\u4e0d\u8981\u63d0\u53d6\u4e34\u65f6\u60c5\u7eea\u3001\u4e00\u6b21\u6027\u4e8b\u4ef6\u3001"
            "\u624b\u673a\u53f7\u3001\u5bc6\u7801\u3001token\u3001\u5730\u5740\u7b49\u654f\u611f\u4fe1\u606f\u3002\n"
            "\u6700\u591a\u8f93\u51fa 2 \u6761\u3002scope \u53ea\u80fd\u662f user \u6216 conversation\u3002\n"
            "\u53ea\u8fd4\u56de JSON\uff0c\u683c\u5f0f\uff1a"
            '{"memories":[{"scope":"user","content":"...","confidence":0.9,"kind":"preference"}]}'
            "\n\n"
            f"conversation_type: {context.conversation_type}\n"
            f"user_message: {user_message}"
        )

    def _parse_candidates(self, raw: Any) -> list[LongTermMemoryCandidate]:
        if hasattr(raw, "content"):
            raw = raw.content
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            data = _loads_json_object(raw)
        elif isinstance(raw, dict):
            data = raw
        else:
            raise TypeError(f"unsupported extractor response type: {type(raw)!r}")

        candidates = []
        for item in data.get("memories", []):
            candidate = _parse_candidate(item)
            if candidate is not None:
                candidates.append(candidate)
        return candidates[:2]


def _loads_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                data, _ = decoder.raw_decode(text[index:])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise

    if not isinstance(data, dict):
        raise TypeError(f"extractor JSON must be an object, got {type(data)!r}")
    return data


def _parse_candidate(item: Any) -> LongTermMemoryCandidate | None:
    if not isinstance(item, dict):
        return None

    scope = item.get("scope")
    kind = item.get("kind", "other")
    content = str(item.get("content", "")).strip()
    try:
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None

    if scope not in VALID_SCOPES or kind not in VALID_KINDS or not content:
        return None
    return LongTermMemoryCandidate(
        scope=scope,  # type: ignore[arg-type]
        content=content,
        confidence=confidence,
        kind=kind,  # type: ignore[arg-type]
    )
