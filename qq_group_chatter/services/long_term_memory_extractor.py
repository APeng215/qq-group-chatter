from __future__ import annotations

import json
from typing import Any

from qq_group_chatter.models import ConversationContext, LongTermMemoryCandidate
from qq_group_chatter.observability import LLM_LATENCY_SECONDS, observe_duration
from qq_group_chatter.prompt_loader import load_prompt


VALID_SCOPES: set[str] = {"user", "conversation"}
VALID_KINDS: set[str] = {
    "identity",
    "preference",
    "constraint",
    "relationship",
    "conversation_rule",
    "other",
}
EXTRACTOR_PROMPT_TEMPLATE = load_prompt("long_term_memory_extractor.txt")


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
        return EXTRACTOR_PROMPT_TEMPLATE.format(
            conversation_type=context.conversation_type,
            user_message=user_message,
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
