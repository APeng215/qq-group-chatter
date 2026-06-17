from __future__ import annotations

import json
import re
from typing import Any

from qq_group_chatter.models import (
    ChatMessage,
    ConversationContext,
    LongTermMemoryOperation,
    LongTermMemoryRecord,
    MemoryKind,
    MemoryMergeAction,
    MemoryScope,
)
from qq_group_chatter.observability import LLM_LATENCY_SECONDS, observe_duration
from qq_group_chatter.prompt_loader import load_prompt


PLANNER_SYSTEM_PROMPT = load_prompt("long_term_memory_planner_system.txt")
PLANNER_PROMPT_TEMPLATE = load_prompt("long_term_memory_planner.txt")
VALID_ACTIONS = {"add", "update", "delete", "skip"}
VALID_SCOPES = {"user", "conversation"}
VALID_KINDS = {
    "identity",
    "preference",
    "constraint",
    "relationship",
    "conversation_rule",
    "other",
}


class LongTermMemoryPlanner:
    def __init__(
        self,
        *,
        llm: Any | None = None,
        min_confidence: float = 0.8,
        max_writable_operations: int = 2,
    ):
        self._llm = llm
        self._min_confidence = min_confidence
        self._max_writable_operations = max_writable_operations

    async def plan(
        self,
        *,
        user_message: str,
        context: ConversationContext,
        user_memories: list[LongTermMemoryRecord],
        conversation_memories: list[LongTermMemoryRecord],
        global_memories: list[LongTermMemoryRecord] | None = None,
        short_term_messages: list[ChatMessage] | None = None,
    ) -> list[LongTermMemoryOperation]:
        if self._llm is None:
            return []
        resolved_global_memories = global_memories or []

        prompt = self._build_prompt(
            user_message=user_message,
            context=context,
            user_memories=user_memories,
            conversation_memories=conversation_memories,
            global_memories=resolved_global_memories,
            short_term_messages=short_term_messages or [],
        )
        with observe_duration(
            metric=LLM_LATENCY_SECONDS,
            labels={"component": "memory_planner"},
            log_name="memory_planner_llm",
            log_fields={"conversation_type": context.conversation_type},
        ):
            try:
                raw = await self._llm.ainvoke(
                    prompt,
                    response_format={"type": "json_object"},
                    system_prompt=PLANNER_SYSTEM_PROMPT,
                    trace_context={
                        "component": "memory_planner",
                        "operation": "plan_memory",
                    },
                )
            except TypeError:
                raw = await self._llm.ainvoke(_combined_prompt(prompt))
        return self._parse_operations(
            raw,
            user_memories=user_memories,
            conversation_memories=conversation_memories,
            global_memories=resolved_global_memories,
        )

    def _build_prompt(
        self,
        *,
        user_message: str,
        context: ConversationContext,
        user_memories: list[LongTermMemoryRecord],
        conversation_memories: list[LongTermMemoryRecord],
        global_memories: list[LongTermMemoryRecord],
        short_term_messages: list[ChatMessage],
    ) -> str:
        return PLANNER_PROMPT_TEMPLATE.format(
            conversation_type=context.conversation_type,
            current_user_qq=context.user_id,
            current_user_nickname=_display_nickname(context.nickname),
            user_message=user_message,
            short_term_history=_format_short_term_history(short_term_messages),
            user_memories_json=_records_json(user_memories),
            conversation_memories_json=_records_json(conversation_memories),
            global_memories_json=_records_json(global_memories),
        )

    def _parse_operations(
        self,
        raw: Any,
        *,
        user_memories: list[LongTermMemoryRecord],
        conversation_memories: list[LongTermMemoryRecord],
        global_memories: list[LongTermMemoryRecord],
    ) -> list[LongTermMemoryOperation]:
        data = _loads_json_object(raw)
        if not isinstance(data, dict):
            return []
        items = data.get("operations")
        if not isinstance(items, list):
            return []

        valid_ids_by_scope = {
            "user": {
                record.id
                for record in [*user_memories, *_records_for_scope(global_memories, "user")]
                if record.id is not None
            },
            "conversation": {
                record.id
                for record in [
                    *conversation_memories,
                    *_records_for_scope(global_memories, "conversation"),
                ]
                if record.id is not None
            },
        }
        operations: list[LongTermMemoryOperation] = []
        writable_count = 0
        for item in items:
            operation = _parse_operation(
                item,
                min_confidence=self._min_confidence,
                valid_ids_by_scope=valid_ids_by_scope,
            )
            if operation is None:
                continue
            if operation.action in {"add", "update", "delete"}:
                if writable_count >= self._max_writable_operations:
                    continue
                writable_count += 1
            operations.append(operation)
        return operations


def _records_json(records: list[LongTermMemoryRecord]) -> str:
    return json.dumps(
        [
            {
                "id": record.id,
                "scope": record.metadata.get("scope"),
                "content": record.content,
                "kind": record.metadata.get("kind"),
            }
            for record in records
        ],
        ensure_ascii=False,
    )


def _format_short_term_history(messages: list[ChatMessage]) -> str:
    if not messages:
        return "无"
    lines = []
    for message in messages:
        speaker = _message_speaker(message)
        lines.append(f"- {speaker} {message.content}")
    return "\n".join(lines)


def _message_speaker(message: ChatMessage) -> str:
    if message.role == "assistant":
        return "[神奈]"
    return f"[QQ:{message.user_id or '未知'} 昵称:{_display_nickname(message.nickname)}]"


def _records_for_scope(
    records: list[LongTermMemoryRecord],
    scope: str,
) -> list[LongTermMemoryRecord]:
    return [record for record in records if record.metadata.get("scope") == scope]


def _combined_prompt(prompt: str) -> str:
    return f"{PLANNER_SYSTEM_PROMPT}\n\n{prompt}"


def _display_nickname(nickname: str | None) -> str:
    if nickname is None:
        return "未设置"
    text = str(nickname).strip()
    return text or "未设置"


def _parse_operation(
    item: Any,
    *,
    min_confidence: float,
    valid_ids_by_scope: dict[str, set[str]],
) -> LongTermMemoryOperation | None:
    if not isinstance(item, dict):
        return None

    action = str(item.get("action", "")).strip().lower()
    scope = str(item.get("scope", "")).strip().lower()
    kind = str(item.get("kind", "")).strip().lower()
    content = str(item.get("content") or "").strip()
    target_id = item.get("target_id")
    try:
        confidence = float(item.get("confidence", 0))
    except (TypeError, ValueError):
        return None

    if action not in VALID_ACTIONS:
        return None
    if scope not in VALID_SCOPES:
        return None
    if kind not in VALID_KINDS:
        return None
    if confidence < min_confidence:
        return None
    if action != "delete" and not content:
        return None
    if action in {"update", "delete"}:
        target_id = str(target_id).strip() if target_id is not None else None
        if not target_id or target_id not in valid_ids_by_scope[scope]:
            return None
    else:
        target_id = str(target_id).strip() if target_id is not None else None

    return LongTermMemoryOperation(
        action=action,  # type: ignore[arg-type]
        scope=scope,  # type: ignore[arg-type]
        target_id=target_id or None,
        content=content,
        kind=kind,  # type: ignore[arg-type]
        confidence=confidence,
    )


def _loads_json_object(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "content"):
        raw = raw.content
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    else:
        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            text = text[first : last + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
