from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path

from qq_group_chatter.models import ChatMessage


class ShortTermMemoryService:
    def __init__(
        self,
        max_messages_per_conversation: int = 300,
        path: str | Path | None = None,
    ):
        self._max_messages_per_conversation = max(1, int(max_messages_per_conversation))
        self._path = Path(path) if path is not None else None
        self._histories: dict[str, deque[ChatMessage]] = defaultdict(
            lambda: deque(maxlen=self._max_messages_per_conversation)
        )
        self._lock = asyncio.Lock()
        self._load()

    async def add_message(self, message: ChatMessage) -> None:
        async with self._lock:
            self._histories[message.conversation_id].append(message)
            self._persist_locked()

    async def get_recent(self, conversation_id: str, limit: int = 20) -> list[ChatMessage]:
        async with self._lock:
            return list(self._histories[conversation_id])[-limit:]

    async def clear(self, conversation_id: str) -> None:
        async with self._lock:
            self._histories.pop(conversation_id, None)
            self._persist_locked()

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        for raw_line in self._path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            message = _loads_message(line)
            if message is None:
                continue
            self._histories[message.conversation_id].append(message)

    def _persist_locked(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for messages in self._histories.values():
            lines.extend(
                json.dumps(asdict(message), ensure_ascii=False, default=str)
                for message in messages
            )
        self._path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _loads_message(line: str) -> ChatMessage | None:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return ChatMessage(
            conversation_id=str(data["conversation_id"]),
            role=data["role"],
            content=str(data["content"]),
            user_id=_optional_text(data.get("user_id")),
            nickname=_optional_text(data.get("nickname")),
            message_id=_optional_text(data.get("message_id")),
            timestamp=float(data["timestamp"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
