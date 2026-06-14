from __future__ import annotations

import asyncio
from collections import defaultdict, deque

from qq_group_chatter.models import ChatMessage


class ShortTermMemoryService:
    def __init__(self, max_messages_per_conversation: int = 30):
        self._histories: dict[str, deque[ChatMessage]] = defaultdict(
            lambda: deque(maxlen=max_messages_per_conversation)
        )
        self._lock = asyncio.Lock()

    async def add_message(self, message: ChatMessage) -> None:
        async with self._lock:
            self._histories[message.conversation_id].append(message)

    async def get_recent(self, conversation_id: str, limit: int = 20) -> list[ChatMessage]:
        async with self._lock:
            return list(self._histories[conversation_id])[-limit:]

    async def clear(self, conversation_id: str) -> None:
        async with self._lock:
            self._histories.pop(conversation_id, None)

