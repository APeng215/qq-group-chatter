import uuid
from pathlib import Path

from qq_group_chatter.models import ChatMessage
from qq_group_chatter.services.short_term_memory import ShortTermMemoryService


def memory_path(name: str) -> Path:
    path = Path("tests/.tmp/short-term-memory") / f"{name}-{uuid.uuid4().hex}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def message(conversation_id: str, content: str, role: str = "user") -> ChatMessage:
    return ChatMessage(
        conversation_id=conversation_id,
        role=role,
        content=content,
        user_id="u1" if role == "user" else None,
        nickname="阿咳" if role == "user" else None,
        message_id=content,
        timestamp=1.0,
    )


async def test_keeps_only_recent_messages_per_conversation():
    service = ShortTermMemoryService(max_messages_per_conversation=2)

    await service.add_message(message("qq_group:1", "one"))
    await service.add_message(message("qq_group:1", "two"))
    await service.add_message(message("qq_group:1", "three", role="assistant"))

    recent = await service.get_recent("qq_group:1", limit=10)

    assert [item.content for item in recent] == ["two", "three"]
    assert [item.role for item in recent] == ["user", "assistant"]


async def test_isolates_group_and_private_conversations():
    service = ShortTermMemoryService(max_messages_per_conversation=10)

    await service.add_message(message("qq_group:1", "group"))
    await service.add_message(message("qq_private:u1", "private"))

    assert [item.content for item in await service.get_recent("qq_group:1")] == ["group"]
    assert [item.content for item in await service.get_recent("qq_private:u1")] == ["private"]


async def test_persists_and_restores_recent_messages_per_conversation():
    path = memory_path("restore")
    service = ShortTermMemoryService(
        max_messages_per_conversation=3,
        path=path,
    )

    for index in range(5):
        await service.add_message(message("qq_group:1", f"group-{index}"))
    await service.add_message(message("qq_private:u1", "private"))

    restored = ShortTermMemoryService(
        max_messages_per_conversation=3,
        path=path,
    )

    assert [item.content for item in await restored.get_recent("qq_group:1", limit=10)] == [
        "group-2",
        "group-3",
        "group-4",
    ]
    assert [item.content for item in await restored.get_recent("qq_group:1", limit=2)] == [
        "group-3",
        "group-4",
    ]
    assert [item.content for item in await restored.get_recent("qq_private:u1", limit=10)] == [
        "private"
    ]


async def test_clear_removes_conversation_from_memory_and_disk():
    path = memory_path("clear")
    service = ShortTermMemoryService(
        max_messages_per_conversation=10,
        path=path,
    )
    await service.add_message(message("qq_group:1", "group"))
    await service.add_message(message("qq_private:u1", "private"))

    await service.clear("qq_group:1")
    restored = ShortTermMemoryService(
        max_messages_per_conversation=10,
        path=path,
    )

    assert await restored.get_recent("qq_group:1", limit=10) == []
    assert [item.content for item in await restored.get_recent("qq_private:u1", limit=10)] == [
        "private"
    ]
