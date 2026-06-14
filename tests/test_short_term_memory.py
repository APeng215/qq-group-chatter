from qq_group_chatter.models import ChatMessage
from qq_group_chatter.services.short_term_memory import ShortTermMemoryService


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
