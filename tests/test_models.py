from qq_group_chatter.models import (
    ConversationContext,
    LongTermMemoryBundle,
    build_group_conversation_context,
    build_private_conversation_context,
    conversation_memory_id,
    user_memory_id,
)


def test_builds_group_conversation_context():
    context = build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )

    assert context == ConversationContext(
        conversation_id="qq_group:888888",
        conversation_type="group",
        user_id="123456",
        group_id="888888",
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )
    assert user_memory_id(context) == "qq_user:123456"
    assert conversation_memory_id(context) == "qq_conversation:qq_group:888888"


def test_builds_private_conversation_context():
    context = build_private_conversation_context(
        user_id=123456,
        message_id="m2",
        nickname=None,
        timestamp=456.0,
    )

    assert context.conversation_id == "qq_private:123456"
    assert context.conversation_type == "private"
    assert context.group_id is None
    assert user_memory_id(context) == "qq_user:123456"
    assert conversation_memory_id(context) == "qq_conversation:qq_private:123456"


def test_long_term_memory_bundle_renders_readable_prompt_section():
    bundle = LongTermMemoryBundle(user_memories=[], conversation_memories=["默认中文"])

    assert bundle.as_prompt_section() == (
        "相关个人长期记忆：\n"
        "- 无\n\n"
        "相关会话长期记忆：\n"
        "- 默认中文"
    )
