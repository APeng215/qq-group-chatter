from qq_group_chatter.models import (
    ConversationContext,
    LongTermMemoryBundle,
    LongTermMemoryRecord,
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
    bundle = LongTermMemoryBundle(
        user_memories=[],
        conversation_memories=[
            LongTermMemoryRecord(
                id=None,
                content="默认中文",
                metadata={"private": "不应出现在 prompt"},
            )
        ],
    )

    assert bundle.as_prompt_section() == (
        "相关个人长期记忆：\n"
        "- 无\n\n"
        "相关会话长期记忆：\n"
        "- 默认中文"
    )
    assert "不应出现在 prompt" not in bundle.as_prompt_section()


def test_long_term_memory_bundle_does_not_render_record_ids_or_metadata():
    bundle = LongTermMemoryBundle(
        user_memories=[
            LongTermMemoryRecord(
                id="mem-user-1",
                content="用户不吃辣",
                metadata={"source": "mem0"},
            )
        ],
        conversation_memories=[
            LongTermMemoryRecord(
                id="mem-conv-1",
                content="当前会话默认中文",
                metadata={"scope": "conversation"},
            )
        ],
    )

    prompt_section = bundle.as_prompt_section()

    assert "- 用户不吃辣" in prompt_section
    assert "- 当前会话默认中文" in prompt_section
    assert "mem-user-1" not in prompt_section
    assert "mem-conv-1" not in prompt_section
    assert "mem0" not in prompt_section
    assert "conversation" not in prompt_section


def test_long_term_memory_bundle_renders_memory_times_as_local_time():
    bundle = LongTermMemoryBundle(
        user_memories=[
            LongTermMemoryRecord(
                id="mem-user-1",
                content="用户不吃辣",
                metadata={
                    "source_created_at": 1781529229.0,
                    "last_seen_at": 1781531640.0,
                },
            )
        ],
        conversation_memories=[
            LongTermMemoryRecord(
                id="mem-conv-1",
                content="当前会话默认中文",
                metadata={
                    "source_created_at": "2026-06-15T13:13:49Z",
                    "last_seen_at": "2026-06-15T13:54:00Z",
                },
            )
        ],
    )

    prompt_section = bundle.as_prompt_section()

    assert "- 用户不吃辣（记录于 2026-06-15 21:13，最后出现 2026-06-15 21:54）" in prompt_section
    assert "- 当前会话默认中文（记录于 2026-06-15 21:13，最后出现 2026-06-15 21:54）" in prompt_section
    assert "21:13:49" not in prompt_section
    assert "21:54:00" not in prompt_section


def test_long_term_memory_bundle_keeps_legacy_records_without_times_readable():
    bundle = LongTermMemoryBundle(
        user_memories=[
            LongTermMemoryRecord(
                id="mem-user-1",
                content="用户不吃辣",
                metadata={},
            )
        ],
        conversation_memories=[],
    )

    prompt_section = bundle.as_prompt_section()

    assert "- 用户不吃辣" in prompt_section
    assert "记录于" not in prompt_section
    assert "最后出现" not in prompt_section
