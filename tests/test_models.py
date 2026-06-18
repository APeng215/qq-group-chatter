from qq_group_chatter.models import (
    ConversationContext,
    LongTermMemoryBundle,
    LongTermMemoryRecord,
    build_group_conversation_context,
    build_private_conversation_context,
    conversation_memory_id,
    user_memory_id,
)


def prompt_context():
    return build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
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
        reply_to_message_id=None,
        nickname="阿咳",
        timestamp=123.0,
    )
    assert user_memory_id(context) == "qq_user:qq_group:888888:123456"
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
    assert user_memory_id(context) == "qq_user:qq_private:123456:123456"
    assert conversation_memory_id(context) == "qq_conversation:qq_private:123456"


def test_group_conversation_context_keeps_reply_message_id():
    context = build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m2",
        nickname="阿咳",
        timestamp=123.0,
        reply_to_message_id="m1",
    )

    assert context.reply_to_message_id == "m1"


def test_user_memory_id_is_isolated_by_conversation():
    group_context = build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )
    private_context = build_private_conversation_context(
        user_id=123456,
        message_id="m2",
        nickname="阿咳",
        timestamp=456.0,
    )

    assert user_memory_id(group_context) != user_memory_id(private_context)


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

    prompt_section = bundle.as_prompt_section(prompt_context())

    assert "相关会话长期记忆：\n- 默认中文" in prompt_section
    assert "相关个人长期记忆" not in prompt_section
    assert "当前 conversation 内相关长期记忆" not in prompt_section
    assert "不应出现在 prompt" not in prompt_section


def test_long_term_memory_bundle_renders_compact_empty_section():
    bundle = LongTermMemoryBundle(
        user_memories=[],
        conversation_memories=[],
        global_memories=[],
    )

    prompt_section = bundle.as_prompt_section(prompt_context())

    assert prompt_section == "长期记忆：无"


def test_long_term_memory_bundle_renders_global_memories_with_source_metadata():
    bundle = LongTermMemoryBundle(
        user_memories=[],
        conversation_memories=[],
        global_memories=[
            LongTermMemoryRecord(
                id="mem-global-1",
                content="小明在上大学",
                metadata={
                    "scope": "user",
                    "kind": "other",
                    "source_user_id": "654321",
                    "source_nickname": "小明",
                    "conversation_id": "qq_group:888888",
                    "conversation_type": "group",
                },
            )
        ],
    )

    prompt_section = bundle.as_prompt_section(prompt_context())

    assert "当前 conversation 内相关长期记忆" in prompt_section
    assert "- 小明在上大学" in prompt_section
    assert (
        "来源：scope=user；source_user_id=654321；source_nickname=小明；"
        "conversation_id=qq_group:888888；conversation_type=group；kind=other"
    ) in prompt_section
    assert "不要把其他人的信息误当成当前发言者自己的信息" in prompt_section


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

    prompt_section = bundle.as_prompt_section(prompt_context())

    assert "- 用户不吃辣" in prompt_section
    assert "- 当前会话默认中文" in prompt_section
    assert "mem-user-1" not in prompt_section
    assert "mem-conv-1" not in prompt_section
    assert "mem0" not in prompt_section
    assert "scope=conversation" not in prompt_section


def test_long_term_memory_bundle_omits_memory_times_from_prompt_context():
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

    prompt_section = bundle.as_prompt_section(prompt_context())

    assert "- 用户不吃辣" in prompt_section
    assert "- 当前会话默认中文" in prompt_section
    assert "记录于" not in prompt_section
    assert "最后出现" not in prompt_section
    assert "2026-06-15" not in prompt_section
    assert "21:13" not in prompt_section
    assert "21:54" not in prompt_section


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

    prompt_section = bundle.as_prompt_section(prompt_context())

    assert "- 用户不吃辣" in prompt_section
    assert "记录于" not in prompt_section
    assert "最后出现" not in prompt_section
