from qq_group_chatter.models import (
    ChatMessage,
    LongTermMemoryOperation,
    LongTermMemoryRecord,
    build_group_conversation_context,
)
from qq_group_chatter.services.long_term_memory_planner import (
    PLANNER_SYSTEM_PROMPT,
    LongTermMemoryPlanner,
)


class FakePlannerLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return self.response


def context():
    return build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )


def user_records():
    return [
        LongTermMemoryRecord(
            id="mem-user-1",
            content="用户喜欢吃辣",
            metadata={"kind": "preference"},
        )
    ]


def conversation_records():
    return [
        LongTermMemoryRecord(
            id="mem-conv-1",
            content="当前会话默认中文",
            metadata={"kind": "conversation_rule"},
        )
    ]


def global_records():
    return [
        LongTermMemoryRecord(
            id="mem-global-user-1",
            content="小明在上大学",
            metadata={"scope": "user", "kind": "other"},
        ),
        LongTermMemoryRecord(
            id="mem-global-conv-1",
            content="当前会话曾聊过旧项目",
            metadata={"scope": "conversation", "kind": "other"},
        ),
    ]


async def test_planner_parses_operations_inside_markdown_fence():
    llm = FakePlannerLLM(
        "```json\n"
        '{"operations":['
        '{"action":"update","scope":"user","target_id":"mem-user-1","content":"用户不吃辣","kind":"preference","confidence":0.92},'
        '{"action":"add","scope":"conversation","target_id":null,"content":"当前会话默认聊游戏","kind":"conversation_rule","confidence":0.91}'
        "]}"
        "\n```"
    )
    planner = LongTermMemoryPlanner(llm=llm)

    operations = await planner.plan(
        user_message="我不吃辣，这个群默认聊游戏",
        context=context(),
        user_memories=user_records(),
        conversation_memories=conversation_records(),
    )

    assert operations == [
        LongTermMemoryOperation(
            action="update",
            scope="user",
            target_id="mem-user-1",
            content="用户不吃辣",
            kind="preference",
            confidence=0.92,
        ),
        LongTermMemoryOperation(
            action="add",
            scope="conversation",
            target_id=None,
            content="当前会话默认聊游戏",
            kind="conversation_rule",
            confidence=0.91,
        ),
    ]
    assert "mem-user-1" in llm.prompts[0]
    assert "用户喜欢吃辣" in llm.prompts[0]
    assert "mem-conv-1" in llm.prompts[0]
    assert "当前会话默认中文" in llm.prompts[0]
    assert "qq_group:888888" not in llm.prompts[0]
    assert "QQ号：123456" in llm.prompts[0]
    assert "昵称：阿咳" in llm.prompts[0]
    assert "QQ号是识别同一用户的稳定身份键" in llm.prompts[0]


async def test_planner_returns_empty_operations_for_empty_or_non_json_response():
    planner = LongTermMemoryPlanner(llm=FakePlannerLLM("没有需要记录的长期记忆。"))

    operations = await planner.plan(
        user_message="打什么",
        context=context(),
        user_memories=[],
        conversation_memories=[],
        global_memories=[],
    )

    assert operations == []


async def test_planner_prompt_includes_short_term_context_as_auxiliary_history():
    llm = FakePlannerLLM({"operations": []})
    planner = LongTermMemoryPlanner(llm=llm)

    await planner.plan(
        user_message="对，我还是想吃那个",
        context=context(),
        user_memories=[],
        conversation_memories=[],
        short_term_messages=[
            ChatMessage(
                conversation_id="qq_group:888888",
                role="assistant",
                content="上次你说晚饭想吃咖喱",
                user_id=None,
                nickname=None,
                message_id=None,
                timestamp=122.0,
            ),
            ChatMessage(
                conversation_id="qq_group:888888",
                role="user",
                content="对，我还是想吃那个",
                user_id="123456",
                nickname="阿咳",
                message_id="m1",
                timestamp=123.0,
            ),
        ],
    )

    assert "短期对话上下文" in llm.prompts[0]
    assert "[神奈] 上次你说晚饭想吃咖喱" in llm.prompts[0]
    assert "[QQ:123456 昵称:阿咳] 对，我还是想吃那个" in llm.prompts[0]
    assert "阿咳：对，我还是想吃那个" not in llm.prompts[0]
    assert "只辅助理解本轮用户消息" in llm.prompts[0]


async def test_planner_prompt_includes_assistant_reply_only_as_confirmation_context():
    llm = FakePlannerLLM({"operations": []})
    planner = LongTermMemoryPlanner(llm=llm)

    await planner.plan(
        user_message="以后说话可爱一点",
        assistant_reply="好呀，那我以后会更可爱一点跟你说话。",
        context=context(),
        user_memories=[],
        conversation_memories=[],
    )

    prompt = llm.prompts[0]
    assert "本轮神奈回复（仅用于判断用户请求是否被接受、拒绝、部分接受或限定；不能作为独立事实来源）" in prompt
    assert "好呀，那我以后会更可爱一点跟你说话。" in prompt
    assert "不要仅因为神奈这样回复就创建事实记忆" in prompt


async def test_planner_skips_invalid_operations_and_invalid_update_target():
    planner = LongTermMemoryPlanner(
        llm=FakePlannerLLM(
            {
                "operations": [
                    {
                        "action": "delete",
                        "scope": "user",
                        "content": "非法动作",
                        "kind": "other",
                        "confidence": 0.9,
                    },
                    {
                        "action": "add",
                        "scope": "group_user",
                        "content": "非法 scope",
                        "kind": "other",
                        "confidence": 0.9,
                    },
                    {
                        "action": "add",
                        "scope": "user",
                        "content": "低置信度",
                        "kind": "other",
                        "confidence": 0.2,
                    },
                    {
                        "action": "update",
                        "scope": "user",
                        "target_id": "missing",
                        "content": "不存在的 target",
                        "kind": "preference",
                        "confidence": 0.91,
                    },
                    {
                        "action": "skip",
                        "scope": "conversation",
                        "content": "跳过",
                        "kind": "other",
                        "confidence": 0.9,
                    },
                ]
            }
        )
    )

    operations = await planner.plan(
        user_message="随便",
        context=context(),
        user_memories=user_records(),
        conversation_memories=conversation_records(),
        global_memories=[],
    )

    assert operations == [
        LongTermMemoryOperation(
            action="skip",
            scope="conversation",
            target_id=None,
            content="跳过",
            kind="other",
            confidence=0.9,
        )
    ]


async def test_planner_accepts_delete_for_existing_memory_id_with_empty_content():
    planner = LongTermMemoryPlanner(
        llm=FakePlannerLLM(
            {
                "operations": [
                    {
                        "action": "delete",
                        "scope": "user",
                        "target_id": "mem-user-1",
                        "content": "",
                        "kind": "preference",
                        "confidence": 0.93,
                    }
                ]
            }
        )
    )

    operations = await planner.plan(
        user_message="忘掉我喜欢吃辣这件事",
        context=context(),
        user_memories=user_records(),
        conversation_memories=conversation_records(),
        global_memories=[],
    )

    assert operations == [
        LongTermMemoryOperation(
            action="delete",
            scope="user",
            target_id="mem-user-1",
            content="",
            kind="preference",
            confidence=0.93,
        )
    ]


async def test_planner_rejects_delete_for_missing_or_cross_scope_target():
    planner = LongTermMemoryPlanner(
        llm=FakePlannerLLM(
            {
                "operations": [
                    {
                        "action": "delete",
                        "scope": "user",
                        "target_id": "missing",
                        "content": "",
                        "kind": "preference",
                        "confidence": 0.93,
                    },
                    {
                        "action": "delete",
                        "scope": "conversation",
                        "target_id": "mem-user-1",
                        "content": "",
                        "kind": "preference",
                        "confidence": 0.93,
                    },
                ]
            }
        )
    )

    operations = await planner.plan(
        user_message="忘掉这些记忆",
        context=context(),
        user_memories=user_records(),
        conversation_memories=conversation_records(),
        global_memories=[],
    )

    assert operations == []


async def test_planner_counts_delete_as_writable_operation():
    planner = LongTermMemoryPlanner(
        llm=FakePlannerLLM(
            {
                "operations": [
                    {
                        "action": "delete",
                        "scope": "user",
                        "target_id": "mem-user-1",
                        "content": "",
                        "kind": "preference",
                        "confidence": 0.9,
                    },
                    {
                        "action": "add",
                        "scope": "conversation",
                        "content": "第二条",
                        "kind": "other",
                        "confidence": 0.9,
                    },
                    {
                        "action": "add",
                        "scope": "user",
                        "content": "第三条",
                        "kind": "other",
                        "confidence": 0.9,
                    },
                ]
            }
        )
    )

    operations = await planner.plan(
        user_message="忘掉一个，再记两个",
        context=context(),
        user_memories=user_records(),
        conversation_memories=conversation_records(),
        global_memories=[],
    )

    assert [operation.action for operation in operations] == ["delete", "add"]


async def test_planner_limits_writable_operations_to_two():
    planner = LongTermMemoryPlanner(
        llm=FakePlannerLLM(
            {
                "operations": [
                    {
                        "action": "add",
                        "scope": "user",
                        "content": "第一条",
                        "kind": "other",
                        "confidence": 0.9,
                    },
                    {
                        "action": "add",
                        "scope": "user",
                        "content": "第二条",
                        "kind": "other",
                        "confidence": 0.9,
                    },
                    {
                        "action": "add",
                        "scope": "user",
                        "content": "第三条",
                        "kind": "other",
                        "confidence": 0.9,
                    },
                ]
            }
        )
    )

    operations = await planner.plan(
        user_message="记住三件事",
        context=context(),
        user_memories=[],
        conversation_memories=[],
        global_memories=[],
    )

    assert [operation.content for operation in operations] == ["第一条", "第二条"]


async def test_planner_prompt_includes_global_memories():
    llm = FakePlannerLLM({"operations": []})
    planner = LongTermMemoryPlanner(llm=llm)

    await planner.plan(
        user_message="小明毕业了吗",
        context=context(),
        user_memories=user_records(),
        conversation_memories=conversation_records(),
        global_memories=global_records(),
    )

    prompt = llm.prompts[0]
    assert "当前 conversation 内其他相关记忆" in prompt
    assert "mem-global-user-1" in prompt
    assert "小明在上大学" in prompt
    assert '"scope": "user"' in prompt
    assert "mem-global-conv-1" in prompt
    assert "当前会话曾聊过旧项目" in prompt
    assert '"scope": "conversation"' in prompt


def test_planner_system_prompt_requires_natural_memory_content():
    assert "可直接给聊天 Agent 使用的自然事实或规则" in PLANNER_SYSTEM_PROMPT
    assert "不要写成“用户说过/用户认为/用户希望/助手应该/助手在回复中”" in PLANNER_SYSTEM_PROMPT
    assert "本会话回复时，每句话末尾加上" in PLANNER_SYSTEM_PROMPT
    assert "不喜欢把「猪」当作贬义或骂人的表达" in PLANNER_SYSTEM_PROMPT
    assert '"content":"偏好用中文交流"' in PLANNER_SYSTEM_PROMPT
    assert '"content":"用户偏好用中文交流"' not in PLANNER_SYSTEM_PROMPT


async def test_planner_accepts_update_and_delete_for_global_memory_ids():
    planner = LongTermMemoryPlanner(
        llm=FakePlannerLLM(
            {
                "operations": [
                    {
                        "action": "update",
                        "scope": "user",
                        "target_id": "mem-global-user-1",
                        "content": "小明已经大学毕业",
                        "kind": "other",
                        "confidence": 0.92,
                    },
                    {
                        "action": "delete",
                        "scope": "conversation",
                        "target_id": "mem-global-conv-1",
                        "content": "",
                        "kind": "other",
                        "confidence": 0.92,
                    },
                ]
            }
        )
    )

    operations = await planner.plan(
        user_message="更新这些记忆",
        context=context(),
        user_memories=user_records(),
        conversation_memories=conversation_records(),
        global_memories=global_records(),
    )

    assert operations == [
        LongTermMemoryOperation(
            action="update",
            scope="user",
            target_id="mem-global-user-1",
            content="小明已经大学毕业",
            kind="other",
            confidence=0.92,
        ),
        LongTermMemoryOperation(
            action="delete",
            scope="conversation",
            target_id="mem-global-conv-1",
            content="",
            kind="other",
            confidence=0.92,
        ),
    ]


async def test_planner_rejects_global_target_id_with_wrong_scope():
    planner = LongTermMemoryPlanner(
        llm=FakePlannerLLM(
            {
                "operations": [
                    {
                        "action": "delete",
                        "scope": "conversation",
                        "target_id": "mem-global-user-1",
                        "content": "",
                        "kind": "other",
                        "confidence": 0.92,
                    }
                ]
            }
        )
    )

    operations = await planner.plan(
        user_message="删掉这个",
        context=context(),
        user_memories=user_records(),
        conversation_memories=conversation_records(),
        global_memories=global_records(),
    )

    assert operations == []
