from qq_group_chatter.models import (
    LongTermMemoryOperation,
    LongTermMemoryRecord,
    build_group_conversation_context,
)
from qq_group_chatter.services.long_term_memory_planner import LongTermMemoryPlanner


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
    )

    assert operations == []


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
    )

    assert [operation.content for operation in operations] == ["第一条", "第二条"]
