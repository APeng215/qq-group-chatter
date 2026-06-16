from qq_group_chatter.models import build_group_conversation_context
from qq_group_chatter.services.long_term_memory_planner import LongTermMemoryPlanner


class TraceContextPlannerLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def ainvoke(
        self,
        prompt,
        *,
        response_format=None,
        system_prompt=None,
        trace_context=None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "response_format": response_format,
                "system_prompt": system_prompt,
                "trace_context": trace_context,
            }
        )
        return self.response


class PromptOnlyPlannerLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return self.response


async def test_planner_passes_json_response_format_and_trace_context_to_llm():
    llm = TraceContextPlannerLLM({"operations": []})
    planner = LongTermMemoryPlanner(llm=llm)

    operations = await planner.plan(
        user_message="hello",
        context=build_group_conversation_context(
            group_id=888888,
            user_id=123456,
            message_id="m1",
            nickname="tester",
            timestamp=123.0,
        ),
        user_memories=[],
        conversation_memories=[],
    )

    assert operations == []
    assert llm.calls[0]["response_format"] == {"type": "json_object"}
    assert llm.calls[0]["system_prompt"].startswith("你是长期记忆规划器。")
    assert "用户消息：" not in llm.calls[0]["system_prompt"]
    assert "用户消息：\nhello" in llm.calls[0]["prompt"]
    assert llm.calls[0]["trace_context"] == {
        "component": "memory_planner",
        "operation": "plan_memory",
    }


async def test_planner_falls_back_for_prompt_only_llm():
    llm = PromptOnlyPlannerLLM(
        {
            "operations": [
                {
                    "action": "add",
                    "scope": "user",
                    "target_id": None,
                    "content": "user prefers concise answers",
                    "kind": "preference",
                    "confidence": 0.9,
                }
            ]
        }
    )
    planner = LongTermMemoryPlanner(llm=llm)

    operations = await planner.plan(
        user_message="hello",
        context=build_group_conversation_context(
            group_id=888888,
            user_id=123456,
            message_id="m1",
            nickname="tester",
            timestamp=123.0,
        ),
        user_memories=[],
        conversation_memories=[],
    )

    assert [operation.content for operation in operations] == ["user prefers concise answers"]
    assert len(llm.prompts) == 1
