from qq_group_chatter.agent.chat_agent import ChatAgent
from qq_group_chatter.models import LongTermMemoryBundle, build_group_conversation_context


def test_chat_agent_prompt_includes_bot_identity():
    agent = ChatAgent()
    context = build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )

    prompt = agent._build_prompt(
        user_message="你叫什么？",
        context=context,
        short_term_messages=[],
        long_term_memory=LongTermMemoryBundle(user_memories=[], conversation_memories=[]),
    )

    assert "牧野神奈" in prompt
    assert "神奈" in prompt
    assert "人类" in prompt
    assert "机器人" not in prompt
