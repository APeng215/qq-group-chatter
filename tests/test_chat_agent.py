from qq_group_chatter.agent.chat_agent import ChatAgent, parse_web_search_request
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
    assert "人类" not in prompt
    assert "不要自称" in prompt
    assert "AI" in prompt
    assert "助手" in prompt
    assert "模型" in prompt


def test_parse_web_search_request_accepts_strict_three_line_protocol():
    request = parse_web_search_request(
        "__NEED_WEB_SEARCH__\n"
        "提示: 我查一下再回你。\n"
        "查询: DeepSeek 最新消息"
    )

    assert request is not None
    assert request.notice == "我查一下再回你。"
    assert request.query == "DeepSeek 最新消息"


def test_parse_web_search_request_rejects_invalid_protocols():
    invalid_replies = [
        "普通回复",
        "__NEED_WEB_SEARCH__\n提示: \n查询: DeepSeek",
        "__NEED_WEB_SEARCH__\n提示: 我查一下\n查询: ",
        "我需要搜索\n__NEED_WEB_SEARCH__\n提示: 我查一下\n查询: DeepSeek",
        "__NEED_WEB_SEARCH__\n提示: 我查一下\n查询: DeepSeek\n马上回来",
    ]

    for reply in invalid_replies:
        assert parse_web_search_request(reply) is None
