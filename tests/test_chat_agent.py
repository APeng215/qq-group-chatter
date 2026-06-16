from qq_group_chatter.agent.chat_agent import (
    ChatAgent,
    ChatReplyDecision,
    WebSearchDecision,
    parse_chat_decision,
)
from qq_group_chatter.models import (
    ChatMessage,
    LongTermMemoryBundle,
    LongTermMemoryRecord,
    build_group_conversation_context,
)
from qq_group_chatter.services.web_search import SearchSource


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


def test_chat_agent_prompt_includes_current_time_and_timed_short_term_history(monkeypatch):
    monkeypatch.setattr(
        "qq_group_chatter.agent.chat_agent.current_time_text",
        lambda: "2026-06-16 23:52",
    )
    agent = ChatAgent()
    context = build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )

    prompt = agent._build_prompt(
        user_message="刚才说到哪了？",
        context=context,
        short_term_messages=[
            ChatMessage(
                conversation_id=context.conversation_id,
                role="user",
                content="前文问题",
                user_id="123456",
                nickname="阿咳",
                message_id="m0",
                timestamp=1781531640.0,
            )
        ],
        long_term_memory=LongTermMemoryBundle(user_memories=[], conversation_memories=[]),
    )

    assert "当前时间：2026-06-16 23:52" in prompt
    assert "[2026-06-15 21:54] 阿咳: 前文问题" in prompt
    assert "21:54:00" not in prompt


def test_parse_chat_decision_accepts_reply_json():
    decision = parse_chat_decision('{"action":"reply","content":"普通回复"}')

    assert decision == ChatReplyDecision(content="普通回复")


def test_parse_chat_decision_accepts_web_search_json():
    decision = parse_chat_decision(
        '{"action":"web_search","notice":"我查一下最新情况，稍等。","query":"DeepSeek 最新消息"}'
    )

    assert decision == WebSearchDecision(
        notice="我查一下最新情况，稍等。",
        query="DeepSeek 最新消息",
    )


def test_parse_chat_decision_rejects_invalid_json_or_schema():
    invalid_replies = [
        "普通回复",
        '{"action":"reply","content":""}',
        '{"action":"reply","content":"普通回复","extra":1}',
        '{"action":"web_search","notice":"","query":"DeepSeek"}',
        '{"action":"web_search","notice":"我查一下","query":""}',
        '{"action":"web_search","notice":"<神奈要先发给对方的等待提示>","query":"DeepSeek"}',
        '{"action":"web_search","notice":"神奈要先发给对方的等待提示","query":"DeepSeek"}',
        '{"action":"web_search","notice":"我查一下","query":"适合搜索的查询词"}',
        '{"action":"unknown","content":"普通回复"}',
    ]

    for reply in invalid_replies:
        assert parse_chat_decision(reply) is None


class RecordingLLM:
    def __init__(self):
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return "神奈基于搜索资料的回复"


async def test_chat_agent_builds_grounded_search_prompt_with_chat_context():
    llm = RecordingLLM()
    agent = ChatAgent(llm=llm)
    context = build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )

    reply = await agent.generate_grounded_search_reply(
        user_message="DeepSeek 今天有什么新闻？",
        search_query="DeepSeek 最新消息",
        search_sources=[
            SearchSource(
                title="来源标题",
                url="https://example.com/news",
                content="摘要",
                raw_content="原网页正文",
            )
        ],
        context=context,
        short_term_messages=[
            ChatMessage(
                conversation_id=context.conversation_id,
                role="user",
                content="前文问题",
                user_id="123456",
                nickname="阿咳",
                message_id="m0",
                timestamp=122.0,
            )
        ],
        long_term_memory=LongTermMemoryBundle(
            user_memories=[
                LongTermMemoryRecord(id="mem-user-1", content="用户不吃辣", metadata={})
            ],
            conversation_memories=[],
        ),
    )

    assert reply == "神奈基于搜索资料的回复"
    prompt = llm.prompts[0]
    assert "不要自称 AI、助手、模型或自动程序" in prompt
    assert "DeepSeek 今天有什么新闻？" in prompt
    assert "DeepSeek 最新消息" in prompt
    assert "来源标题" in prompt
    assert "原网页正文" in prompt
    assert "前文问题" in prompt
    assert "用户不吃辣" in prompt
    assert "https://example.com/news" not in prompt


async def test_grounded_search_prompt_includes_current_time_and_timed_short_term_history(monkeypatch):
    monkeypatch.setattr(
        "qq_group_chatter.agent.chat_agent.current_time_text",
        lambda: "2026-06-16 23:52",
    )
    llm = RecordingLLM()
    agent = ChatAgent(llm=llm)
    context = build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )

    await agent.generate_grounded_search_reply(
        user_message="DeepSeek 今天有什么新闻？",
        search_query="DeepSeek 最新消息",
        search_sources=[],
        context=context,
        short_term_messages=[
            ChatMessage(
                conversation_id=context.conversation_id,
                role="assistant",
                content="前文回答",
                user_id=None,
                nickname=None,
                message_id="m0",
                timestamp=1781531640.0,
            )
        ],
        long_term_memory=LongTermMemoryBundle(user_memories=[], conversation_memories=[]),
    )

    prompt = llm.prompts[0]
    assert "当前时间：2026-06-16 23:52" in prompt
    assert "[2026-06-15 21:54] assistant: 前文回答" in prompt
    assert "21:54:00" not in prompt
