from qq_group_chatter.agent.chat_agent import ChatAgent, ChatReplyDecision, WebSearchDecision
from qq_group_chatter.models import (
    ChatMessage,
    LongTermMemoryBundle,
    LongTermMemoryRecord,
    build_group_conversation_context,
)
from qq_group_chatter.orchestrator import ChatOrchestrator
from qq_group_chatter.services.short_term_memory import ShortTermMemoryService
from qq_group_chatter.services.web_search import SearchSource


class RecordingShortTermMemory:
    def __init__(self, events):
        self._events = events
        self._inner = ShortTermMemoryService(max_messages_per_conversation=10)

    async def add_message(self, message: ChatMessage) -> None:
        self._events.append("short_term.add")
        await self._inner.add_message(message)

    async def get_recent(self, conversation_id: str, limit: int = 20) -> list[ChatMessage]:
        self._events.append("short_term.get_recent")
        return await self._inner.get_recent(conversation_id, limit=limit)


class FakeLongTermMemory:
    def __init__(self, events=None):
        self.events = events
        self.enqueued = []
        self.search_calls = []

    async def enqueue_ingestion(self, job):
        if self.events is not None:
            self.events.append("long_term.enqueue")
        self.enqueued.append(job)

    async def search(self, user_message, context):
        if self.events is not None:
            self.events.append("long_term.search")
        self.search_calls.append({"user_message": user_message, "context": context})
        return LongTermMemoryBundle(
            user_memories=[
                LongTermMemoryRecord(id="mem-user-1", content="用户不吃辣", metadata={})
            ],
            conversation_memories=[
                LongTermMemoryRecord(id="mem-conv-1", content="当前会话默认中文", metadata={})
            ],
        )


class FailingLongTermMemory(FakeLongTermMemory):
    async def search(self, user_message, context):
        self.search_calls.append({"user_message": user_message, "context": context})
        raise RuntimeError("mem0 unavailable")


class FakeResponder:
    def __init__(self, events=None):
        self.events = events
        self.calls = []

    async def generate_reply(self, *, user_message, context, short_term_messages, long_term_memory):
        if self.events is not None:
            self.events.append("agent.generate_reply")
        self.calls.append(
            {
                "user_message": user_message,
                "short_term_messages": short_term_messages,
                "long_term_memory": long_term_memory,
            }
        )
        return ChatReplyDecision(content="好的")


class FixedResponder:
    def __init__(self, decision):
        self.decision = decision
        self.calls = []
        self.grounded_calls = []

    async def generate_reply(self, *, user_message, context, short_term_messages, long_term_memory):
        self.calls.append(
            {
                "user_message": user_message,
                "short_term_messages": short_term_messages,
                "long_term_memory": long_term_memory,
            }
        )
        return self.decision

    async def generate_grounded_search_reply(
        self,
        *,
        user_message,
        search_query,
        search_sources,
        context,
        short_term_messages,
        long_term_memory,
    ):
        self.grounded_calls.append(
            {
                "user_message": user_message,
                "search_query": search_query,
                "search_sources": search_sources,
                "context": context,
                "short_term_messages": short_term_messages,
                "long_term_memory": long_term_memory,
            }
        )
        return "神奈基于搜索资料的回复"


class FakeWebSearch:
    def __init__(self, sources=None):
        self.sources = sources
        self.search_sources_queries = []

    async def search_sources(self, query):
        self.search_sources_queries.append(query)
        if self.sources is not None:
            return self.sources
        return [
            SearchSource(
                title="来源标题",
                url="https://example.com/news",
                content="摘要",
                raw_content="原网页正文",
            )
        ]


async def test_orchestrator_returns_pending_reply_without_recording_assistant_message():
    events = []
    short_term = RecordingShortTermMemory(events)
    long_term = FakeLongTermMemory(events)
    responder = FakeResponder(events)
    orchestrator = ChatOrchestrator(
        short_term_memory=short_term,
        long_term_memory=long_term,
        chat_agent=responder,
    )
    context = build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )

    pending_reply = await orchestrator.handle_message(context=context, user_message="我不吃辣")

    assert pending_reply is not None
    assert pending_reply.content == "好的"
    assert long_term.enqueued[0].user_message == "我不吃辣"
    assert long_term.search_calls[0]["user_message"] == "我不吃辣"
    assert [item.content for item in responder.calls[0]["short_term_messages"]] == ["我不吃辣"]
    assert [item.content for item in long_term.enqueued[0].short_term_messages] == ["我不吃辣"]
    assert responder.calls[0]["long_term_memory"].user_memories[0].content == "用户不吃辣"
    assert long_term.enqueued[0].existing_memories is responder.calls[0]["long_term_memory"]
    assert events == [
        "short_term.add",
        "short_term.get_recent",
        "long_term.search",
        "long_term.enqueue",
        "agent.generate_reply",
    ]
    assert [item.content for item in await short_term.get_recent("qq_group:888888")] == ["我不吃辣"]


async def test_orchestrator_records_assistant_message_after_send_success():
    short_term = ShortTermMemoryService(max_messages_per_conversation=10)
    long_term = FakeLongTermMemory()
    responder = FakeResponder()
    orchestrator = ChatOrchestrator(
        short_term_memory=short_term,
        long_term_memory=long_term,
        chat_agent=responder,
    )
    context = build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )

    pending_reply = await orchestrator.handle_message(context=context, user_message="我不吃辣")
    await orchestrator.record_assistant_reply(pending_reply)

    assert [item.content for item in await short_term.get_recent("qq_group:888888")] == [
        "我不吃辣",
        "好的",
    ]


async def test_orchestrator_ignores_empty_messages():
    orchestrator = ChatOrchestrator(
        short_term_memory=ShortTermMemoryService(),
        long_term_memory=FakeLongTermMemory(),
        chat_agent=ChatAgent(),
    )
    context = build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )

    assert await orchestrator.handle_message(context=context, user_message="   ") is None


async def test_orchestrator_continues_reply_but_skips_ingestion_when_memory_search_fails():
    long_term = FailingLongTermMemory()
    responder = FakeResponder()
    orchestrator = ChatOrchestrator(
        short_term_memory=ShortTermMemoryService(),
        long_term_memory=long_term,
        chat_agent=responder,
    )
    context = build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )

    pending_reply = await orchestrator.handle_message(context=context, user_message="你好")

    assert pending_reply.content == "好的"
    assert long_term.enqueued == []
    assert responder.calls[0]["long_term_memory"] == LongTermMemoryBundle(
        user_memories=[],
        conversation_memories=[],
    )


async def test_orchestrator_runs_web_search_fallback_after_llm_notice():
    short_term = ShortTermMemoryService(max_messages_per_conversation=10)
    long_term = FakeLongTermMemory()
    responder = FixedResponder(
        WebSearchDecision(
            notice="我查一下再回你。",
            query="DeepSeek 最新消息",
        )
    )
    web_search = FakeWebSearch()
    notices = []
    orchestrator = ChatOrchestrator(
        short_term_memory=short_term,
        long_term_memory=long_term,
        chat_agent=responder,
        web_search=web_search,
    )
    context = build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )

    pending_reply = await orchestrator.handle_message(
        context=context,
        user_message="DeepSeek 今天有什么新闻？",
        on_search_start=lambda notice: notices.append(notice),
    )

    assert pending_reply.content == "神奈基于搜索资料的回复"
    assert notices == ["我查一下再回你。"]
    assert web_search.search_sources_queries == ["DeepSeek 最新消息"]
    assert responder.grounded_calls[0]["user_message"] == "DeepSeek 今天有什么新闻？"
    assert responder.grounded_calls[0]["search_query"] == "DeepSeek 最新消息"
    assert responder.grounded_calls[0]["search_sources"][0].raw_content == "原网页正文"
    assert long_term.enqueued[0].user_message == "DeepSeek 今天有什么新闻？"
    assert [item.content for item in await short_term.get_recent("qq_group:888888")] == [
        "DeepSeek 今天有什么新闻？"
    ]


async def test_orchestrator_does_not_search_for_regular_reply():
    web_search = FakeWebSearch()
    orchestrator = ChatOrchestrator(
        short_term_memory=ShortTermMemoryService(),
        long_term_memory=FakeLongTermMemory(),
        chat_agent=FixedResponder(ChatReplyDecision(content="普通回复")),
        web_search=web_search,
    )
    context = build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )
    notices = []

    pending_reply = await orchestrator.handle_message(
        context=context,
        user_message="你好",
        on_search_start=lambda notice: notices.append(notice),
    )

    assert pending_reply.content == "普通回复"
    assert web_search.search_sources_queries == []
    assert notices == []


async def test_orchestrator_returns_search_unavailable_when_fallback_has_no_service():
    orchestrator = ChatOrchestrator(
        short_term_memory=ShortTermMemoryService(),
        long_term_memory=FakeLongTermMemory(),
        chat_agent=FixedResponder(
            WebSearchDecision(
                notice="我查一下再回你。",
                query="DeepSeek 最新消息",
            )
        ),
    )
    context = build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )
    notices = []

    pending_reply = await orchestrator.handle_message(
        context=context,
        user_message="DeepSeek 今天有什么新闻？",
        on_search_start=lambda notice: notices.append(notice),
    )

    assert pending_reply.content == "我现在没法联网搜索，稍后再试。"
    assert notices == []


async def test_orchestrator_returns_no_search_source_message_without_grounded_reply():
    responder = FixedResponder(
        WebSearchDecision(
            notice="我查一下再回你。",
            query="DeepSeek 最新消息",
        )
    )
    web_search = FakeWebSearch(sources=[])
    orchestrator = ChatOrchestrator(
        short_term_memory=ShortTermMemoryService(),
        long_term_memory=FakeLongTermMemory(),
        chat_agent=responder,
        web_search=web_search,
    )
    context = build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )

    pending_reply = await orchestrator.handle_message(
        context=context,
        user_message="DeepSeek 今天有什么新闻？",
        on_search_start=lambda notice: None,
    )

    assert pending_reply.content == "我搜了一下，但没找到足够可靠的网页正文来确认。"
    assert responder.grounded_calls == []
