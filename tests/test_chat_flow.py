from qq_group_chatter.agent.chat_agent import ChatAgent
from qq_group_chatter.models import LongTermMemoryBundle, build_group_conversation_context
from qq_group_chatter.orchestrator import ChatOrchestrator
from qq_group_chatter.services.short_term_memory import ShortTermMemoryService


class FakeLongTermMemory:
    def __init__(self):
        self.enqueued = []
        self.search_calls = []

    async def enqueue_ingestion(self, job):
        self.enqueued.append(job)

    async def search(self, user_message, context):
        self.search_calls.append({"user_message": user_message, "context": context})
        return LongTermMemoryBundle(
            user_memories=["用户不吃辣"],
            conversation_memories=["当前会话默认中文"],
        )


class FakeResponder:
    def __init__(self):
        self.calls = []

    async def generate_reply(self, *, user_message, context, short_term_messages, long_term_memory):
        self.calls.append(
            {
                "user_message": user_message,
                "short_term_messages": short_term_messages,
                "long_term_memory": long_term_memory,
            }
        )
        return "好的"


async def test_orchestrator_writes_short_term_then_enqueues_and_replies():
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

    reply = await orchestrator.handle_message(context=context, user_message="我不吃辣")

    assert reply == "好的"
    assert long_term.enqueued[0].user_message == "我不吃辣"
    assert long_term.search_calls[0]["user_message"] == "我不吃辣"
    assert [item.content for item in responder.calls[0]["short_term_messages"]] == ["我不吃辣"]
    assert responder.calls[0]["long_term_memory"].user_memories == ["用户不吃辣"]
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
