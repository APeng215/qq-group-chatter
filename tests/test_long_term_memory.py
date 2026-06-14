import asyncio

from qq_group_chatter.models import (
    LongTermMemoryCandidate,
    LongTermMemoryIngestionJob,
    build_group_conversation_context,
)
from qq_group_chatter.services.long_term_memory import LongTermMemoryService


class FakeMem0Client:
    def __init__(self):
        self.search_calls = []
        self.add_calls = []
        self.search_results = {}

    def search(self, query, *, filters=None, limit=None):
        self.search_calls.append({"query": query, "filters": filters, "limit": limit})
        return self.search_results.get(filters["user_id"], [])

    def add(self, messages, *, user_id, metadata=None):
        self.add_calls.append({"messages": messages, "user_id": user_id, "metadata": metadata})
        return {"id": f"memory-{len(self.add_calls)}"}


class FakeExtractor:
    def __init__(self, candidates=None, raises=None):
        self.candidates = candidates or []
        self.raises = raises
        self.calls = []

    async def extract(self, *, user_message, context):
        self.calls.append({"user_message": user_message, "context": context})
        if self.raises:
            raise self.raises
        return self.candidates


def context():
    return build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )


async def test_search_queries_user_and_conversation_memories():
    mem0 = FakeMem0Client()
    mem0.search_results = {
        "qq_user:123456": [{"memory": "用户不吃辣"}],
        "qq_conversation:qq_group:888888": [{"memory": "当前会话默认中文"}],
    }
    service = LongTermMemoryService(mem0_client=mem0, extractor=FakeExtractor())

    bundle = await service.search("晚上吃川菜吗", context())

    assert bundle.user_memories == ["用户不吃辣"]
    assert bundle.conversation_memories == ["当前会话默认中文"]
    assert [call["filters"]["user_id"] for call in mem0.search_calls] == [
        "qq_user:123456",
        "qq_conversation:qq_group:888888",
    ]


async def test_ingestion_adds_valid_candidate_asynchronously():
    mem0 = FakeMem0Client()
    extractor = FakeExtractor(
        [
            LongTermMemoryCandidate(
                scope="user",
                content="用户不吃辣",
                confidence=0.92,
                kind="preference",
            )
        ]
    )
    service = LongTermMemoryService(mem0_client=mem0, extractor=extractor)
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(context=context(), user_message="我不吃辣")
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert extractor.calls[0]["user_message"] == "我不吃辣"
    assert mem0.add_calls == [
        {
            "messages": [{"role": "user", "content": "用户不吃辣"}],
            "user_id": "qq_user:123456",
            "metadata": {
                "source": "qq",
                "conversation_id": "qq_group:888888",
                "conversation_type": "group",
                "message_id": "m1",
                "scope": "user",
                "kind": "preference",
            },
        }
    ]


async def test_ingestion_skips_low_confidence_and_duplicate_candidates():
    mem0 = FakeMem0Client()
    mem0.search_results = {"qq_conversation:qq_group:888888": [{"memory": "当前会话默认使用中文交流"}]}
    extractor = FakeExtractor(
        [
            LongTermMemoryCandidate(
                scope="user",
                content="用户今天好困",
                confidence=0.3,
                kind="other",
            ),
            LongTermMemoryCandidate(
                scope="conversation",
                content="当前会话默认使用中文交流",
                confidence=0.91,
                kind="conversation_rule",
            ),
        ]
    )
    service = LongTermMemoryService(mem0_client=mem0, extractor=extractor)
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(context=context(), user_message="这个群默认说中文")
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert mem0.add_calls == []


async def test_worker_errors_do_not_escape():
    service = LongTermMemoryService(
        mem0_client=FakeMem0Client(),
        extractor=FakeExtractor(raises=RuntimeError("extract failed")),
    )
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(context=context(), user_message="我不吃辣")
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()
