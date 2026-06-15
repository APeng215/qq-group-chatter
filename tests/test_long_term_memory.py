import asyncio

from qq_group_chatter.models import (
    LongTermMemoryCandidate,
    LongTermMemoryIngestionJob,
    build_group_conversation_context,
)
from qq_group_chatter.services.long_term_memory import LongTermMemoryService
from qq_group_chatter.services.long_term_memory_extractor import LongTermMemoryExtractor


class FakeMem0Client:
    def __init__(self):
        self.search_calls = []
        self.add_calls = []
        self.search_results = {}
        self.search_raises = None
        self.close_calls = 0

    def search(self, query, *, filters=None, top_k=None):
        self.search_calls.append({"query": query, "filters": filters, "top_k": top_k})
        if self.search_raises:
            raise self.search_raises
        return self.search_results.get(filters["user_id"], [])

    def add(self, messages, *, user_id, metadata=None, infer=True):
        self.add_calls.append(
            {"messages": messages, "user_id": user_id, "metadata": metadata, "infer": infer}
        )
        return {"id": f"memory-{len(self.add_calls)}"}

    def close(self):
        self.close_calls += 1


class FakeQdrantClient:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class FakeVectorStore:
    def __init__(self):
        self.client = FakeQdrantClient()


class FakeMem0ClientWithVectorStore(FakeMem0Client):
    def __init__(self):
        super().__init__()
        self.vector_store = FakeVectorStore()


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


class FakeExtractorLLM:
    def __init__(self, response):
        self.response = response

    async def ainvoke(self, prompt):
        return self.response


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
    assert [call["top_k"] for call in mem0.search_calls] == [5, 5]


async def test_search_returns_empty_bundle_when_mem0_search_fails(monkeypatch):
    mem0 = FakeMem0Client()
    error = RuntimeError("mem0 unavailable")
    mem0.search_raises = error
    recorded_errors = []
    monkeypatch.setattr(
        "qq_group_chatter.services.long_term_memory.record_error",
        lambda stage, exc: recorded_errors.append({"stage": stage, "exc": exc}),
    )
    service = LongTermMemoryService(mem0_client=mem0, extractor=FakeExtractor())

    bundle = await service.search("晚上吃川菜吗", context())

    assert bundle.user_memories == []
    assert bundle.conversation_memories == []
    assert recorded_errors == [
        {"stage": "mem0_search", "exc": error},
        {"stage": "mem0_search", "exc": error},
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

    await service.enqueue_ingestion(LongTermMemoryIngestionJob(context=context(), user_message="我不吃辣"))
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
            "infer": False,
        }
    ]
    assert mem0.search_calls[0]["top_k"] == 3


async def test_ingestion_skips_low_confidence_and_duplicate_candidates():
    mem0 = FakeMem0Client()
    mem0.search_results = {
        "qq_conversation:qq_group:888888": [{"memory": "当前会话默认使用中文交流"}]
    }
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

    await service.enqueue_ingestion(LongTermMemoryIngestionJob(context=context(), user_message="我不吃辣"))
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()


async def test_stop_closes_mem0_client_when_supported():
    mem0 = FakeMem0Client()
    service = LongTermMemoryService(mem0_client=mem0, extractor=FakeExtractor())
    await service.start()

    await service.stop()

    assert mem0.close_calls == 1


async def test_stop_closes_mem0_vector_store_client_when_supported():
    mem0 = FakeMem0ClientWithVectorStore()
    service = LongTermMemoryService(mem0_client=mem0, extractor=FakeExtractor())
    await service.start()

    await service.stop()

    assert mem0.vector_store.client.close_calls == 1


def test_extractor_prompt_is_not_mojibake():
    extractor = LongTermMemoryExtractor(llm=object())
    prompt = extractor._build_prompt(user_message="我不吃辣", context=context())

    assert "长期记忆提取器" in prompt
    assert "最多输出 2 条" in prompt
    assert "我不吃辣" in prompt
    assert "浣犳槸" not in prompt


async def test_extractor_parses_json_inside_markdown_fence():
    extractor = LongTermMemoryExtractor(
        llm=FakeExtractorLLM(
            '```json\n'
            '{"memories":[{"scope":"user","content":"用户不吃辣","confidence":0.92,"kind":"preference"}]}'
            "\n```"
        )
    )

    candidates = await extractor.extract(user_message="我不吃辣", context=context())

    assert candidates == [
        LongTermMemoryCandidate(
            scope="user",
            content="用户不吃辣",
            confidence=0.92,
            kind="preference",
        )
    ]


async def test_extractor_skips_malformed_candidates_without_dropping_valid_ones():
    extractor = LongTermMemoryExtractor(
        llm=FakeExtractorLLM(
            {
                "memories": [
                    "not an object",
                    {
                        "scope": "user",
                        "content": "用户不吃辣",
                        "confidence": "high",
                        "kind": "preference",
                    },
                    {
                        "scope": "conversation",
                        "content": "当前会话默认中文",
                        "confidence": 0.91,
                        "kind": "conversation_rule",
                    },
                ]
            }
        )
    )

    candidates = await extractor.extract(user_message="这个群默认说中文", context=context())

    assert candidates == [
        LongTermMemoryCandidate(
            scope="conversation",
            content="当前会话默认中文",
            confidence=0.91,
            kind="conversation_rule",
        )
    ]
