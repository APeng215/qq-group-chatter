import asyncio

from qq_group_chatter.models import (
    LongTermMemoryBundle,
    LongTermMemoryIngestionJob,
    LongTermMemoryOperation,
    LongTermMemoryRecord,
    build_group_conversation_context,
)
from qq_group_chatter.services.long_term_memory import (
    LongTermMemorySearchError,
    LongTermMemoryService,
    normalize_mem0_records,
)


class FakeMem0Client:
    def __init__(self):
        self.search_calls = []
        self.add_calls = []
        self.update_calls = []
        self.delete_calls = []
        self.get_all_calls = []
        self.health_check_calls = []
        self.search_results = {}
        self.get_all_results = {}
        self.search_raises = None
        self.close_calls = 0

    def search(self, query, *, filters=None, top_k=None):
        call = {"query": query, "filters": filters, "top_k": top_k}
        if query == "__qq_group_chatter_startup_health_check__":
            self.health_check_calls.append(call)
        else:
            self.search_calls.append(call)
        if self.search_raises:
            raise self.search_raises
        return self.search_results.get(filters["user_id"], [])

    def add(self, messages, *, user_id, metadata=None, infer=True):
        self.add_calls.append(
            {"messages": messages, "user_id": user_id, "metadata": metadata, "infer": infer}
        )
        return {"id": f"memory-{len(self.add_calls)}"}

    def update(self, memory_id, data, metadata=None):
        self.update_calls.append({"memory_id": memory_id, "data": data, "metadata": metadata})
        return {"id": memory_id}

    def delete(self, memory_id):
        self.delete_calls.append({"memory_id": memory_id})
        return {"id": memory_id}

    def get_all(self, *, filters=None, top_k=20):
        self.get_all_calls.append({"filters": filters, "top_k": top_k})
        return self.get_all_results.get(filters["user_id"], [])

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


class FakePlanner:
    def __init__(self, operations=None, raises=None):
        self.operations = operations or []
        self.raises = raises
        self.calls = []

    async def plan(self, *, user_message, context, user_memories, conversation_memories):
        self.calls.append(
            {
                "user_message": user_message,
                "context": context,
                "user_memories": user_memories,
                "conversation_memories": conversation_memories,
            }
        )
        if self.raises:
            raise self.raises
        return self.operations


def context():
    return build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )


def test_long_term_memory_prompt_section_labels_current_user_identity():
    bundle = LongTermMemoryBundle(
        user_memories=[
            LongTermMemoryRecord(id="mem-user-1", content="用户不吃辣", metadata={})
        ],
        conversation_memories=[
            LongTermMemoryRecord(id="mem-conv-1", content="当前会话默认中文", metadata={})
        ],
    )

    section = bundle.as_prompt_section(context())

    assert "相关个人长期记忆（当前发言者 QQ号：123456，昵称：阿咳）：" in section
    assert "- 用户不吃辣" in section
    assert "相关会话长期记忆：" in section
    assert "- 当前会话默认中文" in section


async def test_search_queries_user_and_conversation_memories():
    mem0 = FakeMem0Client()
    mem0.search_results = {
        "qq_user:123456": [
            {"id": "mem-user-1", "memory": "用户不吃辣", "metadata": {"kind": "preference"}}
        ],
        "qq_conversation:qq_group:888888": [
            {
                "id": "mem-conv-1",
                "memory": "当前会话默认中文",
                "metadata": {"kind": "conversation_rule"},
            }
        ],
    }
    service = LongTermMemoryService(mem0_client=mem0, planner=FakePlanner())

    bundle = await service.search("晚上吃川菜吗", context())

    assert bundle.user_memories == [
        LongTermMemoryRecord(
            id="mem-user-1",
            content="用户不吃辣",
            metadata={"kind": "preference"},
        )
    ]
    assert bundle.conversation_memories == [
        LongTermMemoryRecord(
            id="mem-conv-1",
            content="当前会话默认中文",
            metadata={"kind": "conversation_rule"},
        )
    ]
    assert [call["filters"]["user_id"] for call in mem0.search_calls] == [
        "qq_user:123456",
        "qq_conversation:qq_group:888888",
    ]
    assert [call["top_k"] for call in mem0.search_calls] == [5, 5]


async def test_search_raises_when_mem0_search_fails(monkeypatch):
    mem0 = FakeMem0Client()
    error = RuntimeError("mem0 unavailable")
    mem0.search_raises = error
    recorded_errors = []
    monkeypatch.setattr(
        "qq_group_chatter.services.long_term_memory.record_error",
        lambda stage, exc: recorded_errors.append({"stage": stage, "exc": exc}),
    )
    service = LongTermMemoryService(mem0_client=mem0, planner=FakePlanner())

    try:
        await service.search("晚上吃川菜吗", context())
    except LongTermMemorySearchError as exc:
        assert exc.__cause__ is error
    else:
        raise AssertionError("search should raise when Mem0 search fails")

    assert recorded_errors
    assert all(item == {"stage": "mem0_search", "exc": error} for item in recorded_errors)


async def test_start_raises_when_mem0_health_check_fails():
    mem0 = FakeMem0Client()
    error = RuntimeError("mem0 unavailable")
    mem0.search_raises = error
    service = LongTermMemoryService(mem0_client=mem0, planner=FakePlanner())

    try:
        await service.start()
    except LongTermMemorySearchError as exc:
        assert exc.__cause__ is error
    else:
        raise AssertionError("start should fail when Mem0 is unavailable")

    assert service._worker is None
    assert mem0.health_check_calls == [
        {
            "query": "__qq_group_chatter_startup_health_check__",
            "filters": {"user_id": "__qq_group_chatter_startup_health_check__"},
            "top_k": 1,
        }
    ]


def test_normalize_mem0_records_keeps_id_content_and_metadata():
    records = normalize_mem0_records(
        {
            "results": [
                "字符串旧格式",
                {
                    "id": "mem-1",
                    "content": "content 字段",
                    "metadata": {"source": "direct"},
                    "score": 0.9,
                },
                {
                    "id": "mem-2",
                    "payload": {
                        "text": "payload 文本",
                        "metadata": {"scope": "user"},
                        "kind": "preference",
                    },
                },
                {
                    "memory": "memory 字段",
                    "user_id": "qq_user:123456",
                },
            ]
        }
    )

    assert records == [
        LongTermMemoryRecord(id=None, content="字符串旧格式", metadata={}),
        LongTermMemoryRecord(
            id="mem-1",
            content="content 字段",
            metadata={"source": "direct", "score": 0.9},
        ),
        LongTermMemoryRecord(
            id="mem-2",
            content="payload 文本",
            metadata={"scope": "user", "kind": "preference"},
        ),
        LongTermMemoryRecord(
            id=None,
            content="memory 字段",
            metadata={"user_id": "qq_user:123456"},
        ),
    ]


async def test_ingestion_calls_planner_once_and_adds_operation_asynchronously():
    mem0 = FakeMem0Client()
    planner = FakePlanner(
        [
            LongTermMemoryOperation(
                action="add",
                scope="user",
                target_id=None,
                content="用户不吃辣",
                kind="preference",
                confidence=0.92,
            )
        ]
    )
    service = LongTermMemoryService(mem0_client=mem0, planner=planner)
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(
            context=context(),
            user_message="我不吃辣",
            existing_memories=LongTermMemoryBundle(user_memories=[], conversation_memories=[]),
        )
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert [call["user_message"] for call in planner.calls] == ["我不吃辣"]
    assert mem0.add_calls == [
        {
            "messages": [{"role": "user", "content": "用户不吃辣"}],
            "user_id": "qq_user:123456",
            "metadata": {
                "source": "qq",
                "conversation_id": "qq_group:888888",
                "conversation_type": "group",
                "message_id": "m1",
                "source_user_id": "123456",
                "source_nickname": "阿咳",
                "scope": "user",
                "kind": "preference",
                "source_created_at": 123.0,
                "last_seen_at": 123.0,
            },
            "infer": False,
        }
    ]
    assert mem0.search_calls == []
    assert mem0.get_all_calls == [
        {"filters": {"user_id": "qq_user:123456"}, "top_k": 1000}
    ]


async def test_ingestion_uses_job_existing_memories_for_duplicate_skip_without_search():
    mem0 = FakeMem0Client()
    planner = FakePlanner(
        [
            LongTermMemoryOperation(
                action="add",
                scope="conversation",
                target_id=None,
                content="当前会话默认使用中文交流",
                kind="conversation_rule",
                confidence=0.91,
            ),
        ]
    )
    service = LongTermMemoryService(mem0_client=mem0, planner=planner)
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(
            context=context(),
            user_message="这个群默认说中文",
            existing_memories=LongTermMemoryBundle(
                user_memories=[],
                conversation_memories=[
                    LongTermMemoryRecord(
                        id="mem-conv-1",
                        content="当前会话默认使用中文交流",
                        metadata={"kind": "conversation_rule"},
                    )
                ],
            ),
        )
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert mem0.search_calls == []
    assert mem0.add_calls == []
    assert len(planner.calls) == 1


async def test_ingestion_skips_duplicate_adds_within_same_planner_result():
    mem0 = FakeMem0Client()
    planner = FakePlanner(
        [
            LongTermMemoryOperation(
                action="add",
                scope="user",
                target_id=None,
                content="用户不吃辣",
                kind="preference",
                confidence=0.92,
            ),
            LongTermMemoryOperation(
                action="add",
                scope="user",
                target_id=None,
                content="用户不吃辣",
                kind="preference",
                confidence=0.92,
            ),
        ]
    )
    service = LongTermMemoryService(mem0_client=mem0, planner=planner)
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(
            context=context(),
            user_message="我不吃辣",
            existing_memories=LongTermMemoryBundle(user_memories=[], conversation_memories=[]),
        )
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert len(mem0.add_calls) == 1


async def test_ingestion_updates_existing_memory_when_planner_returns_update():
    mem0 = FakeMem0Client()
    planner = FakePlanner(
        [
            LongTermMemoryOperation(
                action="update",
                scope="user",
                target_id="mem-user-1",
                content="用户现在不吃辣",
                kind="preference",
                confidence=0.92,
            )
        ]
    )
    service = LongTermMemoryService(mem0_client=mem0, planner=planner)
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(
            context=context(),
            user_message="我现在不吃辣",
            existing_memories=LongTermMemoryBundle(
                user_memories=[
                    LongTermMemoryRecord(
                        id="mem-user-1",
                        content="用户喜欢吃辣",
                        metadata={"source_created_at": 100.0, "kind": "preference"},
                    )
                ],
                conversation_memories=[],
            ),
        )
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert planner.calls[0]["user_memories"][0].id == "mem-user-1"
    assert mem0.add_calls == []
    assert mem0.update_calls == [
        {
            "memory_id": "mem-user-1",
            "data": "用户现在不吃辣",
            "metadata": {
                "source_created_at": 100.0,
                "kind": "preference",
                "source": "qq",
                "conversation_id": "qq_group:888888",
                "conversation_type": "group",
                "message_id": "m1",
                "source_user_id": "123456",
                "source_nickname": "阿咳",
                "scope": "user",
                "last_seen_at": 123.0,
                "last_seen_message_id": "m1",
            },
        }
    ]


async def test_ingestion_update_drops_mem0_reserved_timestamp_metadata():
    mem0 = FakeMem0Client()
    planner = FakePlanner(
        [
            LongTermMemoryOperation(
                action="update",
                scope="user",
                target_id="mem-user-1",
                content="用户现在不吃辣",
                kind="preference",
                confidence=0.92,
            )
        ]
    )
    service = LongTermMemoryService(mem0_client=mem0, planner=planner)
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(
            context=context(),
            user_message="我现在不吃辣",
            existing_memories=LongTermMemoryBundle(
                user_memories=[
                    LongTermMemoryRecord(
                        id="mem-user-1",
                        content="用户喜欢吃辣",
                        metadata={
                            "created_at": 1781529229.0,
                            "updated_at": 1781529230.0,
                            "kind": "preference",
                        },
                    )
                ],
                conversation_memories=[],
            ),
        )
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    metadata = mem0.update_calls[0]["metadata"]
    assert metadata["source_created_at"] == 1781529229.0
    assert "created_at" not in metadata
    assert "updated_at" not in metadata


async def test_ingestion_adds_when_planner_returns_add():
    mem0 = FakeMem0Client()
    planner = FakePlanner(
        [
            LongTermMemoryOperation(
                action="add",
                scope="user",
                target_id=None,
                content="用户不吃辣",
                kind="preference",
                confidence=0.92,
            )
        ]
    )
    service = LongTermMemoryService(mem0_client=mem0, planner=planner)
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(
            context=context(),
            user_message="我不吃辣",
            existing_memories=LongTermMemoryBundle(
                user_memories=[
                    LongTermMemoryRecord(
                        id="mem-user-1",
                        content="用户喜欢吃辣",
                        metadata={"created_at": 100.0},
                    )
                ],
                conversation_memories=[],
            ),
        )
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert mem0.add_calls[0]["messages"] == [{"role": "user", "content": "用户不吃辣"}]
    assert mem0.update_calls == []


async def test_ingestion_skips_when_planner_returns_skip():
    mem0 = FakeMem0Client()
    planner = FakePlanner(
        [
            LongTermMemoryOperation(
                action="skip",
                scope="user",
                target_id=None,
                content="用户不吃辣",
                kind="preference",
                confidence=0.92,
            )
        ]
    )
    service = LongTermMemoryService(mem0_client=mem0, planner=planner)
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(
            context=context(),
            user_message="我不吃辣",
            existing_memories=LongTermMemoryBundle(
                user_memories=[
                    LongTermMemoryRecord(
                        id="mem-user-1",
                        content="用户喜欢吃辣",
                        metadata={},
                    )
                ],
                conversation_memories=[],
            ),
        )
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert mem0.add_calls == []
    assert mem0.update_calls == []


async def test_ingestion_deletes_existing_memory_when_planner_returns_delete():
    mem0 = FakeMem0Client()
    planner = FakePlanner(
        [
            LongTermMemoryOperation(
                action="delete",
                scope="user",
                target_id="mem-user-1",
                content="",
                kind="preference",
                confidence=0.92,
            )
        ]
    )
    service = LongTermMemoryService(mem0_client=mem0, planner=planner)
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(
            context=context(),
            user_message="忘掉我喜欢吃辣这件事",
            existing_memories=LongTermMemoryBundle(
                user_memories=[
                    LongTermMemoryRecord(
                        id="mem-user-1",
                        content="用户喜欢吃辣",
                        metadata={"kind": "preference"},
                    )
                ],
                conversation_memories=[],
            ),
        )
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert mem0.delete_calls == [{"memory_id": "mem-user-1"}]
    assert mem0.add_calls == []
    assert mem0.update_calls == []
    assert mem0.get_all_calls == []


async def test_ingestion_does_not_delete_missing_target():
    mem0 = FakeMem0Client()
    planner = FakePlanner(
        [
            LongTermMemoryOperation(
                action="delete",
                scope="user",
                target_id="missing",
                content="",
                kind="preference",
                confidence=0.92,
            )
        ]
    )
    service = LongTermMemoryService(mem0_client=mem0, planner=planner)
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(
            context=context(),
            user_message="忘掉不存在的记忆",
            existing_memories=LongTermMemoryBundle(
                user_memories=[
                    LongTermMemoryRecord(
                        id="mem-user-1",
                        content="用户喜欢吃辣",
                        metadata={"kind": "preference"},
                    )
                ],
                conversation_memories=[],
            ),
        )
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert mem0.delete_calls == []
    assert mem0.add_calls == []
    assert mem0.update_calls == []
    assert mem0.get_all_calls == []


async def test_ingestion_records_delete_error_without_escaping(monkeypatch):
    class DeleteFailingMem0Client(FakeMem0Client):
        def delete(self, memory_id):
            super().delete(memory_id)
            raise RuntimeError("delete failed")

    mem0 = DeleteFailingMem0Client()
    error_records = []
    monkeypatch.setattr(
        "qq_group_chatter.services.long_term_memory.record_error",
        lambda stage, exc: error_records.append((stage, str(exc))),
    )
    planner = FakePlanner(
        [
            LongTermMemoryOperation(
                action="delete",
                scope="user",
                target_id="mem-user-1",
                content="用户喜欢吃辣",
                kind="preference",
                confidence=0.92,
            )
        ]
    )
    service = LongTermMemoryService(mem0_client=mem0, planner=planner)
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(
            context=context(),
            user_message="忘掉我喜欢吃辣这件事",
            existing_memories=LongTermMemoryBundle(
                user_memories=[
                    LongTermMemoryRecord(
                        id="mem-user-1",
                        content="用户喜欢吃辣",
                        metadata={"kind": "preference"},
                    )
                ],
                conversation_memories=[],
            ),
        )
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert mem0.delete_calls == [{"memory_id": "mem-user-1"}]
    assert error_records == [("mem0_delete", "delete failed")]


async def test_ingestion_skips_sensitive_operations_before_mem0_write():
    mem0 = FakeMem0Client()
    planner = FakePlanner(
        [
            LongTermMemoryOperation(
                action="add",
                scope="user",
                target_id=None,
                content="用户密码是 abc123",
                kind="other",
                confidence=0.95,
            ),
            LongTermMemoryOperation(
                action="add",
                scope="user",
                target_id=None,
                content="用户 token 是 sk-abcdef123456",
                kind="other",
                confidence=0.95,
            ),
            LongTermMemoryOperation(
                action="add",
                scope="conversation",
                target_id=None,
                content="用户住在上海市浦东新区测试路 1 号",
                kind="other",
                confidence=0.95,
            ),
            LongTermMemoryOperation(
                action="add",
                scope="conversation",
                target_id=None,
                content="api key 是 secret-key-value",
                kind="other",
                confidence=0.95,
            ),
        ]
    )
    service = LongTermMemoryService(mem0_client=mem0, planner=planner)
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(
            context=context(),
            user_message="记住这些敏感内容",
            existing_memories=LongTermMemoryBundle(user_memories=[], conversation_memories=[]),
        )
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert mem0.add_calls == []
    assert mem0.update_calls == []


async def test_ingestion_skips_invalid_operations_without_consuming_write_limit():
    mem0 = FakeMem0Client()
    planner = FakePlanner(
        [
            LongTermMemoryOperation(
                action="add",
                scope="user",
                target_id=None,
                content="用户密码是 abc123",
                kind="other",
                confidence=0.95,
            ),
            LongTermMemoryOperation(
                action="add",
                scope="user",
                target_id=None,
                content="用户 token 是 sk-abcdef123456",
                kind="other",
                confidence=0.95,
            ),
            LongTermMemoryOperation(
                action="add",
                scope="user",
                target_id=None,
                content="用户不吃辣",
                kind="preference",
                confidence=0.95,
            ),
        ]
    )
    service = LongTermMemoryService(mem0_client=mem0, planner=planner, max_operations_per_message=2)
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(
            context=context(),
            user_message="记住我不吃辣",
            existing_memories=LongTermMemoryBundle(user_memories=[], conversation_memories=[]),
        )
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert [call["messages"][0]["content"] for call in mem0.add_calls] == ["用户不吃辣"]


async def test_ingestion_skips_formatted_phone_numbers_before_mem0_write():
    mem0 = FakeMem0Client()
    planner = FakePlanner(
        [
            LongTermMemoryOperation(
                action="add",
                scope="user",
                target_id=None,
                content="用户手机号是 138 0013 8000",
                kind="other",
                confidence=0.95,
            ),
            LongTermMemoryOperation(
                action="add",
                scope="conversation",
                target_id=None,
                content="联系方式：138-0013-8000",
                kind="other",
                confidence=0.95,
            ),
        ]
    )
    service = LongTermMemoryService(mem0_client=mem0, planner=planner)
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(
            context=context(),
            user_message="记住联系方式",
            existing_memories=LongTermMemoryBundle(user_memories=[], conversation_memories=[]),
        )
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert mem0.add_calls == []
    assert mem0.update_calls == []


async def test_ingestion_does_not_update_record_without_id():
    mem0 = FakeMem0Client()
    planner = FakePlanner(
        [
            LongTermMemoryOperation(
                action="update",
                scope="user",
                target_id=None,
                content="用户不吃辣",
                kind="preference",
                confidence=0.92,
            )
        ]
    )
    service = LongTermMemoryService(mem0_client=mem0, planner=planner)
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(
            context=context(),
            user_message="我不吃辣",
            existing_memories=LongTermMemoryBundle(
                user_memories=[
                    LongTermMemoryRecord(id=None, content="用户喜欢吃辣", metadata={})
                ],
                conversation_memories=[],
            ),
        )
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert mem0.update_calls == []
    assert mem0.add_calls == []


async def test_planner_errors_do_not_write_memories(monkeypatch):
    mem0 = FakeMem0Client()
    error = RuntimeError("planner failed")
    recorded_errors = []
    monkeypatch.setattr(
        "qq_group_chatter.services.long_term_memory.record_error",
        lambda stage, exc: recorded_errors.append({"stage": stage, "exc": exc}),
    )
    service = LongTermMemoryService(
        mem0_client=mem0,
        planner=FakePlanner(raises=error),
    )
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(
            context=context(),
            user_message="我不吃辣",
            existing_memories=LongTermMemoryBundle(
                user_memories=[
                    LongTermMemoryRecord(
                        id="mem-user-1",
                        content="用户喜欢吃辣",
                        metadata={},
                    )
                ],
                conversation_memories=[],
            ),
        )
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert mem0.add_calls == []
    assert mem0.update_calls == []
    assert recorded_errors == [{"stage": "long_term_memory_planner", "exc": error}]


async def test_prunes_oldest_memories_when_scope_exceeds_limit():
    mem0 = FakeMem0Client()
    mem0.get_all_results = {
        "qq_user:123456": {
            "results": [
                {
                    "id": "oldest",
                    "memory": "最旧记忆",
                    "metadata": {"created_at": 1.0},
                },
                {
                    "id": "newer",
                    "memory": "较新记忆",
                    "metadata": {"created_at": 2.0},
                },
                {
                    "id": "newest",
                    "memory": "最新记忆",
                    "metadata": {"created_at": 3.0},
                },
            ]
        }
    }
    planner = FakePlanner(
        [
            LongTermMemoryOperation(
                action="add",
                scope="user",
                target_id=None,
                content="新增记忆",
                kind="other",
                confidence=0.92,
            )
        ]
    )
    service = LongTermMemoryService(
        mem0_client=mem0,
        planner=planner,
        max_records_per_scope=2,
    )
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(context=context(), user_message="记住这个")
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert mem0.delete_calls == [{"memory_id": "oldest"}]


async def test_does_not_prune_when_scope_is_within_limit():
    mem0 = FakeMem0Client()
    mem0.get_all_results = {
        "qq_user:123456": {
            "results": [
                {
                    "id": "memory-1",
                    "memory": "已有记忆",
                    "metadata": {"created_at": 1.0},
                }
            ]
        }
    }
    planner = FakePlanner(
        [
            LongTermMemoryOperation(
                action="add",
                scope="user",
                target_id=None,
                content="新增记忆",
                kind="other",
                confidence=0.92,
            )
        ]
    )
    service = LongTermMemoryService(
        mem0_client=mem0,
        planner=planner,
        max_records_per_scope=2,
    )
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(context=context(), user_message="记住这个")
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert mem0.delete_calls == []


async def test_ingestion_falls_back_to_mem0_search_without_existing_memories():
    mem0 = FakeMem0Client()
    mem0.search_results = {
        "qq_conversation:qq_group:888888": [{"memory": "当前会话默认使用中文交流"}]
    }
    planner = FakePlanner(
        [
            LongTermMemoryOperation(
                action="add",
                scope="conversation",
                target_id=None,
                content="当前会话默认使用中文交流",
                kind="conversation_rule",
                confidence=0.91,
            ),
        ]
    )
    service = LongTermMemoryService(mem0_client=mem0, planner=planner)
    await service.start()

    await service.enqueue_ingestion(
        LongTermMemoryIngestionJob(context=context(), user_message="这个群默认说中文")
    )
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()

    assert mem0.add_calls == []
    assert mem0.search_calls == [
        {
            "query": "这个群默认说中文",
            "filters": {"user_id": "qq_user:123456"},
            "top_k": 5,
        },
        {
            "query": "这个群默认说中文",
            "filters": {"user_id": "qq_conversation:qq_group:888888"},
            "top_k": 5,
        }
    ]


async def test_worker_errors_do_not_escape():
    service = LongTermMemoryService(
        mem0_client=FakeMem0Client(),
        planner=FakePlanner(raises=RuntimeError("planner failed")),
    )
    await service.start()

    await service.enqueue_ingestion(LongTermMemoryIngestionJob(context=context(), user_message="我不吃辣"))
    await asyncio.wait_for(service.join(), timeout=1)
    await service.stop()


async def test_stop_closes_mem0_client_when_supported():
    mem0 = FakeMem0Client()
    service = LongTermMemoryService(mem0_client=mem0, planner=FakePlanner())
    await service.start()

    await service.stop()

    assert mem0.close_calls == 1


async def test_stop_closes_mem0_vector_store_client_when_supported():
    mem0 = FakeMem0ClientWithVectorStore()
    service = LongTermMemoryService(mem0_client=mem0, planner=FakePlanner())
    await service.start()

    await service.stop()

    assert mem0.vector_store.client.close_calls == 1
