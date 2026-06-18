from qq_group_chatter.models import ChatMessage, build_group_conversation_context
from qq_group_chatter.services.conversation_archive import ConversationArchiveService


class FakeMem0Client:
    def __init__(self, search_results=None):
        self.add_calls = []
        self.search_calls = []
        self.search_results = search_results or []

    def add(self, messages, *, user_id, metadata=None, infer=True):
        self.add_calls.append(
            {
                "messages": messages,
                "user_id": user_id,
                "metadata": metadata,
                "infer": infer,
            }
        )
        return {"id": "archive-1"}

    def search(self, query, *, filters=None, top_k=None):
        self.search_calls.append({"query": query, "filters": filters, "top_k": top_k})
        return self.search_results


def message(*, content="苹果太酸了", message_id="m1", timestamp=1000.0):
    return ChatMessage(
        conversation_id="qq_group:888888",
        role="user",
        content=content,
        user_id="123456",
        nickname="阿咳",
        message_id=message_id,
        timestamp=timestamp,
    )


def context():
    return build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m2",
        nickname="阿咳",
        timestamp=2000.0,
    )


async def test_archive_enqueue_writes_message_with_archive_metadata():
    mem0 = FakeMem0Client()
    service = ConversationArchiveService(mem0_client=mem0)

    await service.enqueue_message(message())
    await service.join()

    assert mem0.add_calls == [
        {
            "messages": [{"role": "user", "content": "苹果太酸了"}],
            "user_id": "qq_archive:qq_group:888888",
            "metadata": {
                "archive_type": "conversation_message",
                "conversation_id": "qq_group:888888",
                "conversation_type": "group",
                "source_user_id": "123456",
                "source_nickname": "阿咳",
                "role": "user",
                "message_id": "m1",
                "timestamp": 1000.0,
            },
            "infer": False,
        }
    ]


async def test_archive_search_filters_current_conversation_and_returns_records():
    mem0 = FakeMem0Client(
        search_results=[
            {
                "id": "r1",
                "memory": "苹果太酸了",
                "score": 0.91,
                "metadata": {
                    "conversation_id": "qq_group:888888",
                    "source_user_id": "123456",
                    "source_nickname": "阿咳",
                    "role": "user",
                    "message_id": "m1",
                    "timestamp": 1000.0,
                },
            }
        ]
    )
    service = ConversationArchiveService(mem0_client=mem0)

    records = await service.search("苹果", context())

    assert mem0.search_calls == [
        {
            "query": "苹果",
            "filters": {
                "user_id": "qq_archive:qq_group:888888",
                "conversation_id": "qq_group:888888",
                "archive_type": "conversation_message",
            },
            "top_k": 20,
        }
    ]
    assert len(records) == 1
    assert records[0].content == "苹果太酸了"
    assert records[0].user_id == "123456"
    assert records[0].score is not None


async def test_archive_search_excludes_current_message():
    mem0 = FakeMem0Client(
        search_results=[
            {
                "id": "current",
                "memory": "苹果",
                "score": 0.99,
                "metadata": {
                    "conversation_id": "qq_group:888888",
                    "source_user_id": "123456",
                    "source_nickname": "阿咳",
                    "role": "user",
                    "message_id": "m2",
                    "timestamp": 2000.0,
                },
            },
            {
                "id": "old",
                "memory": "我买的苹果太酸了",
                "score": 0.91,
                "metadata": {
                    "conversation_id": "qq_group:888888",
                    "source_user_id": "123456",
                    "source_nickname": "阿咳",
                    "role": "user",
                    "message_id": "m1",
                    "timestamp": 1000.0,
                },
            },
        ]
    )
    service = ConversationArchiveService(mem0_client=mem0)

    records = await service.search("苹果", context(), limit=2)

    assert [record.message_id for record in records] == ["m1"]


async def test_archive_rerank_keeps_old_highly_relevant_record_above_recent_weak_record():
    mem0 = FakeMem0Client(
        search_results=[
            {
                "id": "old",
                "memory": "我买的苹果太酸了",
                "score": 0.95,
                "metadata": {
                    "conversation_id": "qq_group:888888",
                    "source_user_id": "123456",
                    "source_nickname": "阿咳",
                    "role": "user",
                    "message_id": "old",
                    "timestamp": 1000.0,
                },
            },
            {
                "id": "new",
                "memory": "我今天吃了香蕉",
                "score": 0.70,
                "metadata": {
                    "conversation_id": "qq_group:888888",
                    "source_user_id": "123456",
                    "source_nickname": "阿咳",
                    "role": "user",
                    "message_id": "new",
                    "timestamp": 2000.0,
                },
            },
        ]
    )
    service = ConversationArchiveService(
        mem0_client=mem0,
        semantic_weight=0.9,
        recency_weight=0.1,
        time_decay_days=90,
    )

    records = await service.search("苹果", context(), now=2000.0, limit=2)

    assert [record.message_id for record in records] == ["old", "new"]
