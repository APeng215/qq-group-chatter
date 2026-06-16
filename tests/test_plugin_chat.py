from types import SimpleNamespace

import pytest

from qq_group_chatter.models import PendingAssistantReply, build_group_conversation_context
from qq_group_chatter.plugins.chat import (
    _handle_regular_chat,
    _send_reply_and_record,
    _send_search_reply,
    setup_search_service,
    should_handle_message,
)


def event(**kwargs):
    defaults = {
        "user_id": 123456,
        "self_id": 654321,
        "message_type": "group",
        "to_me": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_should_ignore_bot_own_message():
    assert should_handle_message(event(user_id=654321, self_id=654321), "hello") is False


def test_should_ignore_empty_message():
    assert should_handle_message(event(), "   ") is False


def test_should_ignore_group_message_not_addressed_to_bot():
    assert should_handle_message(event(message_type="group", to_me=False), "hello") is False


def test_should_handle_private_message():
    assert should_handle_message(event(message_type="private", to_me=False), "hello") is True


def test_should_handle_group_message_addressed_to_bot():
    assert should_handle_message(event(message_type="group", to_me=True), "hello") is True


class FakeBot:
    def __init__(self, raises=None):
        self.raises = raises
        self.sent = []

    async def send(self, event, message):
        if self.raises:
            raise self.raises
        self.sent.append({"event": event, "message": message})


class FakeOrchestrator:
    def __init__(self):
        self.recorded = []
        self.handle_calls = []

    async def record_assistant_reply(self, reply):
        self.recorded.append(reply)

    async def handle_message(self, *, context, user_message, on_search_start=None):
        self.handle_calls.append(
            {
                "context": context,
                "user_message": user_message,
                "on_search_start": on_search_start,
            }
        )
        if on_search_start is not None:
            await on_search_start("我查一下再回你。")
        return PendingAssistantReply(
            context=context,
            content="搜索最终答案",
            timestamp=125.0,
        )


class FakeSearchService:
    def __init__(self):
        self.queries = []

    async def search_reply(self, query):
        self.queries.append(query)
        return f"answer: {query}"


def context():
    return build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="alice",
        timestamp=123.0,
    )


def pending_reply():
    return PendingAssistantReply(
        context=context(),
        content="ok",
        timestamp=124.0,
    )


async def test_send_reply_records_assistant_after_send_success():
    bot = FakeBot()
    orchestrator = FakeOrchestrator()
    source_event = event(to_me=True)
    reply = pending_reply()

    await _send_reply_and_record(bot, source_event, reply, orchestrator)

    assert bot.sent == [{"event": source_event, "message": "ok"}]
    assert orchestrator.recorded == [reply]


async def test_send_reply_logs_error_and_does_not_record_when_send_fails(monkeypatch):
    error = RuntimeError("send failed")
    bot = FakeBot(raises=error)
    orchestrator = FakeOrchestrator()
    recorded_errors = []
    monkeypatch.setattr(
        "qq_group_chatter.plugins.chat.record_error",
        lambda stage, exc: recorded_errors.append({"stage": stage, "exc": exc}),
    )

    with pytest.raises(RuntimeError, match="send failed"):
        await _send_reply_and_record(bot, event(to_me=True), pending_reply(), orchestrator)

    assert orchestrator.recorded == []
    assert recorded_errors == [{"stage": "send_reply", "exc": error}]


async def test_send_search_reply_uses_search_service_and_records_assistant_reply():
    bot = FakeBot()
    service = FakeSearchService()
    orchestrator = FakeOrchestrator()
    source_event = event(to_me=True)

    handled = await _send_search_reply(
        bot,
        source_event,
        context(),
        "搜一下 DeepSeek 最新消息",
        search_service=service,
        orchestrator=orchestrator,
    )

    assert handled is True
    assert service.queries == ["DeepSeek 最新消息"]
    assert bot.sent == [
        {"event": source_event, "message": "我先搜一下，稍等。"},
        {"event": source_event, "message": "answer: DeepSeek 最新消息"},
    ]
    assert orchestrator.recorded[0].content == "answer: DeepSeek 最新消息"


async def test_send_search_reply_ignores_regular_chat_and_slash_command():
    for text in ["普通聊天", "/搜 DeepSeek 最新消息"]:
        handled = await _send_search_reply(
            FakeBot(),
            event(to_me=True),
            context(),
            text,
            search_service=FakeSearchService(),
            orchestrator=FakeOrchestrator(),
        )

        assert handled is False


async def test_send_search_reply_logs_send_failure_without_raising(monkeypatch):
    error = RuntimeError("send failed")
    bot = FakeBot(raises=error)
    service = FakeSearchService()
    orchestrator = FakeOrchestrator()
    recorded_errors = []
    monkeypatch.setattr(
        "qq_group_chatter.plugins.chat.record_error",
        lambda stage, exc: recorded_errors.append({"stage": stage, "exc": exc}),
    )

    handled = await _send_search_reply(
        bot,
        event(to_me=True),
        context(),
        "搜索 DeepSeek 最新消息",
        search_service=service,
        orchestrator=orchestrator,
    )

    assert handled is True
    assert orchestrator.recorded == []
    assert recorded_errors == [
        {"stage": "web_search_notice_send", "exc": error},
        {"stage": "web_search_send", "exc": error},
    ]


async def test_send_search_reply_sends_notice_before_search_failure(monkeypatch):
    class FailingSearchService:
        async def search_reply(self, query):
            raise RuntimeError("search failed")

    bot = FakeBot()
    orchestrator = FakeOrchestrator()
    recorded_errors = []
    source_event = event(to_me=True)
    monkeypatch.setattr(
        "qq_group_chatter.plugins.chat.record_error",
        lambda stage, exc: recorded_errors.append({"stage": stage, "exc": exc}),
    )

    handled = await _send_search_reply(
        bot,
        source_event,
        context(),
        "搜索 DeepSeek 最新消息",
        search_service=FailingSearchService(),
        orchestrator=orchestrator,
    )

    assert handled is True
    assert [item["message"] for item in bot.sent] == [
        "我先搜一下，稍等。",
        "搜索失败，稍后再试。",
    ]
    assert orchestrator.recorded[0].content == "搜索失败，稍后再试。"
    assert recorded_errors[0]["stage"] == "web_search"


async def test_regular_chat_sends_llm_search_notice_before_final_reply():
    bot = FakeBot()
    orchestrator = FakeOrchestrator()
    source_event = event(to_me=True)

    await _handle_regular_chat(
        bot,
        source_event,
        context(),
        "DeepSeek 今天有什么新闻？",
        orchestrator,
    )

    assert [item["message"] for item in bot.sent] == [
        "我查一下再回你。",
        "搜索最终答案",
    ]
    assert orchestrator.recorded[0].content == "搜索最终答案"


def test_setup_search_service_sets_module_global():
    service = FakeSearchService()

    setup_search_service(service)

    import qq_group_chatter.plugins.chat as chat_plugin

    assert chat_plugin.search_service is service
