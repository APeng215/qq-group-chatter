from types import SimpleNamespace

import pytest

from qq_group_chatter.models import PendingAssistantReply, build_group_conversation_context
from qq_group_chatter.plugins.chat import (
    _context_from_event,
    _handle_message_event,
    _handle_regular_chat,
    _message_text_from_event,
    _send_reply_and_record,
    should_handle_message,
    should_record_message,
    should_reply_to_message,
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
    assert should_record_message(event(user_id=654321, self_id=654321), "hello") is False


def test_should_ignore_empty_message():
    assert should_record_message(event(), "   ") is False


def test_should_record_group_message_not_addressed_to_bot_without_replying():
    source_event = event(message_type="group", to_me=False)

    assert should_record_message(source_event, "hello") is True
    assert should_reply_to_message(source_event) is False
    assert should_handle_message(source_event, "hello") is False


def test_should_handle_private_message():
    source_event = event(message_type="private", to_me=False)

    assert should_record_message(source_event, "hello") is True
    assert should_reply_to_message(source_event) is True
    assert should_handle_message(source_event, "hello") is True


def test_should_handle_group_message_addressed_to_bot():
    source_event = event(message_type="group", to_me=True)

    assert should_record_message(source_event, "hello") is True
    assert should_reply_to_message(source_event) is True
    assert should_handle_message(source_event, "hello") is True


def test_message_text_from_event_keeps_image_placeholder():
    source_event = event(
        message=[
            {"type": "image", "data": {"file": "abc.jpg"}},
        ]
    )

    assert _message_text_from_event(source_event) == "[图片]"


def test_message_text_from_event_keeps_text_and_image_placeholder():
    source_event = event(
        message=[
            {"type": "text", "data": {"text": "看这个"}},
            {"type": "image", "data": {"file": "abc.jpg"}},
        ]
    )

    assert _message_text_from_event(source_event) == "看这个 [图片]"


def test_context_from_group_event_uses_group_card_before_nickname(monkeypatch):
    class FakeGroupMessageEvent:
        pass

    class FakePrivateMessageEvent:
        pass

    monkeypatch.setattr("qq_group_chatter.plugins.chat.GroupMessageEvent", FakeGroupMessageEvent)
    monkeypatch.setattr("qq_group_chatter.plugins.chat.PrivateMessageEvent", FakePrivateMessageEvent)
    source_event = FakeGroupMessageEvent()
    source_event.group_id = 888888
    source_event.user_id = 123456
    source_event.message_id = "m1"
    source_event.time = 123.0
    source_event.sender = SimpleNamespace(card="群名片", nickname="QQ昵称")

    context = _context_from_event(source_event)

    assert context is not None
    assert context.nickname == "群名片"


def test_context_from_group_event_falls_back_to_nickname_when_card_is_blank(monkeypatch):
    class FakeGroupMessageEvent:
        pass

    class FakePrivateMessageEvent:
        pass

    monkeypatch.setattr("qq_group_chatter.plugins.chat.GroupMessageEvent", FakeGroupMessageEvent)
    monkeypatch.setattr("qq_group_chatter.plugins.chat.PrivateMessageEvent", FakePrivateMessageEvent)
    source_event = FakeGroupMessageEvent()
    source_event.group_id = 888888
    source_event.user_id = 123456
    source_event.message_id = "m1"
    source_event.time = 123.0
    source_event.sender = SimpleNamespace(card="  ", nickname="QQ昵称")

    context = _context_from_event(source_event)

    assert context is not None
    assert context.nickname == "QQ昵称"


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
        self.recorded_user_messages = []
        self.handle_calls = []

    async def record_assistant_reply(self, reply):
        self.recorded.append(reply)

    async def record_user_message(self, *, context, user_message):
        self.recorded_user_messages.append(
            {
                "context": context,
                "user_message": user_message,
            }
        )

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


def context():
    return build_group_conversation_context(
        group_id=888888,
        user_id=123456,
        message_id="m1",
        nickname="alice",
        timestamp=123.0,
    )


async def test_group_message_not_addressed_to_bot_is_recorded_without_reply(monkeypatch):
    class FakeGroupMessageEvent:
        pass

    class FakePrivateMessageEvent:
        pass

    monkeypatch.setattr("qq_group_chatter.plugins.chat.GroupMessageEvent", FakeGroupMessageEvent)
    monkeypatch.setattr("qq_group_chatter.plugins.chat.PrivateMessageEvent", FakePrivateMessageEvent)
    source_event = FakeGroupMessageEvent()
    source_event.message_type = "group"
    source_event.to_me = False
    source_event.self_id = 654321
    source_event.group_id = 888888
    source_event.user_id = 123456
    source_event.message_id = "m2"
    source_event.time = 123.0
    source_event.sender = SimpleNamespace(card="alice", nickname="alice")
    source_event.message = [{"type": "text", "data": {"text": "刚刚说的上下文"}}]

    bot = FakeBot()
    orchestrator = FakeOrchestrator()

    await _handle_message_event(bot, source_event, orchestrator)

    assert orchestrator.recorded_user_messages[0]["user_message"] == "刚刚说的上下文"
    assert orchestrator.handle_calls == []
    assert bot.sent == []


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


async def test_search_phrase_goes_through_regular_chat():
    bot = FakeBot()
    orchestrator = FakeOrchestrator()
    source_event = event(to_me=True)

    await _handle_regular_chat(
        bot,
        source_event,
        context(),
        "搜一下 DeepSeek 最新消息",
        orchestrator,
    )

    assert orchestrator.handle_calls[0]["user_message"] == "搜一下 DeepSeek 最新消息"
    assert [item["message"] for item in bot.sent] == [
        "我查一下再回你。",
        "搜索最终答案",
    ]
