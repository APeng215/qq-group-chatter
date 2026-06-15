from types import SimpleNamespace

import pytest

from qq_group_chatter.models import PendingAssistantReply, build_group_conversation_context
from qq_group_chatter.plugins.chat import _send_reply_and_record, should_handle_message


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
    assert should_handle_message(event(user_id=654321, self_id=654321), "你好") is False


def test_should_ignore_empty_message():
    assert should_handle_message(event(), "   ") is False


def test_should_ignore_group_message_not_addressed_to_bot():
    assert should_handle_message(event(message_type="group", to_me=False), "大家好") is False


def test_should_handle_private_message():
    assert should_handle_message(event(message_type="private", to_me=False), "你好") is True


def test_should_handle_group_message_addressed_to_bot():
    assert should_handle_message(event(message_type="group", to_me=True), "你好") is True


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

    async def record_assistant_reply(self, reply):
        self.recorded.append(reply)


def pending_reply():
    return PendingAssistantReply(
        context=build_group_conversation_context(
            group_id=888888,
            user_id=123456,
            message_id="m1",
            nickname="阿咳",
            timestamp=123.0,
        ),
        content="好的",
        timestamp=124.0,
    )


async def test_send_reply_records_assistant_after_send_success():
    bot = FakeBot()
    orchestrator = FakeOrchestrator()
    source_event = event(to_me=True)
    reply = pending_reply()

    await _send_reply_and_record(bot, source_event, reply, orchestrator)

    assert bot.sent == [{"event": source_event, "message": "好的"}]
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
