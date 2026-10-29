from __future__ import annotations

from time import time
from qq_group_chatter.agent.tools import setting

CHAT_KEYWORD_TRIGGER_ENABLED = setting("CHAT_KEYWORD_TRIGGER_ENABLED", "true").lower() not in {"0", "false", "off", "no"}
from qq_group_chatter.models import PendingAssistantReply, ImageAttachment
from qq_group_chatter.observability import record_error
from qq_group_chatter.orchestrator import ChatOrchestrator


try:
    from nonebot import get_driver, on_message
    from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, MessageEvent, PrivateMessageEvent
except Exception:  # pragma: no cover - lets service tests run without adapter setup
    get_driver = None
    on_message = None
    Bot = object
    MessageEvent = object
    GroupMessageEvent = object
    PrivateMessageEvent = object


orchestrator: ChatOrchestrator | None = None


def setup_orchestrator(instance: ChatOrchestrator) -> None:
    global orchestrator
    orchestrator = instance


if on_message is not None:
    matcher = on_message(priority=50, block=False)

    @matcher.handle()
    async def handle_nonebot_message(bot: Bot, event: MessageEvent) -> None:
        if orchestrator is None:
            return
        await _handle_message_event(bot, event, orchestrator)


async def _handle_message_event(bot: Bot, event: MessageEvent, orchestrator: ChatOrchestrator) -> None:
        text = _message_text_from_event(event)
        if not should_record_message(event, text):
            return
        context = _context_from_event(event)
        if context is None:
            return
        if not should_reply_to_message(event):
            attachments = _image_attachments_from_event(event)
            await orchestrator.record_user_message(context=context, user_message=text,
                **({"attachments": attachments} if attachments else {}))
            return
        await _handle_regular_chat(bot, event, context, text, orchestrator)


def _context_from_event(event) -> ConversationContext | None:
    from qq_group_chatter.models import (
        build_group_conversation_context,
        build_private_conversation_context,
    )

    timestamp = float(getattr(event, "time", time()))
    message_id = str(getattr(event, "message_id", ""))
    reply_to_message_id = _reply_to_message_id_from_event(event)
    user_id = str(getattr(event, "user_id", ""))
    bot_user_id = _optional_event_text(getattr(event, "self_id", None))
    sender = getattr(event, "sender", None)
    nickname = _sender_text(sender, "nickname")

    if GroupMessageEvent is not object and isinstance(event, GroupMessageEvent):
        display_name = _sender_text(sender, "card") or nickname
        return build_group_conversation_context(
            group_id=getattr(event, "group_id"),
            user_id=user_id,
            message_id=message_id,
            nickname=display_name,
            timestamp=timestamp,
            is_addressed_to_bot=_is_addressed_to_bot(event),
            reply_to_message_id=reply_to_message_id,
            bot_user_id=bot_user_id,
            bot_nickname="神奈",
        )
    if PrivateMessageEvent is not object and isinstance(event, PrivateMessageEvent):
        return build_private_conversation_context(
            user_id=user_id,
            message_id=message_id,
            nickname=nickname,
            timestamp=timestamp,
            is_addressed_to_bot=True,
            reply_to_message_id=reply_to_message_id,
            bot_user_id=bot_user_id,
            bot_nickname="神奈",
        )
    return None


def _optional_event_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sender_text(sender, field: str) -> str | None:
    if sender is None:
        return None
    value = getattr(sender, field, None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _message_text_from_event(event) -> str:
    segments = getattr(event, "message", None)
    if segments is None:
        text = str(event.get_plaintext()).strip()
        return text or _bare_mention_text(event)

    parts: list[str] = []
    for segment in segments:
        segment_type = _segment_value(segment, "type")
        data = _segment_value(segment, "data") or {}
        if segment_type == "text":
            text = _segment_value(data, "text")
            if text is not None and str(text).strip():
                parts.append(str(text).strip())
        elif segment_type == "image":
            parts.append("[图片]")

    if parts:
        return " ".join(parts)
    text = str(event.get_plaintext()).strip()
    return text or _bare_mention_text(event)


def _is_addressed_to_bot(event) -> bool:
    original = getattr(event, "original_message", None)
    if original is not None:
        parts = [str(_segment_value(_segment_value(s, "data") or {}, "text") or "")
                 for s in original if _segment_value(s, "type") == "text"]
        original_text = "".join(parts).lstrip()
    else:
        parts = [str(_segment_value(_segment_value(s, "data") or {}, "text") or "")
                 for s in getattr(event, "message", []) if _segment_value(s, "type") == "text"]
        original_text = "".join(parts).lstrip()
    keyword = original_text.startswith("神奈")
    if keyword and CHAT_KEYWORD_TRIGGER_ENABLED:
        return True
    if bool(getattr(event, "to_me", False)):
        if not keyword or CHAT_KEYWORD_TRIGGER_ENABLED:
            return True
        # Nickname preprocessing also sets to_me. Recover actual mentions
        # from the original event and actual replies from normalized reply.
        reply = getattr(event, "reply", None)
        sender = _segment_value(reply, "sender") if reply is not None else None
        if sender and str(_segment_value(sender, "user_id")) == str(getattr(event, "self_id", "")):
            return True
    self_id = _optional_event_text(getattr(event, "self_id", None))
    if self_id is None:
        return False
    # The adapter may leave a mention in the middle of the message without
    # setting to_me; inspect every segment, but never treat @all as a bot mention.
    for segment in (list(getattr(event, "message", None) or []) + list(original or [])):
        if _segment_value(segment, "type") != "at":
            continue
        data = _segment_value(segment, "data") or {}
        target = _optional_event_text(_segment_value(data, "qq"))
        if target != "all" and target == self_id:
            return True
    return False


def _bare_mention_text(event) -> str:
    # OneBot can strip the leading bot mention before our handler sees it.
    # Preserve the action as text so the orchestrator does not drop it as empty.
    if getattr(event, "message_type", None) == "group" and _is_addressed_to_bot(event):
        return "[用户仅提及了神奈，未附文字]"
    return ""


def _reply_to_message_id_from_event(event) -> str | None:
    # OneBot preprocessing extracts the reply and removes its message segment.
    reply = getattr(event, "reply", None)
    if reply is not None:
        message_id = _optional_event_text(_segment_value(reply, "message_id"))
        if message_id is not None:
            return message_id
    segments = getattr(event, "message", None)
    if segments is None:
        return None
    for segment in segments:
        if _segment_value(segment, "type") != "reply":
            continue
        data = _segment_value(segment, "data") or {}
        message_id = _segment_value(data, "id")
        if message_id is None:
            continue
        text = str(message_id).strip()
        if text:
            return text
    return None


def _image_attachments_from_event(event) -> list[ImageAttachment]:
    attachments = []
    for segment in getattr(event, "message", None) or []:
        if _segment_value(segment, "type") == "image":
            data = _segment_value(segment, "data") or {}
            attachments.append(ImageAttachment(
                position=len(attachments) + 1,
                source_url=_optional_event_text(_segment_value(data, "url")),
                file=_optional_event_text(_segment_value(data, "file")),
            ))
    return attachments


def _segment_value(segment, field: str):
    if isinstance(segment, dict):
        return segment.get(field)
    return getattr(segment, field, None)


def should_handle_message(event, text: str) -> bool:
    return should_record_message(event, text) and should_reply_to_message(event)


def should_record_message(event, text: str) -> bool:
    if not text.strip():
        return False

    user_id = str(getattr(event, "user_id", ""))
    self_id = str(getattr(event, "self_id", ""))
    if self_id and user_id == self_id:
        return False

    return True


def should_reply_to_message(event) -> bool:
    message_type = getattr(event, "message_type", None)
    if message_type == "group" and not _is_addressed_to_bot(event):
        return False

    return True


async def _send_reply_and_record(
    bot,
    event,
    reply: PendingAssistantReply,
    orchestrator: ChatOrchestrator,
) -> None:
    try:
        send_result = await bot.send(event, reply.content)
    except Exception as exc:
        record_error("send_reply", exc)
        raise
    async def on_memory_error_notice(notice_text: str) -> None:
        await bot.send(event, notice_text)

    notice = await orchestrator.record_assistant_reply(
        reply,
        on_memory_error_notice=on_memory_error_notice,
        message_id=_optional_event_text(
            _segment_value(send_result, "message_id")
        ),
    )
    if notice:
        try:
            await bot.send(event, notice)
        except Exception as exc:
            record_error("memory_error_notice_send", exc)


async def _handle_regular_chat(
    bot,
    event,
    context,
    text: str,
    orchestrator: ChatOrchestrator,
) -> None:
    async def on_search_start(notice: str) -> None:
        try:
            await bot.send(event, notice)
        except Exception as exc:
            record_error("web_search_notice_send", exc)

    attachments = _image_attachments_from_event(event)
    reply = await orchestrator.handle_message(
        context=context,
        user_message=text,
        on_search_start=on_search_start,
        **({"attachments": attachments} if attachments else {}),
    )
    if reply:
        await _send_reply_and_record(bot, event, reply, orchestrator)
