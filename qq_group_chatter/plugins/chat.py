from __future__ import annotations

from time import time
from typing import Any

from qq_group_chatter.models import PendingAssistantReply
from qq_group_chatter.observability import record_error
from qq_group_chatter.orchestrator import ChatOrchestrator
from qq_group_chatter.services.web_search import parse_search_command


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
search_service: Any | None = None


def setup_orchestrator(instance: ChatOrchestrator) -> None:
    global orchestrator
    orchestrator = instance


def setup_search_service(instance: Any | None) -> None:
    global search_service
    search_service = instance


if on_message is not None:
    matcher = on_message(priority=50, block=False)

    @matcher.handle()
    async def handle_nonebot_message(bot: Bot, event: MessageEvent) -> None:
        if orchestrator is None:
            return
        text = event.get_plaintext()
        if not should_handle_message(event, text):
            return
        context = _context_from_event(event)
        if context is None:
            return
        if await _send_search_reply(
            bot,
            event,
            context,
            text,
            search_service=search_service,
            orchestrator=orchestrator,
        ):
            return
        reply = await orchestrator.handle_message(context=context, user_message=text)
        if reply:
            await _send_reply_and_record(bot, event, reply, orchestrator)


def _context_from_event(event) -> ConversationContext | None:
    from qq_group_chatter.models import (
        build_group_conversation_context,
        build_private_conversation_context,
    )

    timestamp = float(getattr(event, "time", time()))
    message_id = str(getattr(event, "message_id", ""))
    user_id = str(getattr(event, "user_id", ""))
    sender = getattr(event, "sender", None)
    nickname = getattr(sender, "nickname", None) if sender is not None else None

    if GroupMessageEvent is not object and isinstance(event, GroupMessageEvent):
        return build_group_conversation_context(
            group_id=getattr(event, "group_id"),
            user_id=user_id,
            message_id=message_id,
            nickname=nickname,
            timestamp=timestamp,
        )
    if PrivateMessageEvent is not object and isinstance(event, PrivateMessageEvent):
        return build_private_conversation_context(
            user_id=user_id,
            message_id=message_id,
            nickname=nickname,
            timestamp=timestamp,
        )
    return None


def should_handle_message(event, text: str) -> bool:
    if not text.strip():
        return False

    user_id = str(getattr(event, "user_id", ""))
    self_id = str(getattr(event, "self_id", ""))
    if self_id and user_id == self_id:
        return False

    message_type = getattr(event, "message_type", None)
    if message_type == "group" and not bool(getattr(event, "to_me", False)):
        return False

    return True


async def _send_reply_and_record(
    bot,
    event,
    reply: PendingAssistantReply,
    orchestrator: ChatOrchestrator,
) -> None:
    try:
        await bot.send(event, reply.content)
    except Exception as exc:
        record_error("send_reply", exc)
        raise
    await orchestrator.record_assistant_reply(reply)


async def _send_search_reply(
    bot,
    event,
    context,
    text: str,
    *,
    search_service: Any | None,
    orchestrator: ChatOrchestrator,
) -> bool:
    query = parse_search_command(text)
    if query is None:
        return False
    if search_service is None:
        reply_content = "搜索功能没有配置。请检查 WEB_SEARCH_ENABLED 和 TAVILY_API_KEY。"
    else:
        try:
            reply_content = await search_service.search_reply(query)
        except Exception as exc:
            record_error("web_search", exc)
            reply_content = "搜索失败，稍后再试。"

    reply = PendingAssistantReply(
        context=context,
        content=reply_content,
        timestamp=time(),
    )
    try:
        await bot.send(event, reply.content)
    except Exception as exc:
        record_error("web_search_send", exc)
        return True
    await orchestrator.record_assistant_reply(reply)
    return True
