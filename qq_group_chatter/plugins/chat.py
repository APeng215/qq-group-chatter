from __future__ import annotations

from time import time

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
        text = event.get_plaintext()
        context = _context_from_event(event)
        if context is None:
            return
        reply = await orchestrator.handle_message(context=context, user_message=text)
        if reply:
            await bot.send(event, reply)


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
