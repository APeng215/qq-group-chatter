import subprocess
import sys

from qq_group_chatter.app import create_default_orchestrator
from qq_group_chatter.models import build_private_conversation_context


async def test_default_orchestrator_can_reply_without_external_clients():
    orchestrator = create_default_orchestrator()
    context = build_private_conversation_context(
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )

    reply = await orchestrator.handle_message(context=context, user_message="你好")

    assert reply == "我现在还没有配置聊天模型。"


def test_app_import_does_not_import_nonebot_plugin():
    code = (
        "import sys;"
        "import qq_group_chatter.app;"
        "print('qq_group_chatter.plugins.chat' in sys.modules)"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
