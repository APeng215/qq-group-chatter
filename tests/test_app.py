import subprocess
import sys

from qq_group_chatter.app import create_default_orchestrator
from qq_group_chatter.models import build_private_conversation_context


async def test_default_orchestrator_can_reply_without_external_clients():
    # Run away from the project root .env so this unit test never makes network calls.
    import os
    from pathlib import Path

    old_cwd = Path.cwd()
    test_cwd = Path("tests/.tmp/no_env_app")
    test_cwd.mkdir(parents=True, exist_ok=True)
    os.chdir(test_cwd)
    orchestrator = create_default_orchestrator()
    try:
        context = build_private_conversation_context(
            user_id=123456,
            message_id="m1",
            nickname="阿咳",
            timestamp=123.0,
        )

        reply = await orchestrator.handle_message(context=context, user_message="你好")

        assert reply == "我现在还没有配置聊天模型。"
    finally:
        os.chdir(old_cwd)


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


def test_default_orchestrator_uses_deepseek_when_key_exists(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")

    orchestrator = create_default_orchestrator()

    assert orchestrator._chat_agent._llm.model == "deepseek-v4-pro"
    assert orchestrator._chat_agent._llm.thinking == "disabled"
