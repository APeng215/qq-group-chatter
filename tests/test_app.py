import subprocess
import sys
import uuid
from pathlib import Path

from qq_group_chatter.app import (
    ChatBotApplication,
    NoopMem0Client,
    create_default_application,
    create_default_orchestrator,
)
from qq_group_chatter.models import build_private_conversation_context


def local_tmp_path(name):
    path = Path("tests/.tmp/app") / f"{name}-{uuid.uuid4().hex}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


async def test_default_orchestrator_can_reply_without_external_clients(monkeypatch):
    monkeypatch.setattr("qq_group_chatter.app.create_deepseek_chat_llm", lambda **kwargs: None)
    orchestrator = create_default_orchestrator(mem0_client=NoopMem0Client())
    context = build_private_conversation_context(
        user_id=123456,
        message_id="m1",
        nickname="阿咳",
        timestamp=123.0,
    )

    pending_reply = await orchestrator.handle_message(context=context, user_message="你好")

    assert pending_reply.content == "我现在还没有配置聊天模型。"


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
    monkeypatch.delenv("DEEPSEEK_THINKING", raising=False)

    orchestrator = create_default_orchestrator(mem0_client=NoopMem0Client())

    assert orchestrator._chat_agent._llm.model == "deepseek-v4-pro"
    assert orchestrator._chat_agent._llm.thinking == "enabled"


def test_default_orchestrator_can_disable_deepseek_thinking(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_THINKING", "false")

    orchestrator = create_default_orchestrator(mem0_client=NoopMem0Client())

    assert orchestrator._chat_agent._llm.thinking == "disabled"


def test_default_application_wires_web_search_when_tavily_key_exists(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-secret")
    monkeypatch.setattr("qq_group_chatter.app.create_default_mem0_client", lambda: NoopMem0Client())

    application = create_default_application()

    assert application.web_search is not None
    assert application.orchestrator._web_search is application.web_search


def test_default_application_wires_shared_llm_trace_store(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("QQ_GROUP_CHATTER_LLM_TRACE_PATH", "tests/.tmp/app-llm-traces.jsonl")
    monkeypatch.setenv("QQ_GROUP_CHATTER_LLM_TRACE_MAX_RECORDS", "7")
    monkeypatch.setattr("qq_group_chatter.app.create_default_mem0_client", lambda: NoopMem0Client())

    application = create_default_application()

    chat_llm = application.orchestrator._chat_agent._llm
    planner_llm = application.long_term_memory._planner._llm
    assert application.llm_trace_store is not None
    assert chat_llm.trace_store is application.llm_trace_store
    assert planner_llm.trace_store is application.llm_trace_store
    assert application.llm_trace_store.max_records == 7


def test_default_application_wires_persistent_short_term_memory(monkeypatch):
    path = local_tmp_path("short-term-memory")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("SHORT_TERM_MEMORY_PATH", str(path))
    monkeypatch.setattr("qq_group_chatter.app.create_default_mem0_client", lambda: NoopMem0Client())

    application = create_default_application()
    short_term = application.orchestrator._short_term_memory

    assert short_term._max_messages_per_conversation == 300
    assert short_term._path == path


def test_default_orchestrator_does_not_create_web_search(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setattr(
        "qq_group_chatter.app.create_default_web_search_service",
        lambda: (_ for _ in ()).throw(AssertionError("should not create web search")),
    )

    orchestrator = create_default_orchestrator(mem0_client=NoopMem0Client())

    assert orchestrator._web_search is None


def test_default_orchestrator_uses_in_memory_short_term_memory(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")

    orchestrator = create_default_orchestrator(mem0_client=NoopMem0Client())

    assert orchestrator._short_term_memory._path is None


class FakeLongTermMemory:
    def __init__(self):
        self.started = 0
        self.stopped = 0

    async def start(self):
        self.started += 1

    async def stop(self):
        self.stopped += 1


class FakeOrchestrator:
    def __init__(self):
        self.long_term_memory = FakeLongTermMemory()


async def test_chat_bot_application_starts_and_stops_long_term_memory():
    orchestrator = FakeOrchestrator()
    app = ChatBotApplication(
        orchestrator=orchestrator,
        long_term_memory=orchestrator.long_term_memory,
    )

    await app.start()
    await app.stop()

    assert orchestrator.long_term_memory.started == 1
    assert orchestrator.long_term_memory.stopped == 1
