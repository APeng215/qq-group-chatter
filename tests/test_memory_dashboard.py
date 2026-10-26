import uuid
from pathlib import Path

from qq_group_chatter.memory_dashboard import (
    build_memory_dashboard_snapshot,
    build_llm_traces_snapshot,
    clear_llm_traces_api,
    llm_traces_api,
    memory_dashboard_api,
    memory_dashboard_response,
    memory_dashboard_html,
    setup_memory_dashboard,
)
from qq_group_chatter.llm_tracing import LLMTraceStore


def trace_path(name):
    path = Path("tests/.tmp/dashboard-tracing") / f"{uuid.uuid4().hex}-{name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class FakePoint:
    def __init__(self, point_id, payload):
        self.id = point_id
        self.payload = payload


class FakeVectorStore:
    def list(self, filters=None, top_k=100):
        return (
            [
                FakePoint(
                    "mem-user-1",
                    {
                        "data": "用户喜欢咖啡",
                        "user_id": "qq_user:123456",
                        "kind": "preference",
                        "scope": "user",
                        "source_created_at": 100.0,
                        "created_at": "2026-06-15T13:00:00Z",
                    },
                ),
                FakePoint(
                    "mem-conv-1",
                    {
                        "data": "这个群默认中文交流",
                        "user_id": "qq_conversation:qq_group:888888",
                        "kind": "conversation_rule",
                        "scope": "conversation",
                        "updated_at": "2026-06-15T13:01:00Z",
                    },
                ),
            ],
            None,
        )


class FakeMem0:
    def __init__(self):
        self.vector_store = FakeVectorStore()

    def history(self, memory_id):
        return [
            {
                "memory_id": memory_id,
                "event": "ADD",
                "new_memory": f"created {memory_id}",
                "created_at": "2026-06-15T13:00:00Z",
            }
        ]


class FakeLongTermMemory:
    def __init__(self):
        self._mem0 = FakeMem0()


class FakeApplication:
    def __init__(self):
        self.long_term_memory = FakeLongTermMemory()
        self.llm_trace_store = None


class FakeDriver:
    def __init__(self):
        from fastapi import FastAPI
        self.server_app = FastAPI()
        self.routes = {}

    def setup_http_server(self, setup):
        self.routes[setup.path.path] = setup


def test_setup_memory_dashboard_registers_routes_by_default(monkeypatch):
    monkeypatch.delenv("QQ_GROUP_CHATTER_MEMORY_DASHBOARD_ENABLED", raising=False)
    driver = FakeDriver()

    setup_memory_dashboard(driver, FakeApplication())

    assert driver.routes["/console"].method == "GET"
    assert driver.routes["/memory"].method == "GET"
    assert driver.routes["/api/memory"].method == "GET"
    assert driver.routes["/api/llm-traces"].method == "GET"
    assert driver.routes["/api/llm-traces/clear"].method == "POST"


def test_setup_memory_dashboard_can_be_disabled(monkeypatch):
    monkeypatch.setenv("QQ_GROUP_CHATTER_MEMORY_DASHBOARD_ENABLED", "false")
    driver = FakeDriver()

    setup_memory_dashboard(driver, FakeApplication())

    assert driver.routes == {}


def test_setup_memory_dashboard_registers_routes_when_enabled(monkeypatch):
    monkeypatch.setenv("QQ_GROUP_CHATTER_MEMORY_DASHBOARD_ENABLED", "true")
    driver = FakeDriver()

    setup_memory_dashboard(driver, FakeApplication())

    assert driver.routes["/console"].method == "GET"
    assert driver.routes["/memory"].method == "GET"
    assert driver.routes["/api/memory"].method == "GET"
    assert driver.routes["/api/llm-traces"].method == "GET"
    assert driver.routes["/api/llm-traces/clear"].method == "POST"


def test_build_memory_dashboard_snapshot_lists_memories_and_history():
    snapshot = build_memory_dashboard_snapshot(FakeApplication())

    assert snapshot["summary"] == {
        "total": 2,
        "user": 1,
        "conversation": 1,
        "other": 0,
        "queue_size": 0,
    }
    memories = {item["id"]: item for item in snapshot["memories"]}
    assert memories["mem-user-1"]["content"] == "用户喜欢咖啡"
    assert memories["mem-user-1"]["history"][0]["event"] == "ADD"
    assert memories["mem-conv-1"]["owner_id"] == "qq_conversation:qq_group:888888"


async def test_llm_trace_dashboard_api_returns_snapshot_and_clear_response():
    application = FakeApplication()
    application.llm_trace_store = LLMTraceStore(path=trace_path("traces.jsonl"), max_records=10)
    application.llm_trace_store.record_start(
        component="chat_agent",
        operation="decision",
        model="deepseek-v4-pro",
        thinking="disabled",
        temperature=0.7,
        response_format=None,
        messages=[{"role": "user", "content": "hello"}],
    )

    snapshot = build_llm_traces_snapshot(application)
    response = await llm_traces_api(application)
    clear_response = await clear_llm_traces_api(application)

    assert snapshot["summary"]["total"] == 1
    assert "hello" in response.content
    assert clear_response.status_code == 200
    assert build_llm_traces_snapshot(application)["traces"] == []


async def test_spa_entry_does_not_query_memory_and_assets_are_local(monkeypatch):
    import re
    import qq_group_chatter.memory_dashboard as dashboard

    monkeypatch.setattr(dashboard, "build_memory_dashboard_snapshot",
                        lambda app: (_ for _ in ()).throw(AssertionError("entry queried Mem0")))
    response = await memory_dashboard_response(FakeApplication())
    assert response.status_code == 200
    assert '<div id="root"></div>' in response.content
    sources = re.findall(r'(?:src|href)="([^"]+)"', response.content)
    assert sources and all(source.startswith("/console-assets/") for source in sources)
    for source in sources:
        assert (dashboard.CONSOLE_DIR / source.removeprefix("/console-assets/")).is_file()


async def test_spa_missing_build_has_explicit_error(monkeypatch):
    import qq_group_chatter.memory_dashboard as dashboard

    monkeypatch.setattr(dashboard, "CONSOLE_DIR", trace_path("missing-build"))
    response = await memory_dashboard_response(FakeApplication())
    assert response.status_code == 503
    assert "build_console.py" in response.content


async def test_static_assets_and_path_traversal(monkeypatch):
    import httpx
    import qq_group_chatter.memory_dashboard as dashboard

    root = trace_path("static")
    root.mkdir()
    (root / "asset.js").write_text("window.localAsset = true", encoding="utf-8")
    (root.parent / "private.txt").write_text("private", encoding="utf-8")
    monkeypatch.setattr(dashboard, "CONSOLE_DIR", root)
    monkeypatch.setenv("QQ_GROUP_CHATTER_MEMORY_DASHBOARD_ENABLED", "true")
    driver = FakeDriver()
    setup_memory_dashboard(driver, FakeApplication())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=driver.server_app), base_url="http://test") as client:
        response = await client.get("/console-assets/asset.js")
        assert response.status_code == 200 and "localAsset" in response.text
        for path in ["/console-assets/%2e%2e/private.txt", "/console-assets/not-found.js", "/console-assets/%2e%2e%5cprivate.txt"]:
            assert (await client.get(path)).status_code == 404
