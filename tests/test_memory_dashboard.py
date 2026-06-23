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


def test_memory_dashboard_html_contains_bootstrap_snapshot():
    html = memory_dashboard_html({"summary": {"total": 0}, "memories": [], "errors": []})

    assert "<!doctype html>" in html
    assert "window.__MEMORY_SNAPSHOT__" in html
    assert "<title>运行控制台</title>" in html
    assert "<h1>运行控制台</h1>" in html
    assert "记忆库" in html


async def test_memory_dashboard_handlers_return_html_and_json_responses():
    html_response = await memory_dashboard_response(FakeApplication())
    api_response = await memory_dashboard_api(FakeApplication())

    assert html_response.status_code == 200
    assert "text/html" in html_response.headers["content-type"]
    assert "运行控制台" in html_response.content
    assert "长期记忆" in html_response.content
    assert api_response.status_code == 200
    assert "application/json" in api_response.headers["content-type"]
    assert "用户喜欢咖啡" in api_response.content


def test_memory_dashboard_html_contains_llm_console_assets():
    html = memory_dashboard_html({"summary": {"total": 0}, "memories": [], "errors": []})

    assert "window.__MEMORY_SNAPSHOT__" in html
    assert "LLM" in html
    assert "/api/llm-traces" in html


def test_memory_dashboard_html_preserves_trace_detail_expansion_on_refresh():
    html = memory_dashboard_html({"summary": {"total": 0}, "memories": [], "errors": []})

    assert "captureTraceDetailState" in html
    assert "restoreTraceDetailState" in html
    assert "data-detail-key" in html


def test_memory_dashboard_html_renders_trace_text_newlines_readably():
    html = memory_dashboard_html({"summary": {"total": 0}, "memories": [], "errors": []})

    assert "formatTraceText" in html
    assert 'replace(/\\\\n/g, "\\n")' in html
    assert '<pre>${formatTraceText(item.response_text || "")}</pre>' in html


def test_memory_dashboard_html_renders_trace_messages_by_role_and_content():
    html = memory_dashboard_html({"summary": {"total": 0}, "memories": [], "errors": []})

    assert "function renderTraceMessages(messages)" in html
    assert "trace-message-role" in html
    assert "trace-message-content" in html
    assert "JSON.stringify(item.messages || [], null, 2)" not in html
    assert "<pre>${formatTraceText(messages)}</pre>" not in html
    assert "${renderTraceMessages(item.messages || [])}" in html


def test_memory_dashboard_html_renders_trace_reasoning_content():
    html = memory_dashboard_html({"summary": {"total": 0}, "memories": [], "errors": []})

    assert "function traceHasReasoningContent(item)" in html
    assert "function renderTraceReasoning(item)" in html
    assert "trace-reasoning-content" in html
    assert "thinking 内容" in html
    assert "renderTraceSummary(traces)" in html
    assert "traces.filter(traceHasReasoningContent).length" in html
    assert "${renderTraceReasoning(item)}" in html


def test_memory_dashboard_html_renders_trace_current_user_message():
    html = memory_dashboard_html({"summary": {"total": 0}, "memories": [], "errors": []})

    assert "用户发言" in html
    assert "item.current_user_message" in html


def test_memory_dashboard_html_labels_average_duration_as_chat_only():
    html = memory_dashboard_html({"summary": {"total": 0}, "memories": [], "errors": []})

    assert "聊天平均耗时" in html
    assert '<span class="muted">平均耗时 ms</span>' not in html


def test_memory_dashboard_html_indents_llm_trace_text_blocks():
    html = memory_dashboard_html({"summary": {"total": 0}, "memories": [], "errors": []})

    assert ".trace-message-body" in html
    assert "border-left: 3px solid #bfdbfe" in html
    assert "padding-left: 12px" in html
    assert 'class="trace-json-block"' in html


def test_memory_dashboard_html_does_not_poll_llm_traces():
    html = memory_dashboard_html({"summary": {"total": 0}, "memories": [], "errors": []})

    assert "setInterval" not in html
    assert "traceRefreshEl.addEventListener" in html


def test_memory_dashboard_html_adds_trace_filter_status_and_quick_filters():
    html = memory_dashboard_html({"summary": {"total": 0}, "memories": [], "errors": []})

    assert 'id="trace-filter-status"' in html
    assert 'id="trace-component-chips"' in html
    assert "function renderTraceFilterStatus" in html
    assert "function renderTraceComponentChips" in html
    assert "data-component-filter" in html
    assert "traceFilterStatusEl" in html


def test_memory_dashboard_html_marks_trace_status_and_duration():
    html = memory_dashboard_html({"summary": {"total": 0}, "memories": [], "errors": []})

    assert "function traceStatusClass" in html
    assert "function durationSeverityClass" in html
    assert "function formatDurationMs" in html
    assert 'class="badge ${traceStatusClass(item.status)}"' in html
    assert 'class="duration-pill ${durationSeverityClass(item.duration_ms)}"' in html


def test_memory_dashboard_html_adds_trace_copy_actions():
    html = memory_dashboard_html({"summary": {"total": 0}, "memories": [], "errors": []})

    assert "function copyTraceText" in html
    assert "function traceCopyPayload" in html
    assert "navigator.clipboard.writeText" in html
    assert "data-copy-trace-id" in html
    assert "复制 ID" in html
    assert "复制响应" in html
    assert "traceListEl.addEventListener" in html


def test_memory_dashboard_html_renders_final_trace_result():
    html = memory_dashboard_html({"summary": {"total": 0}, "memories": [], "errors": []})

    assert "function renderTraceResult" in html
    assert "最终输出 / 解析结果" in html
    assert "final_reply" in html
    assert "fallback_reason" in html


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
