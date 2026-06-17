import json
import uuid
from pathlib import Path

from qq_group_chatter.llm_tracing import LLMTraceStore


def trace_dir(name):
    path = Path("tests/.tmp/llm-tracing") / f"{name}-{uuid.uuid4().hex}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_trace_store_groups_events_and_summarizes_statuses():
    tmp_path = trace_dir("groups")
    store = LLMTraceStore(path=tmp_path / "traces.jsonl", max_records=10)
    trace_id = store.record_start(
        component="chat_agent",
        operation="decision",
        model="deepseek-v4-pro",
        thinking="disabled",
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": "hello"}],
    )

    store.record_success(
        trace_id=trace_id,
        response_text='{"action":"reply","content":"hi"}',
        reasoning_content="判断应直接回复。",
        usage={"total_tokens": 12},
        duration_ms=15.2,
    )

    snapshot = store.snapshot()

    assert snapshot["summary"]["total"] == 1
    assert snapshot["summary"]["success"] == 1
    assert snapshot["summary"]["running"] == 0
    assert snapshot["summary"]["error"] == 0
    assert snapshot["summary"]["average_duration_ms"] == 15.2
    trace = snapshot["traces"][0]
    assert trace["trace_id"] == trace_id
    assert trace["component"] == "chat_agent"
    assert trace["operation"] == "decision"
    assert trace["model"] == "deepseek-v4-pro"
    assert trace["thinking"] == "disabled"
    assert trace["response_format"] == {"type": "json_object"}
    assert trace["messages"] == [{"role": "user", "content": "hello"}]
    assert trace["response_text"] == '{"action":"reply","content":"hi"}'
    assert trace["reasoning_content"] == "判断应直接回复。"
    assert trace["usage"] == {"total_tokens": 12}


def test_trace_store_keeps_latest_records_and_ignores_malformed_lines():
    tmp_path = trace_dir("latest")
    path = tmp_path / "traces.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    store = LLMTraceStore(path=path, max_records=2)

    first = store.record_start(
        component="chat_agent",
        operation="first",
        model="deepseek-v4-pro",
        thinking="disabled",
        temperature=0.7,
        response_format=None,
        messages=[],
    )
    store.record_success(
        trace_id=first,
        response_text="first",
        reasoning_content=None,
        usage=None,
        duration_ms=1.0,
    )
    second = store.record_start(
        component="chat_agent",
        operation="second",
        model="deepseek-v4-pro",
        thinking="disabled",
        temperature=0.7,
        response_format=None,
        messages=[],
    )
    third = store.record_start(
        component="memory_planner",
        operation="third",
        model="deepseek-v4-pro",
        thinking="disabled",
        temperature=0.0,
        response_format=None,
        messages=[],
    )

    snapshot = store.snapshot()

    assert [trace["trace_id"] for trace in snapshot["traces"]] == [third, second]
    assert all(trace["trace_id"] != first for trace in snapshot["traces"])
    assert snapshot["errors"] == []
    assert path.read_text(encoding="utf-8").count("\n") == 2


def test_trace_store_clear_truncates_file():
    tmp_path = trace_dir("clear")
    store = LLMTraceStore(path=tmp_path / "traces.jsonl", max_records=10)
    store.record_start(
        component="chat_agent",
        operation="decision",
        model="deepseek-v4-pro",
        thinking="disabled",
        temperature=0.7,
        response_format=None,
        messages=[],
    )

    store.clear()

    assert store.snapshot()["traces"] == []
    assert (tmp_path / "traces.jsonl").read_text(encoding="utf-8") == ""


def test_trace_store_sanitizes_error_messages():
    tmp_path = trace_dir("sanitize")
    store = LLMTraceStore(path=tmp_path / "traces.jsonl", max_records=10)
    trace_id = store.record_start(
        component="chat_agent",
        operation="decision",
        model="deepseek-v4-pro",
        thinking="disabled",
        temperature=0.7,
        response_format=None,
        messages=[],
    )

    store.record_error(
        trace_id=trace_id,
        error=RuntimeError("api_key=sk-secret123456 failed"),
        duration_ms=2.0,
    )

    trace = store.snapshot()["traces"][0]
    assert trace["status"] == "error"
    assert trace["error_type"] == "RuntimeError"
    assert "[REDACTED]" in trace["error_message"]
    assert "sk-secret" not in json.dumps(trace, ensure_ascii=False)
