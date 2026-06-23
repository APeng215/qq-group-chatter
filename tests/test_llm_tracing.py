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
        current_user_message="[QQ:123456 昵称:tester] hello",
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
    assert trace["current_user_message"] == "[QQ:123456 昵称:tester] hello"
    assert trace["response_format"] == {"type": "json_object"}
    assert trace["messages"] == [{"role": "user", "content": "hello"}]
    assert trace["response_text"] == '{"action":"reply","content":"hi"}'
    assert trace["reasoning_content"] == "判断应直接回复。"
    assert trace["usage"] == {"total_tokens": 12}


def test_trace_store_average_duration_only_uses_recent_chat_agent_calls():
    tmp_path = trace_dir("chat-average")
    store = LLMTraceStore(path=tmp_path / "traces.jsonl", max_records=10)
    chat_first = store.record_start(
        component="chat_agent",
        operation="decision",
        model="deepseek-v4-pro",
        thinking="enabled",
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[],
    )
    planner = store.record_start(
        component="memory_planner",
        operation="plan_memory",
        model="deepseek-v4-pro",
        thinking="enabled",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[],
    )
    chat_second = store.record_start(
        component="chat_agent",
        operation="grounded_search_reply",
        model="deepseek-v4-pro",
        thinking="enabled",
        temperature=0.7,
        response_format=None,
        messages=[],
    )

    store.record_success(
        trace_id=chat_first,
        response_text="直接回复",
        reasoning_content=None,
        usage=None,
        duration_ms=100.0,
    )
    store.record_success(
        trace_id=planner,
        response_text='{"operations":[]}',
        reasoning_content=None,
        usage=None,
        duration_ms=1000.0,
    )
    store.record_success(
        trace_id=chat_second,
        response_text="联网搜索回复",
        reasoning_content=None,
        usage=None,
        duration_ms=300.0,
    )

    snapshot = store.snapshot()

    assert snapshot["summary"]["total"] == 3
    assert snapshot["summary"]["success"] == 3
    assert snapshot["summary"]["average_duration_ms"] == 200.0


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


def test_trace_store_records_final_reply_metadata():
    tmp_path = trace_dir("final-reply")
    store = LLMTraceStore(path=tmp_path / "traces.jsonl", max_records=10)
    trace_id = store.record_start(
        component="chat_agent",
        operation="decision",
        model="deepseek-v4-pro",
        thinking="disabled",
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[],
    )
    store.record_success(
        trace_id=trace_id,
        response_text="not json",
        reasoning_content=None,
        usage=None,
        duration_ms=2.0,
    )

    store.record_result(
        trace_id=trace_id,
        parsed_action="fallback",
        final_reply="我刚刚没能整理好回复，稍后再试。",
        fallback_reason="invalid_chat_decision",
    )

    trace = store.snapshot()["traces"][0]
    assert trace["response_text"] == "not json"
    assert trace["parsed_action"] == "fallback"
    assert trace["final_reply"] == "我刚刚没能整理好回复，稍后再试。"
    assert trace["fallback_reason"] == "invalid_chat_decision"
