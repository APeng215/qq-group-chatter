from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qq_group_chatter.observability import sanitize_log_text


@dataclass
class LLMTraceStore:
    path: Path
    max_records: int = 500
    enabled: bool = True

    def __init__(self, path: str | Path, max_records: int = 500, enabled: bool = True):
        self.path = Path(path)
        self.max_records = max(1, int(max_records))
        self.enabled = enabled
        self._lock = threading.Lock()

    @classmethod
    def disabled(cls) -> "LLMTraceStore":
        return cls(path="logs/llm-traces.jsonl", enabled=False)

    @classmethod
    def enabled_store(
        cls,
        *,
        path: str | Path = "logs/llm-traces.jsonl",
        max_records: int = 500,
    ) -> "LLMTraceStore":
        return cls(path=path, max_records=max_records, enabled=True)

    def record_start(
        self,
        *,
        component: str,
        operation: str,
        model: str,
        thinking: str,
        temperature: float,
        response_format: dict[str, Any] | None,
        messages: list[dict[str, Any]],
    ) -> str:
        trace_id = uuid.uuid4().hex
        self._append_event(
            {
                "event": "start",
                "trace_id": trace_id,
                "created_at": _now_iso(),
                "component": component,
                "operation": operation,
                "model": model,
                "thinking": thinking,
                "temperature": temperature,
                "response_format": response_format,
                "messages": messages,
                "status": "running",
            }
        )
        return trace_id

    def record_success(
        self,
        *,
        trace_id: str,
        response_text: str,
        usage: dict[str, Any] | None,
        duration_ms: float,
    ) -> None:
        self._append_event(
            {
                "event": "success",
                "trace_id": trace_id,
                "updated_at": _now_iso(),
                "response_text": response_text,
                "usage": usage,
                "duration_ms": round(float(duration_ms), 3),
                "status": "success",
            }
        )

    def record_error(
        self,
        *,
        trace_id: str,
        error: BaseException,
        duration_ms: float,
    ) -> None:
        self._append_event(
            {
                "event": "error",
                "trace_id": trace_id,
                "updated_at": _now_iso(),
                "duration_ms": round(float(duration_ms), 3),
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": sanitize_log_text(str(error)),
            }
        )

    def snapshot(self) -> dict[str, Any]:
        if not self.enabled:
            traces: list[dict[str, Any]] = []
            errors: list[str] = ["LLM trace store is disabled."]
        else:
            traces, errors = self._read_traces()
        return {
            "generated_at": _now_iso(),
            "summary": _summary(traces),
            "traces": traces,
            "errors": errors,
        }

    def clear(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def _append_event(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            self._compact_locked()

    def _compact_locked(self) -> None:
        traces, _ = self._read_traces_unlocked()
        traces = traces[: self.max_records]
        compact_events = [_trace_to_event(trace) for trace in reversed(traces)]
        self.path.write_text(
            "".join(
                json.dumps(event, ensure_ascii=False, default=str) + "\n"
                for event in compact_events
            ),
            encoding="utf-8",
        )

    def _read_traces(self) -> tuple[list[dict[str, Any]], list[str]]:
        with self._lock:
            return self._read_traces_unlocked()

    def _read_traces_unlocked(self) -> tuple[list[dict[str, Any]], list[str]]:
        if not self.path.exists():
            return [], []
        errors: list[str] = []
        merged: dict[str, dict[str, Any]] = {}
        order: dict[str, int] = {}
        for line_number, raw_line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            trace_id = str(event.get("trace_id") or "")
            if not trace_id:
                errors.append(f"Missing trace_id at line {line_number}.")
                continue
            if trace_id not in merged:
                merged[trace_id] = {"trace_id": trace_id}
                order[trace_id] = line_number
            merged[trace_id].update({k: v for k, v in event.items() if k != "event"})
            merged[trace_id].setdefault("created_at", event.get("updated_at") or _now_iso())
            merged[trace_id].setdefault("status", "running")
        traces = sorted(
            merged.values(),
            key=lambda item: (
                str(item.get("created_at") or item.get("updated_at") or ""),
                order.get(str(item.get("trace_id")), 0),
            ),
            reverse=True,
        )
        return traces[: self.max_records], errors


def _trace_to_event(trace: dict[str, Any]) -> dict[str, Any]:
    event = dict(trace)
    event["event"] = str(trace.get("status") or "snapshot")
    return event


def _summary(traces: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [
        float(trace["duration_ms"])
        for trace in traces
        if isinstance(trace.get("duration_ms"), int | float)
    ]
    average_duration_ms = round(sum(durations) / len(durations), 3) if durations else 0
    return {
        "total": len(traces),
        "running": sum(1 for trace in traces if trace.get("status") == "running"),
        "success": sum(1 for trace in traces if trace.get("status") == "success"),
        "error": sum(1 for trace in traces if trace.get("status") == "error"),
        "average_duration_ms": average_duration_ms,
    }


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
