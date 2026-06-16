from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import traceback
from contextlib import contextmanager
from typing import Any, Iterator

from prometheus_client import Counter, Gauge, Histogram


logger = logging.getLogger("qq_group_chatter")


SENSITIVE_LOG_PATTERNS = [
    re.compile(r"\b\d{11}\b"),
    re.compile(r"(?i)\b(user_id|group_id|self_id)\s*[:=]\s*\d{5,12}\b"),
    re.compile(r"(?i)\bconversation_id\s*[:=]\s*qq_(?:group|private):\d{5,12}\b"),
    re.compile(r"\bqq_(?:user|group|private):\d{5,12}\b"),
    re.compile(
        r"(?i)\b(password|passwd|token|api[\s_-]?key|secret|bearer)\b\s*[:=：是为]\s*['\"]?[^'\"\s,;]+"
    ),
    re.compile(r"(密码|口令|令牌|密钥|秘钥|接口密钥)\s*[:=：是为]\s*['\"]?[^'\"\s,;]+"),
    re.compile(r"(?i)\b(?:sk|ak)-[a-z0-9][a-z0-9_-]{6,}\b"),
]


MESSAGES_TOTAL = Counter(
    "qq_bot_messages_total",
    "Messages handled by the bot.",
    ["conversation_type", "result"],
)
RESPONSE_LATENCY_SECONDS = Histogram(
    "qq_bot_response_latency_seconds",
    "End-to-end response latency.",
)
LLM_LATENCY_SECONDS = Histogram(
    "qq_bot_llm_latency_seconds",
    "LLM call latency.",
    ["component"],
)
MEM0_SEARCH_LATENCY_SECONDS = Histogram(
    "qq_bot_mem0_search_latency_seconds",
    "Mem0 search latency.",
    ["scope"],
)
MEM0_ADD_LATENCY_SECONDS = Histogram(
    "qq_bot_mem0_add_latency_seconds",
    "Mem0 add latency.",
    ["scope"],
)
MEM0_ADD_TOTAL = Counter(
    "qq_bot_mem0_add_total",
    "Mem0 add attempts.",
    ["scope", "result"],
)
MEMORY_INGESTION_QUEUE_SIZE = Gauge(
    "qq_bot_memory_ingestion_queue_size",
    "Pending long-term memory ingestion jobs.",
)
MEMORY_CANDIDATES_TOTAL = Counter(
    "qq_bot_memory_candidates_total",
    "Long-term memory candidates.",
    ["scope", "kind", "result"],
)
MEMORY_DUPLICATE_SKIPS_TOTAL = Counter(
    "qq_bot_memory_duplicate_skips_total",
    "Duplicate long-term memory candidates skipped.",
    ["scope"],
)
ERRORS_TOTAL = Counter(
    "qq_bot_errors_total",
    "Errors by stage and type.",
    ["stage", "error_type"],
)


def hash_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def conversation_log_fields(context: Any) -> dict[str, Any]:
    return {
        "conversation_id_hash": hash_identifier(getattr(context, "conversation_id", None)),
        "conversation_type": getattr(context, "conversation_type", None),
        "user_id_hash": hash_identifier(getattr(context, "user_id", None)),
        "group_id_hash": hash_identifier(getattr(context, "group_id", None)),
        "message_id": getattr(context, "message_id", None),
    }


def log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))


def record_error(stage: str, error: BaseException) -> None:
    ERRORS_TOTAL.labels(stage=stage, error_type=type(error).__name__).inc()
    logger.error(
        "stage=%s error_type=%s message=%s\n%s",
        stage,
        type(error).__name__,
        sanitize_log_text(str(error)),
        _sanitized_traceback_text(error),
    )


def sanitize_log_text(value: str) -> str:
    sanitized = value
    for pattern in SENSITIVE_LOG_PATTERNS:
        sanitized = pattern.sub(_redact_log_match, sanitized)
    return sanitized


def _redact_log_match(match: re.Match[str]) -> str:
    text = match.group(0)
    if ":" in text and text.startswith("qq_"):
        prefix = text.split(":", 1)[0]
        return f"{prefix}:[REDACTED]"
    if "=" in text:
        return f"{text.split('=', 1)[0]}=[REDACTED]"
    if ":" in text:
        return f"{text.split(':', 1)[0]}:[REDACTED]"
    return "[REDACTED]"


def _sanitized_traceback_text(error: BaseException) -> str:
    return sanitize_log_text(
        "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ).rstrip()
    )


@contextmanager
def observe_duration(
    *,
    metric: Histogram,
    log_name: str,
    labels: dict[str, str] | None = None,
    log_fields: dict[str, Any] | None = None,
) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        if labels:
            metric.labels(**labels).observe(duration)
        else:
            metric.observe(duration)
        log_event(
            log_name,
            duration_ms=round(duration * 1000, 3),
            **(log_fields or {}),
        )
