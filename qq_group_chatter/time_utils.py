from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo


LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")
TIME_FORMAT = "%Y-%m-%d %H:%M"


def current_time_text() -> str:
    return datetime.now(LOCAL_TIMEZONE).strftime(TIME_FORMAT)


def format_time_text(value: object) -> str | None:
    parsed = _parse_time(value)
    if parsed is None:
        return None
    return parsed.astimezone(LOCAL_TIMEZONE).strftime(TIME_FORMAT)


def _parse_time(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(float(value), tz=UTC)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = datetime.fromtimestamp(float(text), tz=UTC)
            except ValueError:
                return None
    else:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TIMEZONE)
    return parsed
