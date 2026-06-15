from __future__ import annotations

import logging
import os
import sys
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_LOGGER_NAME = "qq_group_chatter"
DEFAULT_LOG_FILE_NAME = "qq-group-chatter.log"


@dataclass(frozen=True)
class FileLogConfig:
    enabled: bool
    level: int
    path: Path
    rotation: str
    retention: int


def configure_runtime_logging() -> None:
    project_level = _read_level("QQ_GROUP_CHATTER_LOG_LEVEL", logging.INFO)
    framework_level = _read_level("QQ_GROUP_CHATTER_FRAMEWORK_LOG_LEVEL", logging.INFO)
    configure_project_logging(project_level)
    configure_loguru_logging(
        project_min_level=project_level,
        framework_min_level=framework_level,
        file_log_config=file_log_config_from_env(),
    )


def configure_project_logging(
    level: int = logging.INFO,
    *,
    handler: logging.Handler | None = None,
) -> None:
    logger = logging.getLogger(PROJECT_LOGGER_NAME)
    resolved_handler = handler if handler is not None else _create_loguru_handler()
    resolved_handler.setLevel(level)
    logger.handlers = [resolved_handler]
    logger.setLevel(level)
    logger.propagate = False


def configure_loguru_logging(
    *,
    project_min_level: int = logging.INFO,
    framework_min_level: int = logging.WARNING,
    file_log_config: FileLogConfig | None = None,
) -> None:
    try:
        from nonebot.log import default_format, logger
    except Exception:
        return

    logger.remove()
    logger.add(
        sys.stdout,
        level=0,
        diagnose=False,
        filter=lambda record: should_emit_loguru_record(
            record,
            project_min_level=project_min_level,
            framework_min_level=framework_min_level,
        ),
        format=default_format,
    )
    resolved_file_config = file_log_config or file_log_config_from_env()
    if resolved_file_config.enabled:
        resolved_file_config.path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            resolved_file_config.path,
            level=0,
            diagnose=False,
            filter=lambda record: should_emit_loguru_record(
                record,
                project_min_level=resolved_file_config.level,
                framework_min_level=resolved_file_config.level,
            ),
            format=default_format,
            rotation=resolved_file_config.rotation,
            retention=resolved_file_config.retention,
            encoding="utf-8",
        )


def file_log_config_from_env() -> FileLogConfig:
    log_dir = Path(os.getenv("QQ_GROUP_CHATTER_LOG_DIR", "logs"))
    return FileLogConfig(
        enabled=_read_bool("QQ_GROUP_CHATTER_FILE_LOG_ENABLED", True),
        level=_read_level("QQ_GROUP_CHATTER_FILE_LOG_LEVEL", logging.DEBUG),
        path=log_dir / DEFAULT_LOG_FILE_NAME,
        rotation=os.getenv("QQ_GROUP_CHATTER_FILE_LOG_ROTATION", "10 MB"),
        retention=_read_int("QQ_GROUP_CHATTER_FILE_LOG_RETENTION", 5),
    )


def should_emit_loguru_record(
    record: dict[str, Any],
    *,
    project_min_level: int,
    framework_min_level: int,
) -> bool:
    message = str(record.get("message", ""))
    if _is_noisy_onebot_event(message):
        return False

    name = str(record.get("name", ""))
    level_no = int(getattr(record.get("level"), "no", logging.INFO))
    extra = record.get("extra") or {}
    is_project_log = bool(extra.get("qq_group_chatter_project")) or name.startswith(PROJECT_LOGGER_NAME)
    min_level = project_min_level if is_project_log else framework_min_level
    return level_no >= min_level


def _create_loguru_handler() -> logging.Handler:
    try:
        from nonebot.log import logger

        return _ProjectLoguruHandler(logger)
    except Exception:
        return logging.StreamHandler(sys.stdout)


def _is_noisy_onebot_event(message: str) -> bool:
    return "OneBot V11" in message and "[message" in message


def _read_level(env_name: str, default: int) -> int:
    raw = os.getenv(env_name)
    if not raw:
        return default
    normalized = raw.strip().upper()
    if normalized.isdigit():
        return int(normalized)
    return int(getattr(logging, normalized, default))


def _read_bool(env_name: str, default: bool) -> bool:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _read_int(env_name: str, default: int) -> int:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


class _ProjectLoguruHandler(logging.Handler):
    def __init__(self, loguru_logger: Any):
        super().__init__()
        self._logger = loguru_logger

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = self._logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        self._logger.bind(qq_group_chatter_project=True).opt(
            depth=depth,
            exception=record.exc_info,
        ).log(level, record.getMessage())
