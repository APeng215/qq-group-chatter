import logging

from qq_group_chatter.logging_config import (
    configure_runtime_logging,
    configure_project_logging,
    should_emit_loguru_record,
)


def loguru_record(*, name, message, level_no, level_name="INFO"):
    return {
        "name": name,
        "message": message,
        "level": logging.getLevelName(level_no)
        if not isinstance(logging.getLevelName(level_no), str)
        else type("Level", (), {"no": level_no, "name": level_name})(),
        "extra": {},
    }


def test_framework_info_is_hidden_when_framework_level_is_warning():
    record = loguru_record(
        name="nonebot",
        message="Matcher(type='message') running complete",
        level_no=logging.INFO,
    )

    assert (
        should_emit_loguru_record(
            record,
            project_min_level=logging.INFO,
            framework_min_level=logging.WARNING,
        )
        is False
    )


def test_project_info_is_visible_when_framework_level_is_warning():
    record = loguru_record(
        name="qq_group_chatter.observability",
        message='{"event": "message_handled"}',
        level_no=logging.INFO,
    )

    assert (
        should_emit_loguru_record(
            record,
            project_min_level=logging.INFO,
            framework_min_level=logging.WARNING,
        )
        is True
    )


def test_runtime_logging_defaults_show_framework_info(monkeypatch):
    calls = {}
    monkeypatch.delenv("QQ_GROUP_CHATTER_LOG_LEVEL", raising=False)
    monkeypatch.delenv("QQ_GROUP_CHATTER_FRAMEWORK_LOG_LEVEL", raising=False)

    def fake_configure_project_logging(level):
        calls["project_level"] = level

    def fake_configure_loguru_logging(*, project_min_level, framework_min_level):
        calls["project_min_level"] = project_min_level
        calls["framework_min_level"] = framework_min_level

    monkeypatch.setattr(
        "qq_group_chatter.logging_config.configure_project_logging",
        fake_configure_project_logging,
    )
    monkeypatch.setattr(
        "qq_group_chatter.logging_config.configure_loguru_logging",
        fake_configure_loguru_logging,
    )

    configure_runtime_logging()

    assert calls == {
        "project_level": logging.INFO,
        "project_min_level": logging.INFO,
        "framework_min_level": logging.INFO,
    }


def test_project_marker_is_visible_when_record_name_is_not_preserved():
    record = loguru_record(
        name="__main__",
        message='{"event": "message_handled"}',
        level_no=logging.INFO,
    )
    record["extra"]["qq_group_chatter_project"] = True

    assert (
        should_emit_loguru_record(
            record,
            project_min_level=logging.INFO,
            framework_min_level=logging.WARNING,
        )
        is True
    )


def test_onebot_message_sent_payload_is_always_hidden():
    record = loguru_record(
        name="nonebot",
        message="OneBot V11 3998270681 | [message_sent]: {'raw_message': 'secret'}",
        level_no=25,
        level_name="SUCCESS",
    )

    assert (
        should_emit_loguru_record(
            record,
            project_min_level=logging.INFO,
            framework_min_level=logging.INFO,
        )
        is False
    )


def test_onebot_received_event_payload_is_always_hidden():
    record = loguru_record(
        name="nonebot",
        message=(
            "OneBot V11 3998270681 | [message.group.normal]: "
            "{'self_id': 3998270681, 'user_id': 12345678901, "
            "'group_id': 888888, 'raw_message': 'secret'}"
        ),
        level_no=logging.INFO,
    )

    assert (
        should_emit_loguru_record(
            record,
            project_min_level=logging.INFO,
            framework_min_level=logging.INFO,
        )
        is False
    )


def test_configure_project_logging_replaces_handlers_and_disables_propagation():
    logger = logging.getLogger("qq_group_chatter")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    handler = logging.NullHandler()

    try:
        logger.addHandler(logging.StreamHandler())

        configure_project_logging(level=logging.INFO, handler=handler)

        assert logger.handlers == [handler]
        assert logger.level == logging.INFO
        assert logger.propagate is False
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate
