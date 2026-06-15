import logging

from qq_group_chatter.models import build_group_conversation_context
from qq_group_chatter.observability import conversation_log_fields, record_error


def test_conversation_log_fields_hash_identifiers():
    context = build_group_conversation_context(
        group_id=770616062,
        user_id=3998270681,
        message_id="m1",
        nickname="tester",
        timestamp=123.0,
    )

    fields = conversation_log_fields(context)

    assert fields["conversation_id_hash"]
    assert fields["user_id_hash"]
    assert fields["group_id_hash"]
    assert "conversation_id" not in fields
    assert "770616062" not in str(fields)
    assert "3998270681" not in str(fields)


def test_record_error_redacts_sensitive_exception_message(caplog):
    caplog.set_level(logging.ERROR, logger="qq_group_chatter")

    try:
        raise RuntimeError(
            "failed with token=sk-secret api_key='ak-secret' phone 13800138000"
        )
    except RuntimeError as exc:
        record_error("unit_test", exc)

    logged = caplog.text
    assert "sk-secret" not in logged
    assert "ak-secret" not in logged
    assert "13800138000" not in logged
    assert "[REDACTED]" in logged
    assert "Traceback" in logged
    assert "RuntimeError" in logged


def test_record_error_handles_exception_types_that_require_custom_constructor(caplog):
    class CustomValidationError(Exception):
        def __init__(self, message, line_errors):
            super().__init__(message)
            self.line_errors = line_errors

    caplog.set_level(logging.ERROR, logger="qq_group_chatter")

    try:
        raise CustomValidationError("failed with token=sk-secret", line_errors=[])
    except CustomValidationError as exc:
        record_error("unit_test", exc)

    logged = caplog.text
    assert "sk-secret" not in logged
    assert "[REDACTED]" in logged
    assert "CustomValidationError" in logged
    assert "Traceback" in logged
