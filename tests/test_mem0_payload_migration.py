import sqlite3

from qq_group_chatter import mem0_payload_migration
from qq_group_chatter.mem0_payload_migration import fix_timestamp_payload


def test_fix_timestamp_payload_moves_numeric_created_at_and_stringifies_reserved_fields():
    payload = {
        "memory": "remember this",
        "created_at": 1781529229.0,
        "updated_at": 1781529230,
        "metadata": {"created_at": 1781529229.0, "kind": "relationship"},
    }

    fixed = fix_timestamp_payload(payload)

    assert fixed == {
        "source_created_at": 1781529229.0,
        "created_at": "2026-06-15T13:13:49Z",
        "updated_at": "2026-06-15T13:13:50Z",
        "metadata": {
            "source_created_at": 1781529229.0,
            "created_at": "2026-06-15T13:13:49Z",
            "kind": "relationship",
        },
    }


def test_fix_timestamp_payload_preserves_existing_source_created_at():
    payload = {
        "created_at": 1781529229.0,
        "source_created_at": 100.0,
        "metadata": {"created_at": 1781529230.0, "source_created_at": 101.0},
    }

    fixed = fix_timestamp_payload(payload)

    assert "source_created_at" not in fixed
    assert fixed["metadata"]["source_created_at"] == 101.0


def test_fix_timestamp_payload_returns_empty_when_no_numeric_reserved_timestamps():
    assert fix_timestamp_payload({"created_at": "2026-06-15T13:13:49Z"}) == {}


def test_main_reports_local_qdrant_disk_io_error(monkeypatch, capsys):
    def fail_migration(**kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(mem0_payload_migration, "migrate_qdrant_payloads", fail_migration)
    monkeypatch.setattr("sys.argv", ["mem0_payload_migration"])

    exit_code = mem0_payload_migration.main()

    assert exit_code == 1
    assert "Stop the bot" in capsys.readouterr().out
