from __future__ import annotations

import argparse
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_COLLECTION = "qq_group_chatter_memories_baai_bge_small_zh_v1_5_512d"
RESERVED_TIMESTAMP_KEYS = ("created_at", "updated_at")


def fix_timestamp_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return payload updates for Mem0 records with numeric reserved timestamps."""
    updates: dict[str, Any] = {}
    for key in RESERVED_TIMESTAMP_KEYS:
        value = payload.get(key)
        if not _is_number(value):
            continue
        if key == "created_at" and "source_created_at" not in payload:
            updates["source_created_at"] = value
        updates[key] = _timestamp_to_iso_utc(value)

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata_updates = fix_timestamp_payload(metadata)
        if metadata_updates:
            updates["metadata"] = {**metadata, **metadata_updates}

    return updates


def migrate_qdrant_payloads(
    *,
    qdrant_path: str | Path = ".mem0/qdrant",
    collection_name: str = DEFAULT_COLLECTION,
    dry_run: bool = False,
    batch_size: int = 100,
) -> int:
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(qdrant_path))
    try:
        offset = None
        changed = 0
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                updates = fix_timestamp_payload(payload)
                if not updates:
                    continue
                changed += 1
                if not dry_run:
                    client.set_payload(
                        collection_name=collection_name,
                        payload=updates,
                        points=[point.id],
                    )
            if offset is None:
                return changed
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fix local Mem0/Qdrant payloads that contain numeric created_at/updated_at fields."
    )
    parser.add_argument("--qdrant-path", default=".mem0/qdrant")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        changed = migrate_qdrant_payloads(
            qdrant_path=args.qdrant_path,
            collection_name=args.collection,
            dry_run=args.dry_run,
        )
    except sqlite3.OperationalError as exc:
        if "disk i/o error" in str(exc).lower():
            print(
                "failed: local Qdrant SQLite returned disk I/O error. "
                "Stop the bot and any other Python process using .mem0/qdrant, then rerun this command."
            )
            return 1
        raise
    action = "would update" if args.dry_run else "updated"
    print(f"{action} {changed} point(s)")
    return 0


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _timestamp_to_iso_utc(value: int | float) -> str:
    return datetime.fromtimestamp(float(value), tz=UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
