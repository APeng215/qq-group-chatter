from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path


def _run_pytest(args: list[str]) -> int:
    import pytest

    original_mkdir = Path.mkdir

    def mkdir_with_default_windows_permissions(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if os.name == "nt" and mode == 0o700:
            mode = 0o777
        return original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    Path.mkdir = mkdir_with_default_windows_permissions
    try:
        return int(pytest.main(args))
    finally:
        Path.mkdir = original_mkdir


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[1]
    temp_parent = project_root / "tests" / ".tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = temp_parent / f"pytest-parallel-{uuid.uuid4().hex}"

    args = ["-n", "auto", "-q", f"--basetemp={temp_root}"]
    if argv:
        args.extend(argv)
    return _run_pytest(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
