from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from zoneinfo import ZoneInfoNotFoundError


def test_time_utils_falls_back_to_fixed_utc8_when_zoneinfo_data_is_missing(monkeypatch):
    import zoneinfo

    def missing_zoneinfo(key):
        raise ZoneInfoNotFoundError(key)

    monkeypatch.setattr(zoneinfo, "ZoneInfo", missing_zoneinfo)
    module_path = Path(__file__).resolve().parents[1] / "qq_group_chatter" / "time_utils.py"
    spec = importlib.util.spec_from_file_location("time_utils_without_tzdata", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    sys.modules.pop("time_utils_without_tzdata", None)
    spec.loader.exec_module(module)

    assert module.format_time_text(1781529229.0) == "2026-06-15 21:13"
