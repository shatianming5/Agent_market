#!/usr/bin/env python3
"""Compatibility wrapper for the current GP factor miner.

The maintained implementation is ``scripts/gp_factor_mine_v2.py``.  This file
keeps the historical command working while avoiding a second, stale GP miner.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


_V2_PATH = Path(__file__).resolve().with_name("gp_factor_mine_v2.py")


def _load_v2() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_agent_market_gp_factor_mine_v2", _V2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load GP miner v2 from {_V2_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _translate_legacy_args(argv: list[str]) -> list[str]:
    fixed_windows = {
        "--tr3-start": "2024-01-01",
        "--tr3-end": "2025-07-01",
        "--v3-start": "2025-07-01",
        "--v3-end": "2025-12-01",
    }
    translated: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        flag, separator, value = arg.partition("=")
        if flag in fixed_windows:
            if not separator:
                if idx + 1 >= len(argv):
                    raise ValueError(f"{flag} requires a date")
                value = argv[idx + 1]
            legacy_data_start = flag == "--tr3-start" and value == "2023-05-15"
            if value != fixed_windows[flag] and not legacy_data_start:
                raise ValueError(
                    f"{flag}={value} cannot be preserved by v2's fixed training/validation windows "
                    f"({flag}={fixed_windows[flag]}); refusing to change the experiment silently"
                )
        if arg == "--tr3-start":
            translated.append("--data-start")
        elif arg.startswith("--tr3-start="):
            translated.append("--data-start=" + arg.split("=", 1)[1])
        elif arg == "--v3-end":
            translated.append("--data-end")
        elif arg.startswith("--v3-end="):
            translated.append("--data-end=" + arg.split("=", 1)[1])
        elif arg in {"--tr3-end", "--v3-start"}:
            idx += 1
        elif arg.startswith("--tr3-end=") or arg.startswith("--v3-start="):
            pass
        else:
            translated.append(arg)
        idx += 1
    return translated


def main() -> int:
    print(
        "[gp_factor_mine] deprecated wrapper; forwarding to scripts/gp_factor_mine_v2.py. "
        "v2 fitness uses fixed 6-month windows from 2024-01-01 to 2025-07-01, "
        "VAL3 ends 2025-12-01; earlier legacy data-start dates supply history only.",
        file=sys.stderr,
    )
    try:
        translated = _translate_legacy_args(sys.argv[1:])
    except ValueError as exc:
        print(f"[gp_factor_mine] error: {exc}", file=sys.stderr)
        return 2
    sys.argv = [str(_V2_PATH), *translated]
    return int(_load_v2().main())


if __name__ == "__main__":
    raise SystemExit(main())
