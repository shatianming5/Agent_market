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
    translated: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
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
        "[gp_factor_mine] deprecated wrapper; forwarding to scripts/gp_factor_mine_v2.py",
        file=sys.stderr,
    )
    sys.argv = [str(_V2_PATH), *_translate_legacy_args(sys.argv[1:])]
    return int(_load_v2().main())


if __name__ == "__main__":
    raise SystemExit(main())
