#!/usr/bin/env python3
"""Compatibility wrapper for ``agent_market.strategy_miner.cli``."""
from __future__ import annotations

import sys
from pathlib import Path


_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from agent_market.strategy_miner.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
