#!/usr/bin/env python3
"""Run strategy miner startup preflight without starting the mining loop."""
from __future__ import annotations

import argparse
import json
import logging
import time
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agent_market.strategy_miner.dtypes import MinerConfig, Phase
from agent_market.strategy_miner.runner import miner_run_dir
from agent_market.strategy_miner.preflight import run_startup_preflight


def _load_config(path_like: str | None) -> MinerConfig:
    if not path_like:
        return MinerConfig()
    path = Path(path_like)
    if not path.is_absolute():
        path = REPO / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    return MinerConfig.from_dict(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run startup preflight checks for strategy miner.")
    parser.add_argument("--config", type=str, default="configs/strategy_miner_default.json")
    parser.add_argument("--run-id", type=str, default=f"preflight-{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--phase", type=str, default=Phase.STRATEGY_GEN.value)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = _load_config(args.config)
    report = run_startup_preflight(
        cfg,
        miner_dir=miner_run_dir(args.run_id),
        phase=Phase(str(args.phase)),
        raise_on_error=False,
    )

    print(f"ok={report['ok']} warnings={report['warnings']} errors={report['errors']}")
    print(f"report={miner_run_dir(args.run_id) / 'preflight.json'}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
