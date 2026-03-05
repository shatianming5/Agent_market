#!/usr/bin/env python3
"""Backtest+summarize a Strategy Miner candidate.

Designed to run as a JobManager background task.
Outputs are written under: runs/<run_id>/strategy_miner/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Strategy miner candidate backtest")
    parser.add_argument("--run-id", required=True, help="Run id (hex)")
    parser.add_argument("--candidate", default=None, help="Candidate strategy name")
    parser.add_argument(
        "--config",
        default="configs/strategy_miner_default.json",
        help="Miner config JSON (for freqtrade_config/timerange/timeout/reward weights)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from agent_market.strategy_miner.dtypes import MinerConfig, MinerState, Phase
    from agent_market.strategy_miner.runner import miner_run_dir, _save_checkpoint
    from agent_market.strategy_miner.phases import phase_backtest, phase_evaluation

    # Load miner config (best-effort)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _REPO / cfg_path
    if cfg_path.exists():
        cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        config = MinerConfig.from_dict(cfg_raw)
    else:
        logging.getLogger(__name__).warning("Config not found: %s, using defaults", cfg_path)
        config = MinerConfig()

    run_id = str(args.run_id).strip().lower()
    miner_dir = miner_run_dir(run_id)
    cp = miner_dir / "checkpoint.json"
    if not cp.exists():
        raise FileNotFoundError(f"checkpoint not found: {cp}")

    state = MinerState.from_dict(json.loads(cp.read_text(encoding="utf-8")))

    # Select candidate
    target = (args.candidate or "").strip()
    chosen = None
    if target:
        for c in state.candidates:
            if c.name == target:
                chosen = c
                break
        if chosen is None:
            raise ValueError(f"candidate not found: {target}")
    else:
        chosen = state.best_candidate or (state.candidates[-1] if state.candidates else None)

    if chosen is None:
        raise ValueError("no candidates in checkpoint")

    # Make chosen the last candidate for phase functions.
    state.candidates = [c for c in state.candidates if c.name != chosen.name] + [chosen]
    state.iteration = int(chosen.iteration or state.iteration)
    state.phase = Phase.BACKTEST

    phase_backtest(state, config, miner_dir)
    if state.phase == Phase.EVALUATION:
        phase_evaluation(state, config, run_dir=miner_dir)

    # Persist checkpoint updates
    _save_checkpoint(state, miner_dir)

    # Always write a stable job summary artifact.
    out_dir = miner_dir / "backtests" / "jobs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{chosen.name}__iter_{int(chosen.iteration):04d}.json"

    payload = {
        "run_id": state.run_id,
        "candidate": chosen.to_dict(),
        "phase": state.phase.value,
        "iteration": state.iteration,
        "best_reward": state.best_reward,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.getLogger(__name__).info("Wrote %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
