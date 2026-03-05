#!/usr/bin/env python3
"""CLI wrapper for the strategy miner."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src/ is on sys.path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy-level mining via LLM Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/strategy_miner_default.json",
        help="Path to miner config JSON",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint.json for resuming")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override max iterations")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from agent_market.strategy_miner.dtypes import MinerConfig
    from agent_market.strategy_miner.runner import run_strategy_miner

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _REPO / config_path
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        config = MinerConfig.from_dict(raw)
    else:
        logging.warning("Config not found: %s, using defaults", config_path)
        config = MinerConfig()

    # CLI overrides
    if args.max_iterations is not None:
        config.max_iterations = args.max_iterations
    if args.model is not None:
        config.model = args.model

    resume_path = Path(args.resume) if args.resume else None
    state = run_strategy_miner(config, resume=resume_path)

    print(f"\nMining complete: run_id={state.run_id}")
    print(f"  Iterations: {state.iteration}")
    print(f"  Best reward: {state.best_reward:.4f}")
    if state.best_candidate:
        print(f"  Best strategy: {state.best_candidate.name}")


if __name__ == "__main__":
    main()
