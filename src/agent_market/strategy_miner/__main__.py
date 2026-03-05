"""Allow running as: python -m agent_market.strategy_miner"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .dtypes import MinerConfig
from .runner import run_strategy_miner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strategy miner: automated trading strategy generation & evaluation",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to JSON config file (MinerConfig fields)",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to checkpoint.json for resuming a previous run",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Override run_id (hex)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override LLM model name",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="Override max iteration count",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    cfg_dict: dict = {}
    if args.config is not None:
        cfg_dict = json.loads(args.config.read_text(encoding="utf-8"))
    if args.model is not None:
        cfg_dict["model"] = args.model
    if args.max_iterations is not None:
        cfg_dict["max_iterations"] = args.max_iterations

    config = MinerConfig.from_dict(cfg_dict)

    try:
        state = run_strategy_miner(config, run_id=args.run_id, resume=args.resume)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Interrupted by user")
        return 130
    except Exception:
        logging.getLogger(__name__).exception("Strategy miner failed")
        return 1

    print(f"Run {state.run_id} finished: best_reward={state.best_reward:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
