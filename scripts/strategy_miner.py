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
        default=None,
        help="Path to miner config JSON",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint.json for resuming")
    parser.add_argument("--run-id", type=str, default=None, help="Override run_id (hex)")
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

    resume_path = Path(args.resume) if args.resume else None
    config = None

    def _load_config_from_path(path_like: str | Path | None) -> MinerConfig | None:
        if not path_like:
            return None
        config_path = Path(path_like)
        if not config_path.is_absolute():
            config_path = _REPO / config_path
        if not config_path.exists():
            logging.warning("Config not found: %s", config_path)
            return None
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        logging.info("Loaded config from %s", config_path)
        return MinerConfig.from_dict(raw)

    if args.config:
        config = _load_config_from_path(args.config)

    if config is None and resume_path is not None:
        proposal = resume_path.parent / "proposal.json"
        if proposal.exists():
            try:
                payload = json.loads(proposal.read_text(encoding="utf-8"))
                proposal_cfg = payload.get("config")
                if isinstance(proposal_cfg, dict):
                    config = MinerConfig.from_dict(proposal_cfg)
                    logging.info("Loaded config from proposal: %s", proposal)
            except Exception:
                logging.exception("Failed to load config from proposal: %s", proposal)

    if config is None:
        config = _load_config_from_path("configs/strategy_miner_default.json")

    if config is None:
        logging.warning("No valid config found, using MinerConfig defaults")
        config = MinerConfig()

    # CLI overrides
    if args.max_iterations is not None:
        config.max_iterations = args.max_iterations
    if args.model is not None:
        config.model = args.model

    state = run_strategy_miner(config, run_id=args.run_id, resume=resume_path)

    print(f"\nMining complete: run_id={state.run_id}")
    print(f"  Iterations: {state.iteration}")
    print(f"  Best reward: {state.best_score:.4f}")
    if state.best_candidate:
        print(f"  Best strategy: {state.best_candidate.name}")


if __name__ == "__main__":
    main()
