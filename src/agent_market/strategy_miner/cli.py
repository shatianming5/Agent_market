"""Strategy miner command-line entrypoint.

Both ``python scripts/strategy_miner.py`` and
``python -m agent_market.strategy_miner`` delegate here so config resolution is
single-sourced inside the package rather than in the scripts/ tree.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence


_REPO = Path(__file__).resolve().parents[3]


def _load_config_from_path(path_like: str | Path | None, *, strict: bool = False):
    """Load ``MinerConfig`` from JSON.

    When ``strict`` is true for an explicit ``--config``, failures raise
    ``SystemExit`` instead of falling back to defaults.
    """
    if not path_like:
        return None
    from agent_market.strategy_miner.dtypes import MinerConfig

    config_path = Path(path_like)
    if not config_path.is_absolute():
        config_path = _REPO / config_path
    if not config_path.exists():
        msg = f"Config not found: {config_path}"
        if strict:
            raise SystemExit(f"strategy_miner: {msg}")
        logging.warning(msg)
        return None
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Failed to read/parse config {config_path}: {exc}"
        if strict:
            raise SystemExit(f"strategy_miner: {msg}")
        logging.warning(msg)
        return None
    logging.info("Loaded config from %s", config_path)
    return MinerConfig.from_dict(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strategy-level mining via LLM Agent")
    parser.add_argument("--config", type=str, default=None, help="Path to miner config JSON")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint.json for resuming")
    parser.add_argument("--run-id", type=str, default=None, help="Override run_id (hex)")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override max iterations")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--allow-defaults",
        action="store_true",
        help=(
            "opt into MinerConfig() builtin defaults when no --config and "
            "configs/strategy_miner_default.json is missing. Without this flag, "
            "the run fail-closes."
        ),
    )
    return parser


def _config_from_resume(resume_path: Path):
    from agent_market.strategy_miner.dtypes import MinerConfig

    proposal = resume_path.parent / "proposal.json"
    if not proposal.exists():
        raise SystemExit(
            f"strategy_miner: --resume {resume_path} requires either explicit "
            f"--config <path> OR a sibling proposal.json. Neither was "
            f"provided/found. Re-running an old checkpoint with project default "
            f"config silently changes experimental truth - refusing fail-closed."
        )
    try:
        payload = json.loads(proposal.read_text(encoding="utf-8"))
        proposal_cfg = payload.get("config")
        if isinstance(proposal_cfg, dict):
            logging.info("Loaded config from proposal: %s", proposal)
            return MinerConfig.from_dict(proposal_cfg)
        raise SystemExit(
            f"strategy_miner: --resume {resume_path} but its proposal.json has "
            f"no `config` block. Re-run with explicit --config <path>."
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"strategy_miner: --resume {resume_path} but proposal.json failed "
            f"to parse: {exc}. Re-run with explicit --config <path> or fix the "
            f"proposal."
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from agent_market.strategy_miner.dtypes import MinerConfig
    from agent_market.strategy_miner.runner import run_strategy_miner

    resume_path = Path(args.resume) if args.resume else None
    config = _load_config_from_path(args.config, strict=True) if args.config else None

    if config is None and resume_path is not None:
        config = _config_from_resume(resume_path)

    if config is None:
        config = _load_config_from_path("configs/strategy_miner_default.json")

    if config is None:
        if not getattr(args, "allow_defaults", False):
            raise SystemExit(
                "strategy_miner: --config not provided and "
                "configs/strategy_miner_default.json missing. Pass --config "
                "<path> or --allow-defaults to opt into MinerConfig() builtin "
                "defaults explicitly."
            )
        logging.warning("No valid config found; --allow-defaults active, using MinerConfig builtin defaults")
        config = MinerConfig()

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
    return 0
