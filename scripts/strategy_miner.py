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
    parser.add_argument(
        "--allow-defaults", action="store_true",
        help="opt into MinerConfig() builtin defaults when no --config and "
             "configs/strategy_miner_default.json is missing. Without this "
             "flag, the run fail-closes (Codex review R1-#2 — silent default "
             "fallback was a remote-deployment foot-gun)."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from agent_market.strategy_miner.dtypes import MinerConfig
    from agent_market.strategy_miner.runner import run_strategy_miner

    resume_path = Path(args.resume) if args.resume else None
    config = None

    def _load_config_from_path(path_like: str | Path | None, *, strict: bool = False) -> MinerConfig | None:
        """Load MinerConfig from JSON. When ``strict`` (explicit ``--config``),
        any failure raises ``SystemExit`` instead of returning None. Codex
        review R2 — silent fallback on bad explicit config was a remote
        foot-gun (run looked OK but ran with builtin defaults).
        """
        if not path_like:
            return None
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

    if args.config:
        # Explicit --config: any load failure must fail-close (R2 fix)
        config = _load_config_from_path(args.config, strict=True)

    if config is None and resume_path is not None:
        proposal = resume_path.parent / "proposal.json"
        # Codex review R3 fix: --resume without --config MUST locate the
        # original config from proposal.json, never fall back to defaults
        # (which would silently continue an old checkpoint with the wrong
        # config — extremely hard to detect from metrics alone).
        if not proposal.exists():
            raise SystemExit(
                f"strategy_miner: --resume {resume_path} requires either "
                f"explicit --config <path> OR a sibling proposal.json. "
                f"Neither was provided/found. Re-running an old checkpoint "
                f"with project default config silently changes experimental "
                f"truth — refusing fail-closed."
            )
        if proposal.exists():
            try:
                payload = json.loads(proposal.read_text(encoding="utf-8"))
                proposal_cfg = payload.get("config")
                if isinstance(proposal_cfg, dict):
                    config = MinerConfig.from_dict(proposal_cfg)
                    logging.info("Loaded config from proposal: %s", proposal)
                else:
                    raise SystemExit(
                        f"strategy_miner: --resume {resume_path} but its "
                        f"proposal.json has no `config` block. Re-run with "
                        f"explicit --config <path>."
                    )
            except (OSError, json.JSONDecodeError) as exc:
                # R2 fix: bad proposal under --resume must fail-close, not
                # silently fall through to default config.
                raise SystemExit(
                    f"strategy_miner: --resume {resume_path} but proposal.json "
                    f"failed to parse: {exc}. Re-run with explicit --config "
                    f"<path> or fix the proposal."
                )

    if config is None:
        config = _load_config_from_path("configs/strategy_miner_default.json")

    if config is None:
        # Codex review R1-#2 (remote readiness): refusing to silently fall
        # back to MinerConfig() defaults. On remote, that path used to look
        # like "the run worked", but actually used neither the user's
        # config nor the project default — extremely confusing. Caller must
        # opt into defaults explicitly via --allow-defaults.
        if not getattr(args, "allow_defaults", False):
            raise SystemExit(
                "strategy_miner: --config not provided and "
                "configs/strategy_miner_default.json missing. "
                "Pass --config <path> or --allow-defaults to opt into "
                "MinerConfig() builtin defaults explicitly."
            )
        logging.warning("No valid config found; --allow-defaults active, using MinerConfig builtin defaults")
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
