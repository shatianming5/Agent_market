"""Sealed holdout validation — run exactly once at end of mining."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from agent_market import paths
from agent_market.backtest_results import build_backtest_summary, find_latest_backtest_zip
from .dtypes import MinerConfig, StrategyCandidate

logger = logging.getLogger(__name__)


def run_sealed_holdout(
    candidate: StrategyCandidate,
    config: MinerConfig,
    run_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Run the sealed holdout backtest — called exactly ONCE at run completion.

    Returns holdout summary dict or None if holdout is not configured/fails.
    """
    holdout_timerange = str(getattr(config, "holdout_timerange", "") or "").strip()
    if not holdout_timerange:
        logger.info("No holdout_timerange configured, skipping sealed holdout")
        return None

    sandbox = candidate.strategy_path.parent.parent.parent
    strategies_dir = sandbox / "user_data" / "strategies"
    if not strategies_dir.exists():
        strategies_dir = candidate.strategy_path.parent

    ft_config = paths.resolve_repo_path(config.freqtrade_config)
    results_dir = sandbox / "user_data" / "backtest_results" / "holdout"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Clear any stale results
    for stale in results_dir.glob("backtest-result-*.zip"):
        try:
            stale.unlink()
        except Exception:
            pass

    cmd = [
        sys.executable, "-m", "freqtrade", "backtesting",
        "--config", str(ft_config),
        "--strategy", candidate.name,
        "--strategy-path", str(strategies_dir),
        "--timerange", holdout_timerange,
        "--userdir", str(sandbox / "user_data"),
    ]

    logger.info("Running sealed holdout for %s on %s", candidate.name, holdout_timerange)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(paths.REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=config.backtest_timeout,
            check=False,
        )
        if proc.returncode != 0:
            logger.warning("Sealed holdout failed (rc=%d): %s", proc.returncode, (proc.stderr or "")[-300:])
            return None

        zip_path = find_latest_backtest_zip(results_dir)
        if zip_path is None:
            logger.warning("No holdout backtest result found")
            return None

        summary = build_backtest_summary(zip_path)

        # Compare selection vs holdout performance
        selection_profit = (candidate.backtest_summary or {}).get("profit_total_pct", 0)
        holdout_profit = summary.get("profit_total_pct", 0)
        delta = abs(selection_profit - (holdout_profit or 0))

        result = {
            "holdout_timerange": holdout_timerange,
            "holdout_summary": summary,
            "selection_profit_pct": selection_profit,
            "holdout_profit_pct": holdout_profit,
            "delta_pct": delta,
            "overfitting_flag": delta > abs(selection_profit) * 0.5 if selection_profit else False,
        }

        logger.info(
            "Sealed holdout result: selection=%.2f%% holdout=%.2f%% delta=%.2f%% overfit=%s",
            selection_profit or 0, holdout_profit or 0, delta,
            result["overfitting_flag"],
        )
        return result
    except subprocess.TimeoutExpired:
        logger.warning("Sealed holdout timed out")
        return None
    except Exception as exc:
        logger.warning("Sealed holdout error: %s", exc)
        return None
