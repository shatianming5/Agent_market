"""Reward computation from backtest results with optional factor-level scoring."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _clip(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _safe_float(v: Any) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def compute_reward(
    backtest_summary: Dict[str, Any],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Compute a scalar reward in [-1, 1] from backtest summary fields.

    Returns (reward, component_scores).
    """
    profit_pct = _safe_float(backtest_summary.get("profit_total_pct"))
    trades = _safe_float(backtest_summary.get("trades"))
    winrate = _safe_float(backtest_summary.get("winrate"))
    max_dd = _safe_float(backtest_summary.get("max_drawdown_abs"))
    avg_profit = _safe_float(backtest_summary.get("avg_profit_pct"))

    # Normalize each metric to roughly [-1, 1]
    # profit_pct: -50..+100 → -1..1
    s_profit = _clip(profit_pct / 50.0, -1.0, 1.0)

    # Sharpe proxy: avg_profit / (1 + |max_dd|) normalized
    dd_denom = 1.0 + abs(max_dd)
    raw_sharpe = avg_profit / dd_denom if dd_denom > 0 else 0.0
    s_sharpe = _clip(raw_sharpe / 2.0, -1.0, 1.0)

    # max_drawdown: 0 is best, -100 is worst → higher is better
    s_drawdown = _clip(1.0 - abs(max_dd) / 50.0, -1.0, 1.0)

    # winrate: 0..1 → -1..1  (0.5 = neutral)
    s_winrate = _clip((winrate - 0.5) * 4.0, -1.0, 1.0)

    # trade_count: want 10-200 trades. Too few or too many penalized.
    if trades < 5:
        s_trades = -0.5
    elif trades < 10:
        s_trades = 0.0
    elif trades <= 200:
        s_trades = 1.0
    elif trades <= 500:
        s_trades = 0.5
    else:
        s_trades = 0.0

    # stability: combine drawdown and winrate
    s_stability = (s_drawdown + s_winrate) / 2.0

    components = {
        "sharpe": s_sharpe,
        "profit_pct": s_profit,
        "max_drawdown": s_drawdown,
        "winrate": s_winrate,
        "trade_count": s_trades,
        "stability": s_stability,
    }

    total_weight = sum(weights.get(k, 0.0) for k in components)
    if total_weight <= 0:
        return 0.0, components

    reward = sum(
        weights.get(k, 0.0) * v for k, v in components.items()
    ) / total_weight

    reward = _clip(reward, -1.0, 1.0)
    return reward, components


def compute_factor_score(
    features_parquet: Optional[Path] = None,
    expression: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Run factor-level scoring using factor_compiler.scoring if available.

    Returns scoring dict or None if dependencies are missing.
    """
    if features_parquet is None or expression is None:
        return None

    try:
        import pandas as pd
        from agent_market.factor_compiler.scoring.aggregate import score_factors_to_artifacts
    except ImportError:
        logger.debug("factor_compiler.scoring not available, skipping factor-level scoring")
        return None

    features_path = Path(features_parquet)
    if not features_path.exists():
        logger.debug("Features parquet not found: %s", features_path)
        return None

    try:
        df = pd.read_parquet(features_path)
    except Exception as e:
        logger.warning("Failed to read features parquet: %s", e)
        return None

    # Check for target column
    target_col = None
    for candidate in ("target", "return_1h", "return_4h", "close_pct"):
        if candidate in df.columns:
            target_col = candidate
            break
    if target_col is None:
        logger.debug("No target column found in features, skipping factor scoring")
        return None

    # Find factor columns (non-target numeric)
    factor_cols = [
        c for c in df.select_dtypes(include=["number"]).columns
        if c != target_col and not c.startswith("_")
    ]
    if not factor_cols:
        return None

    if out_dir is None:
        import tempfile
        out_dir = Path(tempfile.mkdtemp(prefix="factor_score_"))
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = score_factors_to_artifacts(
            df,
            factor_cols=factor_cols[:20],
            target_col=target_col,
            out_dir=out_dir,
        )
        items = result.get("items", [])
        if items:
            best = max(items, key=lambda x: abs(x.get("ic", 0) or 0))
            return {
                "best_factor": best.get("name"),
                "best_ic": best.get("ic"),
                "best_sharpe": best.get("sharpe_net"),
                "factors_scored": len(items),
                "pareto_front": result.get("pareto_front", []),
            }
    except Exception as e:
        logger.warning("Factor scoring failed: %s", e)

    return None


def compute_enhanced_reward(
    backtest_summary: Dict[str, Any],
    weights: Dict[str, float],
    factor_scores: Optional[Dict[str, Any]] = None,
    factor_weight: float = 0.10,
) -> Tuple[float, Dict[str, float]]:
    """Enhanced reward combining backtest metrics and factor-level scores.

    The factor score is an optional bonus/penalty on top of the base reward.
    """
    base_reward, components = compute_reward(backtest_summary, weights)

    if factor_scores is None:
        return base_reward, components

    # Factor IC contributes a bonus
    best_ic = _safe_float(factor_scores.get("best_ic"))
    s_factor_ic = _clip(best_ic * 10.0, -1.0, 1.0)

    # Factor sharpe contributes a bonus
    best_sharpe = _safe_float(factor_scores.get("best_sharpe"))
    s_factor_sharpe = _clip(best_sharpe / 2.0, -1.0, 1.0)

    s_factor = (s_factor_ic + s_factor_sharpe) / 2.0
    components["factor_quality"] = s_factor

    enhanced = base_reward * (1.0 - factor_weight) + s_factor * factor_weight
    enhanced = _clip(enhanced, -1.0, 1.0)

    return enhanced, components
