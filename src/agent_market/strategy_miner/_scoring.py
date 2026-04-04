from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ._helpers import _coerce_float
from .dtypes import MinerConfig, StrategyCandidate


def _compute_psr(sharpe: float, observations: int, total_candidates: int = 100) -> float:
    """Deflate a sharpe-like score for small samples and multiple testing."""
    import math
    if observations <= 2 or not math.isfinite(sharpe):
        return 0.0
    se = math.sqrt((1 + 0.25 * sharpe ** 2) / observations)
    if se <= 0:
        return sharpe
    if total_candidates > 1:
        expected_max = math.sqrt(2 * math.log(total_candidates)) * se
    else:
        expected_max = 0.0
    deflated = sharpe - expected_max
    return deflated


def _compute_effective_score(
    *,
    config: MinerConfig,
    sharpe: float,
    observations: int,
    trades: int,
    winrate: float,
    profit_factor: float,
    profit_pct: float,
    max_drawdown_pct: float,
    positive_days_ratio: float,
    return_over_drawdown: float,
    metric_flags: list[str] | None,
    total_candidates: int,
) -> float:
    """Reward robust daily edge, realized profit, turnover, and stability."""

    import math

    clipped_sharpe = max(-4.0, min(4.0, float(sharpe)))
    psr = _compute_psr(clipped_sharpe, max(3, observations), total_candidates=total_candidates)

    target_trades = max(
        1,
        int(getattr(config, "target_trades", 0) or 0),
        int(getattr(config, "min_trades", 0) or 0),
    )
    min_acceptable_trades = int(getattr(config, "min_acceptable_trades", 0) or 0)
    if min_acceptable_trades <= 0:
        min_acceptable_trades = max(1, int(getattr(config, "min_trades", 0) or 0))

    # DCA/grid strategies have lower trade counts but higher per-trade value
    dca_mode = bool(getattr(config, "position_adjustment_enable", False))
    effective_target = max(1, target_trades // 3) if dca_mode else target_trades
    effective_min = max(1, min_acceptable_trades // 3) if dca_mode else min_acceptable_trades

    trade_bonus = 0.0
    if trades > 0:
        trade_ratio = float(trades) / float(effective_target)
        trade_bonus += 0.8 * min(1.0, trade_ratio)
        if trade_ratio > 1.0:
            trade_bonus += 0.2 * min(1.0, trade_ratio - 1.0)

    if effective_min > 0 and trades < effective_min:
        shortage = float(effective_min - trades) / float(effective_min)
        trade_bonus -= 0.8 * min(1.0, shortage)

    winrate_floor = 0.50
    min_winrate = float(getattr(config, "min_winrate", 0.0) or 0.0)
    if min_winrate > 0:
        winrate_floor = max(0.45, min(0.75, min_winrate - 0.05))
    winrate_bonus = 0.0
    if winrate > winrate_floor:
        denom = max(0.05, 1.0 - winrate_floor)
        winrate_bonus = 0.45 * min(1.0, (winrate - winrate_floor) / denom)

    pf_floor = max(1.0, float(getattr(config, "min_profit_factor", 0.0) or 0.0), 1.15)
    pf_bonus = 0.0
    if math.isfinite(profit_factor) and profit_factor > 1.0:
        denom = max(0.25, pf_floor - 1.0)
        pf_bonus = 0.45 * min(1.0, (profit_factor - 1.0) / denom)
        if profit_factor > pf_floor:
            pf_bonus += 0.15 * min(1.0, (profit_factor - pf_floor) / max(0.25, pf_floor))

    profit_bonus = 0.0
    if math.isfinite(profit_pct):
        profit_bonus = 0.50 * math.tanh(profit_pct / 12.0)

    drawdown_bonus = 0.0
    if math.isfinite(max_drawdown_pct) and max_drawdown_pct > 0:
        drawdown_bonus -= 0.30 * min(1.5, max_drawdown_pct / 12.0)
        if math.isfinite(return_over_drawdown) and return_over_drawdown > 0:
            drawdown_bonus += 0.25 * min(1.0, return_over_drawdown / 3.0)

    day_bonus = 0.0
    if math.isfinite(positive_days_ratio) and positive_days_ratio > 0.5:
        day_bonus = 0.25 * min(1.0, (positive_days_ratio - 0.5) / 0.3)

    metric_penalty = 0.0
    if metric_flags:
        metric_penalty -= 0.15 * min(3.0, float(len(metric_flags)))

    return psr + trade_bonus + winrate_bonus + pf_bonus + profit_bonus + drawdown_bonus + day_bonus + metric_penalty


def _training_score_adjustment(candidate: StrategyCandidate) -> float:
    summary = getattr(candidate, "training_summary", None) or {}
    if not isinstance(summary, dict):
        return 0.0
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    rolling = summary.get("rolling") if isinstance(summary.get("rolling"), dict) else {}

    train_rmse = _coerce_float(metrics.get("rmse_train"), 0.0)
    valid_rmse = _coerce_float(metrics.get("rmse_valid"), 0.0)
    adjust = 0.0

    if train_rmse > 0 and valid_rmse > 0:
        ratio = valid_rmse / max(train_rmse, 1e-9)
        if ratio <= 1.05:
            adjust += 0.15
        elif ratio >= 1.50:
            adjust -= 0.35
        elif ratio >= 1.25:
            adjust -= 0.15

    rolling_mean = _coerce_float(rolling.get("rmse_valid_mean"), 0.0)
    rolling_std = _coerce_float(rolling.get("rmse_valid_std"), 0.0)
    if rolling_mean > 0:
        if rolling_std <= rolling_mean * 0.15:
            adjust += 0.10
        elif rolling_std >= rolling_mean * 0.40:
            adjust -= 0.10

    return adjust


def _training_robustness_violations(candidate: StrategyCandidate) -> list[str]:
    summary = getattr(candidate, "training_summary", None) or {}
    if not isinstance(summary, dict):
        return []

    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    rolling = summary.get("rolling") if isinstance(summary.get("rolling"), dict) else {}

    train_rmse = _coerce_float(metrics.get("rmse_train"), 0.0)
    valid_rmse = _coerce_float(metrics.get("rmse_valid"), 0.0)
    rolling_mean = _coerce_float(rolling.get("rmse_valid_mean"), 0.0)
    rolling_std = _coerce_float(rolling.get("rmse_valid_std"), 0.0)

    violations: list[str] = []
    if train_rmse > 0 and valid_rmse > 0:
        ratio = valid_rmse / max(train_rmse, 1e-9)
        if ratio >= 1.75:
            violations.append(f"validation_overfit:{ratio:.2f}")

    if rolling_mean > 0:
        stability_ratio = rolling_std / max(rolling_mean, 1e-9)
        if stability_ratio >= 0.75:
            violations.append(f"rolling_instability:{stability_ratio:.2f}")

    return violations


def _check_per_pair_robustness(
    backtest_summary: Dict[str, Any],
    *,
    min_pair_profit_pct: float = -0.5,
) -> tuple[bool, List[str]]:
    """Check that each pair stays above the configured loss floor."""
    violations = []
    results_per_pair = backtest_summary.get("results_per_pair") or backtest_summary.get("pair_results")
    if not results_per_pair:
        return True, []  # Can't check, pass by default
    def _pair_profit(d: dict) -> float:
        for key in ("profit_total_pct", "profit_pct", "profit_total"):
            v = d.get(key)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return 0.0

    if isinstance(results_per_pair, list):
        for pair_data in results_per_pair:
            if not isinstance(pair_data, dict):
                continue
            pair = pair_data.get("pair") or pair_data.get("key", "?")
            profit = _pair_profit(pair_data)
            if profit < float(min_pair_profit_pct):
                violations.append(f"pair:{pair} profit={profit:.2f}%")
    elif isinstance(results_per_pair, dict):
        for pair, pair_data in results_per_pair.items():
            if not isinstance(pair_data, dict):
                continue
            profit = _pair_profit(pair_data)
            if profit < float(min_pair_profit_pct):
                violations.append(f"pair:{pair} profit={profit:.2f}%")
    return len(violations) == 0, violations


def _safe_metric(summary: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Extract a float metric from backtest summary, defaulting on None/error."""
    try:
        v = summary.get(key)
        if v is None:
            return default
        f = float(v)
        import math
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _safe_ratio_metric(summary: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in summary:
            return _safe_metric(summary, key, default)
    return default
