"""Candidate portfolio construction for strategy miner final outputs (D12)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from .dtypes import StrategyCandidate


@dataclass
class PortfolioCandidate:
    name: str
    reward: float
    returns: "pd.Series"
    summary: Dict[str, Any]


def _daily_return_series(candidate: StrategyCandidate) -> "pd.Series | None":
    summary = dict(candidate.backtest_summary or {})
    rows = list(summary.get("daily_profit") or [])
    if not rows:
        return None

    index: list[str] = []
    values: list[float] = []
    equity = float(summary.get("starting_balance") or 1.0)
    if abs(equity) <= 1e-12:
        equity = 1.0

    for row in rows:
        day = None
        profit_abs = None
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            day = str(row[0])
            profit_abs = row[1]
        elif isinstance(row, dict):
            day = str(row.get("date") or row.get("day") or "")
            profit_abs = row.get("profit_abs", row.get("profit_total_abs", row.get("profit")))
        if not day:
            continue
        try:
            profit_f = float(profit_abs)
        except Exception:
            continue
        base_equity = equity if abs(equity) > 1e-12 else 1.0
        values.append(profit_f / base_equity)
        index.append(day)
        equity += profit_f

    if len(values) < 3:
        return None
    series = pd.Series(values, index=pd.to_datetime(index), name=candidate.name, dtype="float64")
    return series.sort_index()


def _cap_weights(weights: Dict[str, float], max_weight: float) -> Dict[str, float]:
    if not weights:
        return {}
    max_weight = float(max_weight)
    if max_weight <= 0 or max_weight >= 1:
        total = sum(float(v) for v in weights.values()) or 1.0
        return {k: float(v) / total for k, v in weights.items()}

    remaining = {str(k): max(float(v), 0.0) for k, v in weights.items()}
    capped: Dict[str, float] = {}
    budget = 1.0

    while remaining:
        raw_total = sum(remaining.values()) or float(len(remaining))
        proposed = {
            name: (value / raw_total) * budget if raw_total > 0 else budget / len(remaining)
            for name, value in remaining.items()
        }
        over = [name for name, value in proposed.items() if value > max_weight + 1e-12]
        if not over:
            capped.update(proposed)
            break
        for name in over:
            capped[name] = max_weight
            budget -= max_weight
            remaining.pop(name, None)
        if budget <= 1e-12:
            break

    total = sum(capped.values()) or 1.0
    normalized = {k: float(v) / total for k, v in capped.items()}
    for key, value in list(normalized.items()):
        normalized[key] = round(float(value), 6)
    return normalized


def _fallback_equal_weight(columns: Iterable[str]) -> Dict[str, float]:
    names = [str(c) for c in columns]
    if not names:
        return {}
    weight = round(1.0 / len(names), 6)
    weights = {name: weight for name in names}
    diff = round(1.0 - sum(weights.values()), 6)
    if names:
        weights[names[0]] = round(weights[names[0]] + diff, 6)
    return weights


def _returns_stats(returns_df: "pd.DataFrame", weights: Dict[str, float]) -> Dict[str, Any]:
    if returns_df.empty or not weights:
        return {}
    weight_series = pd.Series(weights).reindex(returns_df.columns).fillna(0.0).astype("float64")
    port = (returns_df * weight_series).sum(axis=1)
    ann_factor = 365.0
    mean = float(port.mean())
    vol = float(port.std(ddof=1))
    sharpe = (mean * ann_factor) / (vol * (ann_factor ** 0.5)) if vol > 0 else None
    corr = returns_df.corr().fillna(0.0)
    abs_corr = corr.abs()
    upper = abs_corr.where(~np.tril(np.ones(abs_corr.shape)).astype(bool))
    flattened = [float(v) for v in upper.stack().tolist()]
    return {
        "n_assets": int(len(weights)),
        "ann_return": round(mean * ann_factor, 6),
        "ann_vol": round(vol * (ann_factor ** 0.5), 6),
        "sharpe": round(float(sharpe), 6) if sharpe is not None else None,
        "max_pairwise_corr": round(max(flattened), 6) if flattened else 0.0,
        "avg_pairwise_corr": round(sum(flattened) / len(flattened), 6) if flattened else 0.0,
        "max_weight": round(max(weights.values()), 6),
        "min_weight": round(min(weights.values()), 6),
    }


def build_candidate_portfolio_report(
    candidates: List[StrategyCandidate],
    *,
    top_k: int = 3,
    min_candidates: int = 2,
    correlation_threshold: float = 0.85,
    max_weight: float = 0.60,
) -> Dict[str, Any] | None:
    """Construct a diversified candidate portfolio from evaluated strategies."""
    ranked = sorted(
        [
            c
            for c in candidates
            if c.backtest_summary is not None and c.reward is not None and bool(getattr(c, "constraints_ok", True))
        ],
        key=lambda c: float(c.reward or -1e9),
        reverse=True,
    )
    if not ranked:
        return None

    pool: list[PortfolioCandidate] = []
    skipped: list[Dict[str, Any]] = []
    for candidate in ranked[: max(1, int(top_k))]:
        series = _daily_return_series(candidate)
        if series is None:
            skipped.append({"name": candidate.name, "reason": "missing_daily_profit"})
            continue
        pool.append(
            PortfolioCandidate(
                name=candidate.name,
                reward=float(candidate.reward or 0.0),
                returns=series,
                summary=dict(candidate.backtest_summary or {}),
            )
        )

    if not pool:
        return None

    returns_df = pd.concat([item.returns for item in pool], axis=1, join="inner").dropna(how="any")
    if returns_df.empty:
        return {
            "method": "unavailable",
            "passed": False,
            "reason": "no_overlapping_daily_returns",
            "inputs": {
                "top_k": int(top_k),
                "min_candidates": int(min_candidates),
                "correlation_threshold": float(correlation_threshold),
                "max_weight": float(max_weight),
            },
            "skipped": skipped,
        }

    corr = returns_df.corr().fillna(0.0)
    selected_names: list[str] = []
    correlation_drops: list[Dict[str, Any]] = []
    ranked_names = [item.name for item in pool if item.name in returns_df.columns]
    for name in ranked_names:
        if not selected_names:
            selected_names.append(name)
            continue
        max_corr = max(abs(float(corr.loc[name, other])) for other in selected_names)
        if max_corr > float(correlation_threshold):
            correlation_drops.append(
                {"name": name, "reason": "correlation_pruned", "max_corr_to_selected": round(max_corr, 6)}
            )
            continue
        selected_names.append(name)

    if len(selected_names) < int(min_candidates):
        selected_names = ranked_names[: max(1, min(len(ranked_names), int(min_candidates)))]

    selected_returns = returns_df[selected_names].copy()
    method = "hrp"
    try:
        from agent_market.portfolio_opt import optimize_hrp

        optimized = optimize_hrp(selected_returns)
        weights = {str(k): float(v) for k, v in dict(optimized.get("weights") or {}).items()}
        stats = dict(optimized.get("stats") or {})
    except Exception as exc:
        method = "equal_weight_fallback"
        weights = _fallback_equal_weight(selected_returns.columns)
        stats = {"warning": str(exc)[:300]}

    weights = _cap_weights(weights, float(max_weight))
    stats.update(_returns_stats(selected_returns, weights))

    return {
        "method": method,
        "passed": bool(weights),
        "weights": weights,
        "stats": stats,
        "inputs": {
            "top_k": int(top_k),
            "min_candidates": int(min_candidates),
            "correlation_threshold": float(correlation_threshold),
            "max_weight": float(max_weight),
        },
        "selected_candidates": [
            {
                "name": item.name,
                "reward": item.reward,
                "profit_total_pct": item.summary.get("profit_total_pct"),
                "max_drawdown_pct": item.summary.get("max_drawdown_pct"),
            }
            for item in pool
            if item.name in selected_names
        ],
        "dropped_candidates": skipped + correlation_drops,
        "correlation_matrix": corr.loc[selected_names, selected_names].round(6).to_dict()
        if selected_names
        else {},
    }
