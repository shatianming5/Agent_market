"""Local WQ-aligned dollar-neutral portfolio simulation.

Port of QuantGPT wq_simulate.py (MIT). Computes WQ BRAIN-aligned
sharpe / turnover / returns / fitness from a (date, ticker, factor_value,
daily_ret) DataFrame so we can pre-screen alpha candidates BEFORE spending
WQ daily simulation budget.

Adds a wrapper `simulate_expression_locally(expr, ohlcv_df, settings)` that
- parses expr via expr_parser
- computes factor_value across the universe by walking the AST against the
  cached OHLCV DataFrame
- calls wq_simulate
- returns the same shape as a SimulationResult
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _np():
    import numpy as np
    return np


def _pd():
    import pandas as pd
    return pd


# ── Core simulation (port) ──────────────────────────────────────────────

def wq_simulate(
    work_df,
    rebalance_dates: list,
    *,
    trading_days_per_year: int = 252,
    fitness_gate: float = 0.5,
) -> dict[str, Any]:
    """Simulate WQ-aligned dollar-neutral portfolio.

    Args:
        work_df: DataFrame with columns trade_date, stock_code, factor_value, daily_ret.
        rebalance_dates: sorted list of rebalance dates (subset of trade_date values).
    """
    np = _np()
    empty = {
        "wq_sharpe": 0.0, "wq_turnover": 0.0, "wq_returns": 0.0,
        "wq_fitness": 0.0, "wq_max_weight": 0.0, "wq_rating": "Needs Improvement",
        "margin_bps": 0.0, "submittable": False,
        "sub_universe": {}, "wq_is_tests": {},
    }
    if len(rebalance_dates) < 2:
        return empty

    rebal_set = set(rebalance_dates)
    dates = sorted(work_df["trade_date"].unique())

    daily_pnl: list[float] = []
    turnovers: list[float] = []
    current_weights = None
    pending_weights = None
    prev_weights = None
    max_weight = 0.0

    for date in dates:
        day_data = work_df[work_df["trade_date"] == date]

        if pending_weights is not None:
            prev_weights = current_weights
            current_weights = pending_weights
            pending_weights = None
            if prev_weights is not None:
                all_stocks = prev_weights.index.union(current_weights.index)
                old_w = prev_weights.reindex(all_stocks, fill_value=0)
                new_w = current_weights.reindex(all_stocks, fill_value=0)
                turnovers.append(float((new_w - old_w).abs().sum() / 2))

        if date in rebal_set:
            factor_vals = day_data.set_index("stock_code")["factor_value"].dropna()
            if len(factor_vals) < 2:
                continue
            weights = factor_vals - factor_vals.mean()
            total_abs = weights.abs().sum()
            if total_abs < 1e-12:
                continue
            weights = weights / total_abs
            max_weight = max(max_weight, float(weights.abs().max()))
            pending_weights = weights

        if current_weights is None:
            continue

        returns = day_data.set_index("stock_code")["daily_ret"]
        aligned = current_weights.index.intersection(returns.index)
        if len(aligned) == 0:
            continue
        daily_pnl.append(float((current_weights.loc[aligned] * returns.loc[aligned]).sum()))

    if len(daily_pnl) < 10:
        return empty

    pnl_series = np.array(daily_pnl)
    pnl_mean = float(np.mean(pnl_series))
    pnl_std = float(np.std(pnl_series, ddof=1))
    wq_sharpe = float(np.sqrt(trading_days_per_year) * pnl_mean / pnl_std) if pnl_std > 0 else 0.0

    n_days = len(daily_pnl)
    total_turnover = sum(turnovers)
    wq_turnover = total_turnover / n_days if n_days else 0.0
    wq_returns = pnl_mean * trading_days_per_year / 0.5
    effective_to = max(wq_turnover, 0.125)
    wq_fitness = (
        float(wq_sharpe * np.sqrt(abs(wq_returns) / effective_to))
        if wq_sharpe != 0 else 0.0
    )

    rating = _wq_rating(wq_fitness, gate=fitness_gate)
    daily_to_annualized = wq_turnover * trading_days_per_year
    margin_bps = wq_returns / daily_to_annualized * 10000 if daily_to_annualized > 0 else 0.0

    sub_uni = _sub_universe_sharpe(work_df, rebalance_dates, trading_days_per_year)
    is_tests = _is_tests(wq_sharpe, wq_fitness, wq_returns, wq_turnover, max_weight, sub_uni)
    submittable = all(t["pass"] for t in is_tests.values())

    return {
        "wq_sharpe": round(wq_sharpe, 4),
        "wq_turnover": round(wq_turnover, 4),
        "wq_returns": round(wq_returns, 4),
        "wq_fitness": round(wq_fitness, 4),
        "wq_max_weight": round(max_weight, 4),
        "wq_rating": rating,
        "passes_local_gate": wq_fitness >= fitness_gate,
        "fitness_gate": round(fitness_gate, 4),
        "margin_bps": round(margin_bps, 1),
        "submittable": submittable,
        "sub_universe": sub_uni,
        "wq_is_tests": is_tests,
    }


def _wq_rating(fi: float, *, gate: float = 0.5) -> str:
    """Bucket fitness into 5 quality levels.

    ``gate`` controls only the "Average" / "Needs Improvement" cut — the
    higher absolute milestones (Good ≥ 1.0, Excellent ≥ 1.5, Spectacular ≥ 2.5)
    are universal and unchanged. ``calibrate-local`` recommends the gate.
    """
    if fi >= 2.5:
        return "Spectacular"
    if fi >= 1.5:
        return "Excellent"
    if fi >= 1.0:
        return "Good"
    if fi >= gate:
        return "Average"
    return "Needs Improvement"


def _sub_universe_sharpe(work_df, rebalance_dates, trading_days_per_year=252, seed: int = 42):
    """Split stocks 50/50, compute dollar-neutral sharpe on each half."""
    np = _np()
    all_stocks = work_df["stock_code"].unique()
    if len(all_stocks) < 10:
        return {"sub_sharpe_min": 0.0, "threshold": 1.19, "pass": False}

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(all_stocks)
    mid = len(shuffled) // 2
    half_a = set(shuffled[:mid])
    half_b = set(shuffled[mid:])
    rebal_set = set(rebalance_dates)
    sharpes: list[float] = []

    for stock_set in [half_a, half_b]:
        sub_df = work_df[work_df["stock_code"].isin(stock_set)]
        dates = sorted(sub_df["trade_date"].unique())
        current_weights = None
        pending_weights = None
        pnls: list[float] = []
        for date in dates:
            day_data = sub_df[sub_df["trade_date"] == date]
            if pending_weights is not None:
                current_weights = pending_weights
                pending_weights = None
            if date in rebal_set:
                factor_vals = day_data.set_index("stock_code")["factor_value"].dropna()
                if len(factor_vals) < 2:
                    continue
                weights = factor_vals - factor_vals.mean()
                total_abs = weights.abs().sum()
                if total_abs < 1e-12:
                    continue
                pending_weights = weights / total_abs
            if current_weights is None:
                continue
            returns = day_data.set_index("stock_code")["daily_ret"]
            aligned = current_weights.index.intersection(returns.index)
            if len(aligned) == 0:
                continue
            pnls.append(float((current_weights.loc[aligned] * returns.loc[aligned]).sum()))
        if len(pnls) < 10:
            sharpes.append(0.0)
            continue
        arr = np.array(pnls)
        std = float(np.std(arr, ddof=1))
        sharpes.append(float(np.sqrt(trading_days_per_year) * np.mean(arr) / std) if std > 0 else 0.0)

    min_sh = min(sharpes) if sharpes else 0.0
    threshold = float(np.sqrt(trading_days_per_year) * max(0.065, 0.5 * 0.15))
    return {
        "sub_sharpe_a": round(sharpes[0], 3) if sharpes else 0.0,
        "sub_sharpe_b": round(sharpes[1], 3) if len(sharpes) > 1 else 0.0,
        "sub_sharpe_min": round(min_sh, 3),
        "threshold": round(threshold, 3),
        "pass": min_sh >= threshold,
    }


def _is_tests(sharpe: float, fitness: float, returns: float, turnover: float,
              max_weight: float, sub_uni: dict) -> dict:
    return {
        "sharpe": {"value": round(sharpe, 4), "threshold": 1.25,
                   "label": "Sharpe ≥ 1.25", "pass": sharpe >= 1.25},
        "fitness": {"value": round(fitness, 4), "threshold": 1.0,
                    "label": "Fitness ≥ 1.0", "pass": fitness >= 1.0},
        "returns": {"value": round(returns, 4), "threshold": 0.063,
                    "label": "Returns ≥ 6.3%", "pass": abs(returns) >= 0.063},
        "turnover_range": {"value": round(turnover, 4), "threshold_min": 0.01,
                           "threshold_max": 0.70,
                           "label": "Turnover ∈ [1%, 70%]",
                           "pass": 0.01 <= turnover <= 0.70},
        "weight": {"value": round(max_weight, 4), "threshold": 0.10,
                   "label": "Max Weight ≤ 10%", "pass": max_weight <= 0.10},
        "sub_universe": {"value": round(sub_uni.get("sub_sharpe_min", 0.0), 4),
                         "threshold": sub_uni.get("threshold", 1.19),
                         "label": f"Sub-Universe Sharpe ≥ {sub_uni.get('threshold', 1.19):.2f}",
                         "pass": sub_uni.get("pass", False)},
    }


# ── Expression evaluator (limited subset) ───────────────────────────────

def evaluate_expression(expr: str, ohlcv: Any) -> Any:
    """Evaluate a FASTEXPR over the cached OHLCV DataFrame.

    Returns a long-format DataFrame with columns
    [trade_date, stock_code, factor_value, daily_ret] suitable for wq_simulate.

    Supports a subset of operators (rank, ts_mean/rank/zscore/sum/max/delta/
    delay/decay_linear/corr, group_zscore/rank, hump, scale, abs/log/sqrt/sign,
    arithmetic, multi-line bindings). The full FASTEXPR set we whitelist is
    documented in operators.py — those not implemented here raise
    NotImplementedError so the agent falls back to remote simulate.
    """
    pd = _pd()
    np = _np()
    from .expr_parser import Parser, Node, tokenize

    if ohlcv is None or len(ohlcv) == 0:
        raise ValueError("empty OHLCV; run fetch-data first")

    # ohlcv: long-format index (date, ticker), columns open/high/low/close/volume
    # Convert to wide panels keyed by date × ticker for efficient ops
    panels = {col: ohlcv[col].unstack(level="ticker") for col in ohlcv.columns}
    panels["returns"] = panels["close"].pct_change(fill_method=None)
    # adv20/60/120/180: rolling mean of dollar volume (close × volume)
    dollar_vol = panels["close"] * panels["volume"]
    for win in (20, 60, 120, 180):
        panels[f"adv{win}"] = dollar_vol.rolling(win, min_periods=max(5, win // 4)).mean()
    # vwap proxy: same as close (we don't have intraday)
    panels["vwap"] = panels["close"]

    # Parse expression
    ast = Parser(tokenize(expr)).parse_program()

    bindings: dict[str, Any] = {}

    def _eval(node: Node) -> Any:
        if node.kind == "program":
            result = None
            for stmt in node.args:
                result = _eval(stmt)
            return result
        if node.kind == "assign":
            val = _eval(node.args[0])
            bindings[node.value] = val
            return val
        if node.kind == "number":
            return float(node.value)
        if node.kind == "ident":
            name = node.value
            if name in bindings:
                return bindings[name]
            if name in panels:
                return panels[name]
            if name in {"sector", "industry", "subindustry"}:
                # We don't have sector data wired here; return an empty group label panel
                # so group_* falls back to a single-group cross-section.
                return _make_constant_panel(panels["close"], 0)
            raise NotImplementedError(f"identifier '{name}' not available locally")
        if node.kind == "unaryop":
            inner = _eval(node.args[0])
            return -inner if node.value == "-" else inner
        if node.kind == "binop":
            left = _eval(node.args[0])
            right = _eval(node.args[1])
            op = node.value
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right.replace(0, np.nan) if hasattr(right, "replace") else left / right
            # comparison (returns bool panel)
            if op == ">":
                return left > right
            if op == "<":
                return left < right
            if op == ">=":
                return left >= right
            if op == "<=":
                return left <= right
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            raise NotImplementedError(f"binop {op}")
        if node.kind == "call":
            fname = node.value
            args = [_eval(a) for a in node.args]
            return _apply(fname, args)
        raise NotImplementedError(f"node kind {node.kind}")

    result_panel = _eval(ast)
    if not hasattr(result_panel, "stack"):
        raise ValueError(f"expression did not yield a panel; got {type(result_panel)}")

    # Convert wide → long
    factor_long = result_panel.stack(future_stack=True).rename("factor_value").reset_index()
    factor_long.columns = ["trade_date", "stock_code", "factor_value"]
    ret_long = panels["returns"].stack(future_stack=True).rename("daily_ret").reset_index()
    ret_long.columns = ["trade_date", "stock_code", "daily_ret"]
    work = factor_long.merge(ret_long, on=["trade_date", "stock_code"], how="inner")
    work = work.dropna(subset=["factor_value", "daily_ret"])
    return work


def _make_constant_panel(template: Any, val: Any) -> Any:
    """Make a panel like template but filled with `val`."""
    pd = _pd()
    return pd.DataFrame(val, index=template.index, columns=template.columns)


def _apply(fname: str, args: list[Any]) -> Any:
    """Apply a FASTEXPR operator to args (panels or scalars)."""
    np = _np()
    pd = _pd()

    def _ax_rolling(panel, win: int, fn: str):
        roll = panel.rolling(int(win), min_periods=max(2, int(win) // 4))
        if fn == "mean":
            return roll.mean()
        if fn == "sum":
            return roll.sum()
        if fn == "max":
            return roll.max()
        if fn == "rank":
            return roll.rank(pct=True)
        if fn == "zscore":
            return (panel - roll.mean()) / roll.std(ddof=0).replace(0, np.nan)
        if fn == "decay_linear":
            weights = np.arange(1, int(win) + 1, dtype=float)
            weights /= weights.sum()
            return panel.rolling(int(win), min_periods=max(2, int(win) // 4)).apply(
                lambda x: float(np.dot(x, weights[-len(x):])) if len(x) > 0 else np.nan,
                raw=True,
            )
        raise NotImplementedError(f"rolling {fn}")

    if fname in ("ts_mean", "ts_sum", "ts_max", "ts_rank", "ts_zscore", "ts_decay_linear"):
        return _ax_rolling(args[0], int(args[1]), fname.replace("ts_", ""))
    if fname == "ts_delta":
        return args[0] - args[0].shift(int(args[1]))
    if fname == "ts_delay":
        return args[0].shift(int(args[1]))
    if fname == "ts_corr":
        x, y, win = args[0], args[1], int(args[2])
        return x.rolling(win, min_periods=max(2, win // 4)).corr(y)
    if fname == "rank":
        return args[0].rank(axis=1, pct=True)
    if fname == "scale":
        x = args[0]
        ax_max = x.abs().sum(axis=1).replace(0, np.nan)
        return x.div(ax_max, axis=0)
    if fname in ("group_rank", "group_zscore", "group_neutralize", "group_mean"):
        # Without sector data we treat all stocks as one group → fall back
        return args[0].rank(axis=1, pct=True) if fname == "group_rank" else args[0]
    if fname == "abs":
        return args[0].abs()
    if fname == "log":
        return np.log(args[0].clip(lower=1e-12))
    if fname == "sqrt":
        return np.sqrt(args[0].clip(lower=0))
    if fname == "sign":
        return np.sign(args[0])
    if fname == "power":
        return np.sign(args[0]) * (args[0].abs() ** args[1])  # protect negatives
    if fname == "signed_power":
        return np.sign(args[0]) * (args[0].abs() ** args[1])
    if fname == "if_else":
        return args[1].where(args[0], args[2])
    if fname == "hump":
        # Approximation: clip per-period change in factor
        threshold = float(args[1]) if len(args) > 1 else 0.01
        x = args[0]
        delta = x - x.shift(1)
        delta_clipped = delta.where(delta.abs() >= threshold, 0)
        return x.shift(1).fillna(0) + delta_clipped.fillna(x)
    if fname == "min":
        return args[0].combine(args[1], np.minimum) if len(args) == 2 else args[0]
    if fname == "max":
        return args[0].combine(args[1], np.maximum) if len(args) == 2 else args[0]
    raise NotImplementedError(f"operator '{fname}' not implemented locally")


# ── High-level wrapper ──────────────────────────────────────────────────

@dataclass
class LocalSimResult:
    expr: str
    wq_sharpe: float
    wq_fitness: float
    wq_turnover: float
    wq_returns: float
    submittable: bool
    rating: str
    raw: dict[str, Any]


def simulate_expression_locally(
    expr: str,
    ohlcv: Optional[Any] = None,
    *,
    rebalance_freq: int = 5,
    tag: Optional[str] = None,
    fitness_gate: Optional[float] = None,
) -> LocalSimResult:
    """Evaluate ``expr`` against cached OHLCV and run wq_simulate.

    ``rebalance_freq=5`` means rebalance every 5 trading days (weekly).

    The "passes_local_gate" bit in the returned ``raw`` dict uses, in order:
      1. the explicit ``fitness_gate`` kwarg (caller override)
      2. the calibration file at ``calibration/{tag}/threshold.json`` if
         ``tag`` is given and a calibration has been run
      3. the legacy 0.5 default (back-compat)
    """
    pd = _pd()
    if ohlcv is None:
        from .data_loader import load_cached_ohlcv
        ohlcv = load_cached_ohlcv()
    if ohlcv is None or len(ohlcv) == 0:
        raise RuntimeError("no cached OHLCV; run `wq_brain fetch-data` first")

    if fitness_gate is None:
        if tag:
            from .calibration import load_calibrated_threshold
            fitness_gate = load_calibrated_threshold(tag, default=0.5)
        else:
            fitness_gate = 0.5

    work = evaluate_expression(expr, ohlcv)
    dates = sorted(pd.to_datetime(work["trade_date"]).unique())
    rebalance_dates = list(dates[::rebalance_freq])
    raw = wq_simulate(work, rebalance_dates, fitness_gate=fitness_gate)
    return LocalSimResult(
        expr=expr,
        wq_sharpe=raw["wq_sharpe"],
        wq_fitness=raw["wq_fitness"],
        wq_turnover=raw["wq_turnover"],
        wq_returns=raw["wq_returns"],
        submittable=raw["submittable"],
        rating=raw["wq_rating"],
        raw=raw,
    )
