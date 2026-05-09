"""Compute time-period performance analysis from a LEAN backtest result.

Produces lean_analysis.json with:
  - equity_curve          raw equity points {date, equity, drawdown, return_pct}
  - monthly_returns       per-month aggregates
  - quarterly_returns     per-quarter aggregates
  - drawdown_episodes     top-K drawdown periods with recovery
  - pair_contribution     per-symbol realized P&L, win_rate, Herfindahl index
  - regime_segments       consecutive loss/gain streaks, volatility regimes
  - vs_rank_comparison    deviation vs rank backtest monthly curve (if available)
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .lean_bridge import (  # private helpers are OK within the same package
    _as_float,
    _chart_series_values,
    _filled_events_from_orders,
    _load_lean_payload,
    _load_lean_related_payloads,
)

__all__ = ["compute_lean_analysis"]


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _unix_to_dt(ts: float) -> datetime:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


def _equity_series_to_df(
    equity_series: Sequence[Tuple[float, float]]
) -> pd.DataFrame:
    """Convert [(unix_ts, equity), ...] to a DataFrame with a tz-naive DatetimeIndex."""
    if not equity_series:
        return pd.DataFrame(columns=["equity"])
    rows = [{"dt": _unix_to_dt(ts), "equity": float(v)} for ts, v in equity_series]
    df = pd.DataFrame(rows).set_index("dt")
    df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["equity"] = df["equity"].clip(lower=0.0)
    # Normalize to start=1.0 if >1 (LEAN uses dollar values)
    start = df["equity"].iloc[0] if not df.empty else 1.0
    if start > 2.0:
        df["equity"] = df["equity"] / float(start)
    # Running peak + drawdown
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"].clip(lower=1e-12)
    df["return_pct"] = df["equity"].pct_change() * 100.0
    return df


# ---------------------------------------------------------------------------
# Monthly / quarterly aggregates
# ---------------------------------------------------------------------------

def _period_stats(group: pd.DataFrame) -> Dict[str, Any]:
    start_eq = float(group["equity"].iloc[0])
    end_eq = float(group["equity"].iloc[-1])
    ret = (end_eq - start_eq) / max(start_eq, 1e-12) * 100.0
    max_dd = float(group["drawdown"].min() * 100.0)
    return {
        "return_pct": round(ret, 4),
        "start_equity": round(start_eq, 6),
        "end_equity": round(end_eq, 6),
        "max_dd_in_period": round(max_dd, 4),
        "periods": int(len(group)),
    }


def _monthly_returns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    rows = []
    for period, grp in df.groupby(pd.Grouper(freq="ME")):
        if grp.empty:
            continue
        stats = _period_stats(grp)
        stats["period"] = str(period)[:7]  # "YYYY-MM"
        rows.append(stats)
    return rows


def _quarterly_returns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    rows = []
    for period, grp in df.groupby(pd.Grouper(freq="QE")):
        if grp.empty:
            continue
        stats = _period_stats(grp)
        stats["period"] = f"{period.year}-Q{period.quarter}"
        rows.append(stats)
    return rows


# ---------------------------------------------------------------------------
# Drawdown episodes
# ---------------------------------------------------------------------------

def _drawdown_episodes(df: pd.DataFrame, top_k: int = 3) -> List[Dict[str, Any]]:
    """Identify distinct drawdown episodes: peak → trough → recovery."""
    if df.empty or len(df) < 2:
        return []

    equity = df["equity"].values
    dates = df.index
    n = len(equity)

    episodes: List[Dict[str, Any]] = []
    peak_idx = 0
    peak_val = equity[0]
    in_dd = False
    trough_idx = 0
    trough_val = equity[0]

    for i in range(1, n):
        v = equity[i]
        if v >= peak_val:
            if in_dd:
                # recovery complete
                episodes.append(
                    _build_episode(dates, equity, peak_idx, trough_idx, i, recovered=True)
                )
                in_dd = False
            peak_val = v
            peak_idx = i
            trough_idx = i
            trough_val = v
        else:
            if not in_dd:
                in_dd = True
                trough_idx = i
                trough_val = v
            elif v < trough_val:
                trough_val = v
                trough_idx = i

    if in_dd:
        episodes.append(
            _build_episode(dates, equity, peak_idx, trough_idx, n - 1, recovered=False)
        )

    episodes.sort(key=lambda e: e["depth_pct"])  # most negative first
    return episodes[:top_k]


def _build_episode(
    dates: pd.DatetimeIndex,
    equity: Any,
    peak_idx: int,
    trough_idx: int,
    end_idx: int,
    *,
    recovered: bool,
) -> Dict[str, Any]:
    peak_val = float(equity[peak_idx])
    trough_val = float(equity[trough_idx])
    depth = (trough_val - peak_val) / max(peak_val, 1e-12) * 100.0
    start_date = dates[peak_idx]
    trough_date = dates[trough_idx]
    end_date = dates[end_idx]
    duration_days = max(0, (trough_date - start_date).days)
    recovery_days = max(0, (end_date - trough_date).days) if recovered else -1
    return {
        "start": str(start_date.date()),
        "trough": str(trough_date.date()),
        "end": str(end_date.date()) if recovered else None,
        "depth_pct": round(depth, 4),
        "duration_days": duration_days,
        "recovery_days": recovery_days,
        "recovered": recovered,
    }


# ---------------------------------------------------------------------------
# Per-pair P&L from fills (simplified avg-cost method)
# ---------------------------------------------------------------------------

def _pair_pnl_from_fills(
    fills: List[Dict[str, Any]],
    equity_series: Sequence[Tuple[float, float]],
    start_equity: float = 100000.0,
) -> Dict[str, Any]:
    """Compute per-symbol realized P&L using average cost tracking."""
    if not fills:
        return {"pairs": [], "herfindahl_index": None, "top_winners": [], "top_losers": []}

    # symbol → {qty, avg_cost, realized_pnl, trades, wins}
    positions: Dict[str, Dict[str, Any]] = {}

    def get_pos(sym: str) -> Dict[str, Any]:
        if sym not in positions:
            positions[sym] = {"qty": 0.0, "avg_cost": 0.0, "realized_pnl": 0.0, "trades": 0, "wins": 0}
        return positions[sym]

    for fill in sorted(fills, key=lambda f: f.get("timestamp", 0)):
        sym = str(fill.get("symbol") or "")
        qty = float(fill.get("quantity") or 0.0)
        price = float(fill.get("price") or 0.0)
        if abs(qty) < 1e-12 or not sym:
            continue
        pos = get_pos(sym)
        existing_qty = pos["qty"]

        if existing_qty == 0.0:
            # Opening new position
            pos["qty"] = qty
            pos["avg_cost"] = price
        elif (existing_qty > 0) == (qty > 0):
            # Adding to position (same side)
            total_qty = existing_qty + qty
            pos["avg_cost"] = (existing_qty * pos["avg_cost"] + qty * price) / total_qty
            pos["qty"] = total_qty
        else:
            # Reducing or flipping position
            close_qty = min(abs(qty), abs(existing_qty))
            sign = 1.0 if existing_qty > 0 else -1.0
            pnl = sign * close_qty * (price - pos["avg_cost"])
            pos["realized_pnl"] += pnl
            pos["trades"] += 1
            if pnl > 0:
                pos["wins"] += 1
            remaining = existing_qty + qty
            if abs(remaining) < 1e-12:
                pos["qty"] = 0.0
                pos["avg_cost"] = 0.0
            else:
                pos["qty"] = remaining
                if (remaining > 0) != (existing_qty > 0):
                    pos["avg_cost"] = price  # flipped

    total_realized = sum(p["realized_pnl"] for p in positions.values())
    result_pairs = []
    for sym, pos in positions.items():
        pnl = pos["realized_pnl"]
        pnl_pct = pnl / max(abs(total_realized), 1e-12) * 100.0 if total_realized != 0 else 0.0
        trades = pos["trades"]
        win_rate = pos["wins"] / max(trades, 1) * 100.0
        result_pairs.append({
            "pair": sym,
            "realized_pnl_usd": round(pnl, 4),
            "pnl_pct_of_total": round(pnl_pct, 4),
            "pnl_pct_of_start_equity": round(pnl / max(start_equity, 1e-12) * 100.0, 4),
            "trades": trades,
            "win_rate": round(win_rate, 2),
        })

    result_pairs.sort(key=lambda r: r["realized_pnl_usd"], reverse=True)

    # Herfindahl index on abs(pnl) → concentration
    abs_pnls = [abs(p["realized_pnl_usd"]) for p in result_pairs]
    total_abs = sum(abs_pnls)
    if total_abs > 1e-12:
        herfindahl = sum((v / total_abs) ** 2 for v in abs_pnls)
    else:
        herfindahl = None

    return {
        "pairs": result_pairs,
        "herfindahl_index": round(herfindahl, 4) if herfindahl is not None else None,
        "top_winners": [p["pair"] for p in result_pairs[:5] if p["realized_pnl_usd"] > 0],
        "top_losers": [p["pair"] for p in reversed(result_pairs) if p["realized_pnl_usd"] < 0][:5],
        "total_realized_pnl_usd": round(total_realized, 4),
    }


# ---------------------------------------------------------------------------
# Regime / streak analysis
# ---------------------------------------------------------------------------

def _regime_segments(monthly: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute streak, volatility, regime stats from monthly_returns list."""
    if not monthly:
        return {}

    rets = [m["return_pct"] for m in monthly]

    # Consecutive loss months
    max_loss_streak = 0
    cur_loss = 0
    max_gain_streak = 0
    cur_gain = 0
    for r in rets:
        if r < 0:
            cur_loss += 1
            cur_gain = 0
        else:
            cur_gain += 1
            cur_loss = 0
        max_loss_streak = max(max_loss_streak, cur_loss)
        max_gain_streak = max(max_gain_streak, cur_gain)

    worst_month = min(monthly, key=lambda m: m["return_pct"])
    best_month = max(monthly, key=lambda m: m["return_pct"])
    positive_months = sum(1 for r in rets if r > 0)
    negative_months = sum(1 for r in rets if r < 0)
    positive_pct = positive_months / max(len(rets), 1) * 100.0

    # Monthly volatility
    if len(rets) > 1:
        import statistics
        monthly_vol = statistics.stdev(rets)
    else:
        monthly_vol = 0.0

    # Max consecutive loss months starting point
    loss_streak_months = []
    cur = []
    for m in monthly:
        if m["return_pct"] < 0:
            cur.append(m["period"])
        else:
            if len(cur) == max_loss_streak and max_loss_streak > 0:
                loss_streak_months = cur[:]
            cur = []
    if cur and len(cur) == max_loss_streak:
        loss_streak_months = cur[:]

    return {
        "total_months": len(monthly),
        "positive_months": positive_months,
        "negative_months": negative_months,
        "positive_month_pct": round(positive_pct, 2),
        "consecutive_loss_months": max_loss_streak,
        "consecutive_gain_months": max_gain_streak,
        "worst_month": {"period": worst_month["period"], "return_pct": worst_month["return_pct"]},
        "best_month": {"period": best_month["period"], "return_pct": best_month["return_pct"]},
        "monthly_volatility": round(monthly_vol, 4),
        "loss_streak_periods": loss_streak_months,
    }


# ---------------------------------------------------------------------------
# Rank vs LEAN comparison
# ---------------------------------------------------------------------------

def _vs_rank_comparison(
    lean_monthly: List[Dict[str, Any]],
    rank_curve: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Compute divergence between LEAN and rank backtest at monthly granularity."""
    if not rank_curve:
        return {"available": False, "divergence_score": None}

    try:
        rank_df = pd.DataFrame(rank_curve)
        if "date" not in rank_df.columns or "equity" not in rank_df.columns:
            return {"available": False, "reason": "rank_curve missing date/equity columns"}
        rank_df["date"] = pd.to_datetime(rank_df["date"])
        rank_df = rank_df.set_index("date").sort_index()
        rank_df["equity"] = pd.to_numeric(rank_df["equity"], errors="coerce")
        rank_monthly_rets: Dict[str, float] = {}
        for period, grp in rank_df.groupby(pd.Grouper(freq="ME")):
            if grp.empty:
                continue
            r = (float(grp["equity"].iloc[-1]) - float(grp["equity"].iloc[0])) / max(float(grp["equity"].iloc[0]), 1e-12) * 100.0
            rank_monthly_rets[str(period)[:7]] = r

        diffs = []
        for m in lean_monthly:
            key = m["period"]
            if key in rank_monthly_rets:
                diffs.append(m["return_pct"] - rank_monthly_rets[key])

        if not diffs:
            return {"available": False, "reason": "no overlapping periods"}

        import statistics
        abs_diffs = [abs(d) for d in diffs]
        divergence_score = min(100.0, sum(abs_diffs) / max(len(abs_diffs), 1) * 2)  # 0-100 heuristic
        worst_divergence_idx = abs_diffs.index(max(abs_diffs))
        worst_month = lean_monthly[worst_divergence_idx]["period"] if worst_divergence_idx < len(lean_monthly) else None

        return {
            "available": True,
            "divergence_score": round(divergence_score, 2),
            "mean_abs_diff_pct": round(statistics.mean(abs_diffs), 4),
            "worst_divergence_month": worst_month,
            "worst_divergence_pct": round(max(abs_diffs), 4),
            "compared_months": len(diffs),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_lean_analysis(
    *,
    lean_result: Path | str,
    output: Path | str,
    timeframe: str = "1h",
    rank_curve: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute rich time-period analysis from a LEAN backtest result.

    Args:
        lean_result: Path to the LEAN result JSON (or directory containing it).
        output: Destination path for lean_analysis.json.
        timeframe: OHLCV timeframe string (for context only).
        rank_curve: Optional rank backtest equity curve (list of {date, equity} dicts)
                    used to compute vs_rank_comparison divergence.

    Returns:
        The analysis dict that was written to *output*.
    """
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _load_lean_payload(lean_result)
    related = _load_lean_related_payloads(payload)
    summary_payload = related["summary"]
    full_payload = related["full"]
    order_events = related["order_events"]

    # Pull equity curve from LEAN charts
    raw_equity = _chart_series_values(summary_payload, "Strategy Equity", "Equity")
    if not raw_equity:
        raw_equity = _chart_series_values(full_payload or {}, "Strategy Equity", "Equity")

    eq_df = _equity_series_to_df(raw_equity)

    # Sampled curve for JSON (max 300 points)
    if len(eq_df) > 300:
        step = max(1, len(eq_df) // 300)
        sampled_df = eq_df.iloc[::step]
    else:
        sampled_df = eq_df

    equity_curve = [
        {
            "date": str(idx.date()),
            "equity": round(float(row["equity"]), 6),
            "drawdown": round(float(row["drawdown"]), 6),
            "return_pct": round(float(row["return_pct"]) if math.isfinite(float(row["return_pct"])) else 0.0, 4),
        }
        for idx, row in sampled_df.iterrows()
    ]

    # Time-period aggregates
    monthly = _monthly_returns(eq_df)
    quarterly = _quarterly_returns(eq_df)

    # Drawdown episodes
    dd_episodes = _drawdown_episodes(eq_df, top_k=3)

    # Per-pair P&L from fills
    fills = []
    if order_events:
        from .lean_bridge import _lean_execution_stats  # noqa: F401
        # Reuse filled events extraction
        fills_raw = _filled_events_from_orders(full_payload or {})
        if not fills_raw:
            fills_raw = [e for e in order_events if isinstance(e, dict)]
        # Normalize fills to expected format
        for ev in fills_raw:
            qty = _as_float(ev.get("fillQuantity") or ev.get("FillQuantity") or ev.get("quantity") or ev.get("Quantity"))
            price = _as_float(ev.get("fillPrice") or ev.get("FillPrice") or ev.get("price") or ev.get("Price"))
            sym_raw = ev.get("symbolValue") or ev.get("symbol") or ev.get("Symbol")
            if isinstance(sym_raw, dict):
                sym_raw = sym_raw.get("value") or sym_raw.get("permtick") or ""
            sym = str(sym_raw or "").split(".", 1)[0]
            ts_raw = ev.get("time") or ev.get("lastFillTime") or ev.get("Time")
            ts = None
            if ts_raw is not None:
                f = _as_float(ts_raw)
                if f is not None:
                    ts = f
                else:
                    try:
                        ts = float(pd.Timestamp(ts_raw).timestamp())
                    except Exception:
                        pass
            if qty and price and sym and ts:
                fills.append({"symbol": sym, "quantity": float(qty), "price": float(price), "timestamp": float(ts)})

    stats_raw = summary_payload.get("statistics") or summary_payload.get("Statistics") or summary_payload
    start_equity_raw = _as_float(
        (stats_raw.get("start_equity") if isinstance(stats_raw, dict) else None)
        or (stats_raw.get("Start Equity") if isinstance(stats_raw, dict) else None)
        or 100000.0
    )
    start_equity = float(start_equity_raw or 100000.0)

    pair_contribution = _pair_pnl_from_fills(fills, raw_equity, start_equity=start_equity)

    # Regime segments
    regime = _regime_segments(monthly)

    # vs-rank comparison
    vs_rank = _vs_rank_comparison(monthly, rank_curve)

    result: Dict[str, Any] = {
        "timeframe": timeframe,
        "equity_curve": equity_curve,
        "monthly_returns": monthly,
        "quarterly_returns": quarterly,
        "drawdown_episodes": dd_episodes,
        "pair_contribution": pair_contribution,
        "regime_segments": regime,
        "vs_rank_comparison": vs_rank,
    }

    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
