from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_add(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None and b is None:
        return None
    return float(a or 0.0) + float(b or 0.0)


def _quantiles(values: Iterable[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None]  # type: ignore[arg-type]
    if not vals:
        return {"p0": None, "p5": None, "p25": None, "p50": None, "p75": None, "p95": None, "p100": None}
    try:
        import numpy as np  # noqa: PLC0415

        arr = np.asarray(vals, dtype=float)
        qs = np.percentile(arr, [0, 5, 25, 50, 75, 95, 100]).tolist()
        return {
            "p0": float(qs[0]),
            "p5": float(qs[1]),
            "p25": float(qs[2]),
            "p50": float(qs[3]),
            "p75": float(qs[4]),
            "p95": float(qs[5]),
            "p100": float(qs[6]),
        }
    except Exception:
        vals.sort()
        n = len(vals)
        return {
            "p0": float(vals[0]),
            "p5": float(vals[int(0.05 * (n - 1))]),
            "p25": float(vals[int(0.25 * (n - 1))]),
            "p50": float(vals[int(0.50 * (n - 1))]),
            "p75": float(vals[int(0.75 * (n - 1))]),
            "p95": float(vals[int(0.95 * (n - 1))]),
            "p100": float(vals[-1]),
        }


def summarize_trades(trades: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Compute v1 TCA metrics from strategy trades.

    Returns: (summary, distributions, per_pair_rows)
    """
    n = int(len(trades))
    profit_abs_vals: list[float] = []
    profit_ratio_vals: list[float] = []
    duration_vals: list[float] = []
    fee_vals: list[float] = []

    wins = 0
    profit_abs_total: Optional[float] = None
    profit_ratio_total: Optional[float] = None
    fees_total: Optional[float] = None

    per_pair: dict[str, dict[str, Any]] = defaultdict(lambda: {"trades": 0, "wins": 0, "profit_abs_total": 0.0, "profit_ratio_total": 0.0, "fees_total": 0.0})

    for t in trades:
        pair = str(t.get("pair") or "").strip() or "UNKNOWN"
        pa = _as_float(t.get("profit_abs"))
        pr = _as_float(t.get("profit_ratio"))
        dur = _as_float(t.get("trade_duration"))
        fee_open = _as_float(t.get("fee_open"))
        fee_close = _as_float(t.get("fee_close"))
        funding_fees = _as_float(t.get("funding_fees"))
        fees = (fee_open or 0.0) + (fee_close or 0.0) + (funding_fees or 0.0)

        if pa is not None:
            profit_abs_vals.append(pa)
            profit_abs_total = _safe_add(profit_abs_total, pa)
        if pr is not None:
            profit_ratio_vals.append(pr)
            profit_ratio_total = _safe_add(profit_ratio_total, pr)
        if dur is not None:
            duration_vals.append(dur)
        fee_vals.append(float(fees))
        fees_total = _safe_add(fees_total, fees)

        if pa is not None and pa > 0:
            wins += 1

        row = per_pair[pair]
        row["trades"] += 1
        if pa is not None and pa > 0:
            row["wins"] += 1
        row["profit_abs_total"] += float(pa or 0.0)
        row["profit_ratio_total"] += float(pr or 0.0)
        row["fees_total"] += float(fees or 0.0)

    win_rate = (wins / n) if n > 0 else None
    summary = {
        "trades": n,
        "win_rate": float(win_rate) if win_rate is not None else None,
        "profit_abs_total": float(profit_abs_total) if profit_abs_total is not None else None,
        "profit_ratio_total": float(profit_ratio_total) if profit_ratio_total is not None else None,
        "fees_total": float(fees_total) if fees_total is not None else None,
        "slippage_bps_est": None,
    }
    distributions = {
        "profit_abs": _quantiles(profit_abs_vals),
        "profit_ratio": _quantiles(profit_ratio_vals),
        "trade_duration_min": _quantiles(duration_vals),
        "fees": _quantiles(fee_vals),
    }
    per_pair_rows = []
    for pair, row in sorted(per_pair.items(), key=lambda kv: (-kv[1]["trades"], kv[0])):
        trades_n = int(row["trades"])
        w = int(row["wins"])
        per_pair_rows.append(
            {
                "pair": pair,
                "trades": trades_n,
                "win_rate": (w / trades_n) if trades_n > 0 else None,
                "profit_abs_total": float(row["profit_abs_total"]),
                "profit_ratio_total": float(row["profit_ratio_total"]),
                "fees_total": float(row["fees_total"]),
            }
        )
    return summary, distributions, per_pair_rows


__all__ = ["summarize_trades"]

