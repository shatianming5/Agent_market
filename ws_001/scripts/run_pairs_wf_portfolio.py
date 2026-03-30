#!/usr/bin/env python3
"""Run research pipeline for cointegrated pairs.

Steps:
1) Scan cointegrated pairs using pairs_engine.scan_pairs()
2) Backtest the top 3 pairs with walk-forward validation
3) Build an optimal portfolio from the validated pairs
4) Write a summary to results/agent_report.json
"""

from __future__ import annotations

import sys
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

# When executed as `python scripts/...`, add project root so `import pairs_engine` works.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class WalkForwardStats:
    n_windows: int
    profits: List[float]
    sharpes: List[float]
    trades: List[int]
    win_rates: List[float]
    max_dd_pcts: List[float]
    mean_profit: float
    mean_sharpe: float
    std_profit: float
    std_sharpe: float
    pct_profitable_windows: float
    total_trades: int
    passed: bool
    verdict: str


def _ensure_results_dir() -> Path:
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _walk_forward_pair(
    asset_a: str,
    asset_b: str,
    *,
    exchange: str = "gate",
    timeframe: str = "1h",
    n_windows: int = 4,
    lookback: int = 80,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    maker_fee_bps: float = 1.0,
) -> Tuple[WalkForwardStats, List[Dict[str, Any]]]:
    """Simple walk-forward validation for PairsEngine backtest.

    We slice the full history into n_windows equal-length test windows.
    For each window we include a warm-up region so rolling stats are valid.
    """
    from pairs_engine import PairsEngine

    pe = PairsEngine(asset_a, asset_b, exchange=exchange, timeframe=timeframe)
    full_df = pe.load_data()
    n = len(full_df)
    if n < (lookback + 50) * (n_windows + 1):
        # Not enough data for robust WF, still attempt with whatever windows fit.
        n_windows = max(1, min(n_windows, max(1, n // max(200, lookback + 50))))

    window_size = n // (n_windows + 1)
    window_results: List[Dict[str, Any]] = []
    profits: List[float] = []
    sharpes: List[float] = []
    trades: List[int] = []
    win_rates: List[float] = []
    max_dd_pcts: List[float] = []

    for wi in range(n_windows):
        test_start = (wi + 1) * window_size
        test_end = min(test_start + window_size, n)
        warm_start = max(0, test_start - lookback - 50)

        pe_w = PairsEngine(asset_a, asset_b, exchange=exchange, timeframe=timeframe)
        pe_w._df = full_df.iloc[warm_start:test_end].reset_index(drop=True)
        sig = pe_w.generate_signals(lookback=lookback, entry_z=entry_z, exit_z=exit_z)
        bt = pe_w.backtest(sig, maker_fee_bps=maker_fee_bps)

        # Focus on out-of-sample segment only for reporting.
        test_begin_local = test_start - warm_start
        test_begin_local = max(0, min(test_begin_local, len(sig) - 1))
        if test_begin_local > 0:
            sig_test = sig.iloc[test_begin_local:].reset_index(drop=True)
            bt_test = pe_w.backtest(sig_test, maker_fee_bps=maker_fee_bps)
        else:
            bt_test = bt

        wr = {
            "window": wi + 1,
            "warm_start": int(warm_start),
            "test_range_idx": [int(test_start), int(test_end)],
            "profit_pct": float(bt_test.profit_pct),
            "sharpe": float(bt_test.sharpe),
            "trades": int(bt_test.trades),
            "win_rate": float(bt_test.win_rate),
            "max_dd_pct": float(bt_test.max_dd_pct),
        }
        window_results.append(wr)
        profits.append(wr["profit_pct"])
        sharpes.append(wr["sharpe"])
        trades.append(wr["trades"])
        win_rates.append(wr["win_rate"])
        max_dd_pcts.append(wr["max_dd_pct"])

    profits_arr = np.array(profits, dtype=float)
    sharpes_arr = np.array(sharpes, dtype=float)
    pct_profitable = float(np.mean(profits_arr > 0)) if len(profits_arr) else 0.0
    total_trades = int(np.sum(trades))

    # Pass rule: at least 50% profitable windows and mean Sharpe > 0.
    mean_sharpe = float(np.nanmean(sharpes_arr)) if len(sharpes_arr) else 0.0
    passed = bool(pct_profitable >= 0.5 and mean_sharpe >= 0.0 and total_trades > 0)
    verdict = (
        (f"PASS" if passed else "FAIL")
        + f": mean_sharpe={mean_sharpe:.3f}, profitable_windows={pct_profitable:.0%}, total_trades={total_trades}"
    )

    stats = WalkForwardStats(
        n_windows=int(n_windows),
        profits=[float(x) for x in profits],
        sharpes=[float(x) for x in sharpes],
        trades=[int(x) for x in trades],
        win_rates=[float(x) for x in win_rates],
        max_dd_pcts=[float(x) for x in max_dd_pcts],
        mean_profit=float(np.nanmean(profits_arr)) if len(profits_arr) else 0.0,
        mean_sharpe=mean_sharpe,
        std_profit=float(np.nanstd(profits_arr)) if len(profits_arr) else 0.0,
        std_sharpe=float(np.nanstd(sharpes_arr)) if len(sharpes_arr) else 0.0,
        pct_profitable_windows=pct_profitable,
        total_trades=total_trades,
        passed=passed,
        verdict=verdict,
    )

    return stats, window_results


def _portfolio_optimize(
    validated: List[Dict[str, Any]],
    *,
    w_step: float = 0.05,
) -> Dict[str, Any]:
    """Optimize weights over validated pairs.

    Objective: maximize portfolio "window Sharpe" using per-window profit series.
    Constraint: weights >= 0, sum(weights)=1.
    """
    if not validated:
        return {
            "ok": False,
            "reason": "no validated pairs",
        }

    n = len(validated)
    profits_mat = []
    for v in validated:
        profits_mat.append(v["walk_forward"]["profits"])
    profits_mat = np.array(profits_mat, dtype=float)  # [n_pairs, n_windows]
    n_windows = profits_mat.shape[1]
    if n_windows < 2:
        return {
            "ok": False,
            "reason": "insufficient windows for optimization",
        }

    steps = int(round(1.0 / w_step))
    best = {
        "sharpe": -1e9,
        "weights": None,
        "mean_profit": None,
        "std_profit": None,
        "profits": None,
    }

    # Simple grid search for up to 3 assets (as required by the task).
    # For n<3 we still use the same approach.
    def eval_w(w: np.ndarray) -> None:
        p = w @ profits_mat
        mu = float(np.mean(p))
        sd = float(np.std(p) + 1e-12)
        sharpe = mu / sd
        if sharpe > best["sharpe"]:
            best.update(
                {
                    "sharpe": float(sharpe),
                    "weights": [float(x) for x in w],
                    "mean_profit": mu,
                    "std_profit": sd,
                    "profits": [float(x) for x in p],
                }
            )

    if n == 1:
        eval_w(np.array([1.0]))
    elif n == 2:
        for i in range(steps + 1):
            w1 = i * w_step
            w2 = 1.0 - w1
            if w2 < -1e-9:
                continue
            eval_w(np.array([w1, w2]))
    else:
        # n==3
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                w1 = i * w_step
                w2 = j * w_step
                w3 = 1.0 - w1 - w2
                if w3 < -1e-9:
                    continue
                eval_w(np.array([w1, w2, w3]))

    name_map = [f"{v['pair_a']}|{v['pair_b']}" for v in validated]
    weights = best["weights"]
    alloc = {name_map[i]: weights[i] for i in range(len(weights))} if weights else {}

    return {
        "ok": True,
        "objective": "maximize window-profit sharpe (mean/std)",
        "w_step": w_step,
        "portfolio_window_profits": best["profits"],
        "portfolio_mean_profit": best["mean_profit"],
        "portfolio_std_profit": best["std_profit"],
        "portfolio_sharpe": best["sharpe"],
        "weights": alloc,
    }


def main() -> None:
    from pairs_engine import scan_pairs

    out_dir = _ensure_results_dir()

    # 1) scan
    scanned = scan_pairs(exchange="gate", timeframe="1h", min_correlation=0.8)
    scanned_coint = [p for p in scanned if p.get("cointegrated")]
    top3 = scanned_coint[:3]

    # 2) walk-forward backtest
    validated: List[Dict[str, Any]] = []
    for p in top3:
        wf_stats, wf_windows = _walk_forward_pair(
            p["pair_a"],
            p["pair_b"],
            exchange="gate",
            timeframe="1h",
            n_windows=4,
            lookback=80,
            entry_z=2.0,
            exit_z=0.5,
            maker_fee_bps=1.0,
        )

        record = {
            **p,
            "walk_forward": asdict(wf_stats),
            "walk_forward_windows": wf_windows,
        }
        if wf_stats.passed:
            validated.append(record)

    # 3) portfolio optimization
    portfolio = _portfolio_optimize(validated, w_step=0.05)

    # 4) report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exchange": "gate",
        "timeframe": "1h",
        "scan": {
            "min_correlation": 0.8,
            "n_scanned": len(scanned),
            "n_cointegrated": len(scanned_coint),
            "top3": top3,
        },
        "walk_forward": {
            "params": {
                "n_windows": 4,
                "lookback": 80,
                "entry_z": 2.0,
                "exit_z": 0.5,
                "maker_fee_bps": 1.0,
            },
            "validated_pairs": validated,
        },
        "portfolio": portfolio,
        "notes": [
            "PairsEngine is spot-mode (no shorts). 'Pairs' are relative-value directional allocations, not a fully hedged long/short spread.",
            "Portfolio optimization uses per-window profit series; this is a coarse grid over non-negative weights summing to 1.",
        ],
    }

    out_path = out_dir / "agent_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] Wrote report: {out_path}")


if __name__ == "__main__":
    main()
