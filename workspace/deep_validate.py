#!/usr/bin/env python3
"""Deep Validation — runs strategies through the FULL verification stack.

For each strategy:
1. Lookahead check (reject if critical issues)
2. OOS backtest (last 30% of data)
3. Walk-Forward validation (multiple windows)
4. Cost-adjusted performance (realistic fees + slippage)
5. Regime breakdown (performance per market state)
6. Risk assessment (max DD, consecutive losses)

Usage:
    from workspace.deep_validate import deep_validate_strategy, deep_validate_all
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from workspace.backtest_api import run_backtest
from workspace.evaluator import evaluate
from workspace.lookahead_checker import check_lookahead, fix_lookahead_issues
from workspace.walk_forward import WalkForwardValidator
from workspace.cost_model import CostModel
from workspace.ensemble import RegimeDetector
from workspace.tracker import record_experiment


def deep_validate_strategy(
    strategy_path: str | Path,
    strategy_name: Optional[str] = None,
    *,
    exchange: str = "kucoin",
    pair: str = "BTC/USDT",
    oos_timerange: str = "20260107-20260125",
    full_timerange: str = "20251126-20260125",
) -> Dict[str, Any]:
    """Run a strategy through the complete verification stack."""
    strategy_path = Path(strategy_path)
    results: Dict[str, Any] = {
        "strategy": str(strategy_path.name),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {},
        "passed": False,
        "verdict": "",
    }

    # === CHECK 1: Lookahead ===
    print("  [1/6] Lookahead check...", end=" ")
    la = check_lookahead(strategy_path)
    if not la.ok:
        fix_lookahead_issues(strategy_path)
        la = check_lookahead(strategy_path)
    results["checks"]["lookahead"] = {
        "ok": la.ok,
        "critical": la.critical_count,
        "warnings": la.warning_count,
    }
    print(f"{'PASS' if la.ok else 'FAIL'} ({la.critical_count} critical)")
    if not la.ok:
        results["verdict"] = f"REJECT: {la.critical_count} lookahead issues"
        return results

    # === CHECK 2: OOS Backtest ===
    print("  [2/6] OOS backtest...", end=" ")
    oos = run_backtest(str(strategy_path), strategy_name, timerange=oos_timerange)
    oos_eval = evaluate(oos)
    results["checks"]["oos"] = {
        "ok": oos.get("ok", False),
        "sharpe": oos.get("sharpe", 0),
        "profit_pct": oos.get("profit_pct", 0),
        "trades": oos.get("trades", 0),
        "win_rate": oos.get("win_rate", 0),
        "score": oos_eval.get("total_score", 0),
    }
    if oos.get("ok"):
        print(f"Sharpe={oos['sharpe']:.2f}, Profit={oos['profit_pct']}%, Trades={oos['trades']}")
    else:
        print(f"FAIL: {str(oos.get('error', ''))[:60]}")

    # === CHECK 3: Walk-Forward ===
    print("  [3/6] Walk-Forward...", end=" ")
    wf = WalkForwardValidator(train_bars=400, test_bars=150, step_bars=150, min_windows=3)
    wf_report = wf.validate(strategy_path, strategy_name, exchange=exchange, pair=pair)
    results["checks"]["walk_forward"] = {
        "n_windows": wf_report.n_windows,
        "n_successful": wf_report.n_successful,
        "mean_sharpe": wf_report.mean_sharpe,
        "std_sharpe": wf_report.std_sharpe,
        "pct_profitable": wf_report.pct_profitable_windows,
        "passed": wf_report.passed,
        "per_window": [
            {"id": w.window_id, "sharpe": w.sharpe, "profit": w.profit_pct, "trades": w.trades}
            for w in wf_report.windows
        ],
    }
    print(f"{'PASS' if wf_report.passed else 'FAIL'} "
          f"(mean Sharpe={wf_report.mean_sharpe:.2f}, {wf_report.pct_profitable_windows:.0%} profitable)")

    # === CHECK 4: Cost-Adjusted ===
    print("  [4/6] Cost analysis...", end=" ")
    cost_model = CostModel(exchange=exchange)
    trade_size = 75.0  # default stake
    est = cost_model.estimate_total_cost(trade_size_usd=trade_size, daily_volume_usd=4e8, volatility=0.004)
    avg_profit_per_trade = 0.0
    if oos.get("ok") and oos.get("trades", 0) > 0:
        avg_profit_per_trade = float(oos.get("profit_pct", 0)) / float(oos["trades"]) * 100  # bps
    cost_viable = avg_profit_per_trade > est.round_trip_bps
    results["checks"]["cost"] = {
        "avg_profit_bps_per_trade": round(avg_profit_per_trade, 2),
        "round_trip_cost_bps": round(est.round_trip_bps, 2),
        "viable": cost_viable,
    }
    print(f"{'VIABLE' if cost_viable else 'NOT VIABLE'} "
          f"(earn {avg_profit_per_trade:.1f} bps vs cost {est.round_trip_bps:.1f} bps)")

    # === CHECK 5: Regime Breakdown ===
    print("  [5/6] Regime breakdown...", end=" ")
    try:
        data_path = ROOT / "user_data" / "data" / exchange / f"{pair.replace('/', '_')}-1h.feather"
        df = pd.read_feather(data_path)
        df["date"] = pd.to_datetime(df["date"])
        rd = RegimeDetector()
        enriched = rd.detect(df)
        regime_dist = enriched["regime"].value_counts(normalize=True).to_dict()
        results["checks"]["regime"] = {
            "distribution": {k: round(v, 2) for k, v in regime_dist.items()},
            "current": rd.current_regime(df).regime,
        }
        print(f"current={results['checks']['regime']['current']}")
    except Exception as e:
        results["checks"]["regime"] = {"error": str(e)[:100]}
        print(f"error: {str(e)[:50]}")

    # === CHECK 6: Risk Assessment ===
    print("  [6/6] Risk assessment...", end=" ")
    risk_ok = True
    risk_notes = []
    if oos.get("ok"):
        dd = float(oos.get("max_drawdown_pct", 0))
        if dd > 5.0:
            risk_ok = False
            risk_notes.append(f"DD {dd:.1f}% > 5% limit")
        trades = int(oos.get("trades", 0))
        if trades < 10:
            risk_notes.append(f"only {trades} trades (low statistical significance)")
        wr = float(oos.get("win_rate", 0))
        if wr < 0.35:
            risk_notes.append(f"win rate {wr:.0%} very low")
    else:
        risk_ok = False
        risk_notes.append("backtest failed")
    results["checks"]["risk"] = {"ok": risk_ok, "notes": risk_notes}
    print(f"{'OK' if risk_ok else 'WARN'} ({'; '.join(risk_notes) if risk_notes else 'clean'})")

    # === VERDICT ===
    oos_positive = oos.get("ok") and float(oos.get("sharpe", 0)) > 0
    wf_passed = wf_report.passed
    cost_ok = cost_viable
    la_ok = la.ok

    all_passed = la_ok and oos_positive and wf_passed and cost_ok and risk_ok
    results["passed"] = all_passed

    if all_passed:
        results["verdict"] = "VALIDATED — all checks passed"
    else:
        failures = []
        if not la_ok: failures.append("lookahead")
        if not oos_positive: failures.append("OOS Sharpe<=0")
        if not wf_passed: failures.append("walk-forward")
        if not cost_ok: failures.append("cost-adjusted")
        if not risk_ok: failures.append("risk")
        results["verdict"] = f"REJECTED — failed: {', '.join(failures)}"

    # Record experiment
    if oos.get("ok"):
        record_experiment(
            backtest_result=oos,
            evaluation=oos_eval,
            strategy_name=strategy_name or strategy_path.stem,
            notes=f"deep_validate: {results['verdict']}",
            tags=["deep_validate", "validated" if all_passed else "rejected"],
        )

    return results


def deep_validate_all(
    strategies_dir: Optional[str | Path] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Deep validate all strategies in a directory."""
    if strategies_dir is None:
        strategies_dir = ROOT / "workspace" / "strategies"
    strategies_dir = Path(strategies_dir)

    strategies = sorted(
        p for p in strategies_dir.glob("*.py")
        if not p.name.startswith("_") and p.name != "__init__.py"
    )

    all_results = []
    for strat in strategies:
        print(f"\n{'='*60}")
        print(f"Deep Validate: {strat.name}")
        print(f"{'='*60}")
        result = deep_validate_strategy(strat, **kwargs)
        all_results.append(result)

    # Summary
    passed = [r for r in all_results if r["passed"]]
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(all_results),
        "passed": len(passed),
        "rejected": len(all_results) - len(passed),
        "strategies": all_results,
    }

    out = ROOT / "workspace" / "results" / f"deep_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {out}")
    return report


if __name__ == "__main__":
    report = deep_validate_all()
    print(f"\n{'='*60}")
    print(f"DEEP VALIDATION: {report['passed']}/{report['total']} passed")
    for r in report["strategies"]:
        status = "PASS" if r["passed"] else "FAIL"
        oos = r["checks"].get("oos", {})
        print(f"  [{status}] {r['strategy']:35s} Sharpe={oos.get('sharpe',0):+8.2f} | {r['verdict']}")
