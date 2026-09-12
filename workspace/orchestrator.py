"""Orchestrator — ties all workspace modules into an automated research loop.

Usage:
    from workspace.orchestrator import run_experiment, run_research_loop

    # Single experiment
    result = run_experiment(strategy_path="workspace/strategies/my.py")

    # Full loop: backtest all strategies, rank, report
    report = run_research_loop()
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

from workspace.backtest_api import run_backtest
from workspace.evaluator import evaluate
from workspace.tracker import record_experiment


def run_experiment(
    strategy_path: str | Path,
    strategy_name: Optional[str] = None,
    *,
    model_name: Optional[str] = None,
    notes: str = "",
    tags: Optional[List[str]] = None,
    timerange: str = "20251126-20260125",
    config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run a single experiment: backtest → evaluate → record.

    Returns combined result with backtest metrics, evaluation score, and experiment ID.
    """
    strategy_path = str(strategy_path)

    # 1. Backtest
    bt_result = run_backtest(
        strategy_path=strategy_path,
        strategy_name=strategy_name,
        config_path=config_path,
        timerange=timerange,
    )

    # 2. Evaluate
    evaluation = evaluate(bt_result)

    # 3. Record
    record = record_experiment(
        backtest_result=bt_result,
        evaluation=evaluation,
        strategy_name=strategy_name or bt_result.get("strategy", "unknown"),
        model_name=model_name,
        notes=notes,
        tags=tags or [],
    )

    return {
        "experiment_id": record["id"],
        "backtest": bt_result,
        "evaluation": evaluation,
        "record": record,
    }


def scan_strategies() -> List[Path]:
    """Find all strategy .py files in workspace/strategies/."""
    strategies_dir = ROOT / "workspace" / "strategies"
    return sorted(
        p for p in strategies_dir.glob("*.py")
        if not p.name.startswith("_") and p.name != "__init__.py"
    )


def run_research_loop(
    *,
    timerange: str = "20251126-20260125",
    config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run all strategies in workspace/strategies/, evaluate, rank.

    Returns a research report with all results and recommendations.
    """
    strategies = scan_strategies()
    results: List[Dict[str, Any]] = []

    for strat_path in strategies:
        print(f"\n{'='*60}")
        print(f"Running: {strat_path.name}")
        print(f"{'='*60}")

        exp = run_experiment(
            strategy_path=strat_path,
            timerange=timerange,
            config_path=config_path,
            notes=f"research_loop auto-run",
            tags=["research_loop"],
        )
        bt = exp["backtest"]
        ev = exp["evaluation"]

        if bt.get("ok"):
            print(f"  Trades: {bt['trades']}, Sharpe: {bt['sharpe']:.4f}, "
                  f"Profit: {bt['profit_pct']}%, Score: {ev['total_score']}/100 ({ev['grade']})")
        else:
            print(f"  FAILED: {bt.get('error', 'unknown')[:100]}")

        results.append(exp)

    # Rank by evaluation score
    successful = [r for r in results if r["backtest"].get("ok")]
    successful.sort(key=lambda r: r["evaluation"]["total_score"], reverse=True)

    # Build report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_strategies": len(strategies),
        "successful_runs": len(successful),
        "failed_runs": len(results) - len(successful),
        "ranking": [],
        "best": None,
        "recommendations": [],
    }

    for rank, r in enumerate(successful, 1):
        bt = r["backtest"]
        ev = r["evaluation"]
        entry = {
            "rank": rank,
            "experiment_id": r["experiment_id"],
            "strategy": bt.get("strategy", "unknown"),
            "strategy_path": bt.get("strategy_path", ""),
            "score": ev["total_score"],
            "grade": ev["grade"],
            "sharpe": bt.get("sharpe", 0),
            "profit_pct": bt.get("profit_pct", 0),
            "max_drawdown_pct": bt.get("max_drawdown_pct", 0),
            "trades": bt.get("trades", 0),
        }
        report["ranking"].append(entry)

    if successful:
        best = successful[0]
        report["best"] = {
            "strategy": best["backtest"].get("strategy"),
            "score": best["evaluation"]["total_score"],
            "grade": best["evaluation"]["grade"],
        }
        report["recommendations"] = best["evaluation"].get("suggestions", [])

    # Save report
    report_path = ROOT / "workspace" / "results" / f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    return report


__all__ = ["run_experiment", "run_research_loop", "scan_strategies"]
