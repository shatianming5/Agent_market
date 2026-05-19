"""Report Generator — daily/weekly research reports.

Usage:
    from workspace.report_generator import generate_daily_report
    report = generate_daily_report()
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]


def generate_daily_report() -> Dict[str, Any]:
    """Generate a daily research status report."""
    from workspace.strategy_lifecycle import LifecycleManager
    from workspace.tracker import list_experiments
    from workspace.performance_monitor import PerformanceMonitor

    lm = LifecycleManager()
    pm = PerformanceMonitor()
    exps = list_experiments()

    summary = lm.summary()
    health = pm.daily_check()

    # Experiments stats
    recent = [e for e in exps[-20:] if e.get("backtest_ok")]
    sharpes = [e["metrics"]["sharpe"] for e in recent if e.get("metrics", {}).get("sharpe")]
    profits = [e["metrics"]["profit_pct"] for e in recent if e.get("metrics", {}).get("profit_pct")]

    report = {
        "type": "daily",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategies": {
            "total": summary["total"],
            "by_state": summary["by_state"],
            "active_names": [s["name"] for s in summary["strategies"] if s["state"] == "active"],
            "paper_names": [s["name"] for s in summary["strategies"] if s["state"] == "paper"],
            "paper_returns": [
                {
                    "name": s["name"],
                    "state": s["state"],
                    "paper_days": s.get("paper_days", 0),
                    "paper_dates": s.get("paper_dates", []),
                    "paper_pnl": s.get("paper_pnl", []),
                    "paper_daily_equity": s.get("paper_daily_equity", {}),
                    "paper_last_equity": s.get("paper_last_equity"),
                    "cumulative_return_pct": s.get("cumulative_return_pct", 0.0),
                    "latest_day_return_pct": s.get("latest_day_return_pct"),
                    "paper_last_processed_at": s.get("paper_last_processed_at"),
                    "paper_state_path": s.get("paper_state_path"),
                }
                for s in summary["strategies"]
                if s.get("paper_days", 0) > 0 or s.get("paper_state_path")
            ],
        },
        "health": {
            "healthy": health.get("healthy", 0),
            "issues": len(health.get("issues", [])),
            "issue_details": health.get("issues", [])[:5],
        },
        "experiments": {
            "total": len(exps),
            "recent_20": {
                "avg_sharpe": round(sum(sharpes) / len(sharpes), 2) if sharpes else 0,
                "avg_profit": round(sum(profits) / len(profits), 2) if profits else 0,
                "positive_sharpe": sum(1 for s in sharpes if s > 0),
            },
        },
        "next_actions": _suggest_actions(summary, health),
    }

    # Save
    reports_dir = ROOT / "workspace" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"daily_{datetime.now().strftime('%Y%m%d')}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    return report


def _suggest_actions(summary: Dict, health: Dict) -> list:
    actions = []
    by_state = summary.get("by_state", {})

    if by_state.get("paper", 0) == 0 and by_state.get("active", 0) == 0:
        actions.append("No strategies in paper/active — run discovery cycle")
    if health.get("issues"):
        actions.append(f"{len(health['issues'])} health issues — review pair stability")
    if by_state.get("discovered", 0) > 3:
        actions.append(f"{by_state['discovered']} strategies awaiting validation — run gate_pipeline")
    if by_state.get("paper", 0) > 0:
        actions.append("Paper strategies active — check daily PnL")

    return actions if actions else ["System healthy — continue monitoring"]


__all__ = ["generate_daily_report"]
