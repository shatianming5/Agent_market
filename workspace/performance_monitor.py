"""Performance Monitor — tracks strategy health and detects alpha decay.

Usage:
    from workspace.performance_monitor import PerformanceMonitor
    pm = PerformanceMonitor()
    report = pm.daily_check()
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class PerformanceMonitor:
    """Monitors active strategies for performance degradation."""

    def __init__(self, log_path: Optional[str | Path] = None):
        self.log_path = Path(log_path or ROOT / "workspace" / "results" / "monitor_log.jsonl")

    def check_pair_health(
        self,
        pair_a: str,
        pair_b: str,
        *,
        exchange: str = "gate",
        lookback: int = 100,
    ) -> Dict[str, Any]:
        """Check if a pairs trade is still healthy."""
        from workspace.pairs_engine import PairsEngine

        pe = PairsEngine(pair_a, pair_b, exchange=exchange)
        try:
            coint = pe.cointegration_test(lookback=lookback)
        except Exception as e:
            return {"healthy": False, "reason": str(e)[:100]}

        # Health checks
        issues = []
        if not coint.get("cointegrated"):
            issues.append("cointegration lost")
        if coint.get("correlation", 0) < 0.7:
            issues.append(f"correlation dropped to {coint['correlation']:.2f}")
        if coint.get("half_life_bars", 9999) > 500:
            issues.append(f"half-life too long: {coint['half_life_bars']:.0f} bars")

        return {
            "pair": f"{pair_a}/{pair_b}",
            "healthy": len(issues) == 0,
            "issues": issues,
            "correlation": coint.get("correlation", 0),
            "cointegrated": coint.get("cointegrated", False),
            "half_life": coint.get("half_life_bars", 0),
        }

    def rolling_sharpe(
        self,
        returns: List[float],
        window: int = 20,
    ) -> List[float]:
        """Compute rolling Sharpe ratio."""
        if len(returns) < window:
            return []
        arr = np.array(returns)
        sharpes = []
        for i in range(window, len(arr) + 1):
            w = arr[i - window:i]
            if np.std(w) > 0:
                sharpes.append(float(np.mean(w) / np.std(w) * np.sqrt(252)))
            else:
                sharpes.append(0.0)
        return sharpes

    def detect_alpha_decay(
        self,
        sharpe_series: List[float],
        *,
        threshold: float = 0.0,
        consecutive_days: int = 5,
    ) -> Dict[str, Any]:
        """Detect if alpha is decaying (Sharpe consistently below threshold)."""
        if len(sharpe_series) < consecutive_days:
            return {"decaying": False, "reason": "insufficient data"}

        recent = sharpe_series[-consecutive_days:]
        all_below = all(s < threshold for s in recent)
        trend = np.polyfit(range(len(sharpe_series[-20:])), sharpe_series[-20:], 1)[0] if len(sharpe_series) >= 20 else 0

        return {
            "decaying": all_below,
            "recent_sharpes": recent,
            "trend_slope": float(trend),
            "recommendation": "RETIRE" if all_below else ("WATCH" if trend < -0.1 else "OK"),
        }

    def daily_check(
        self,
        strategies: Optional[List[Dict[str, Any]]] = None,
        *,
        apply_transitions: bool = False,
    ) -> Dict[str, Any]:
        """Run daily health checks on all monitored strategies."""
        from workspace.strategy_lifecycle import LifecycleManager, StrategyState

        lm = LifecycleManager()
        if strategies is None:
            strategies = lm.list_by_state(StrategyState.PAPER) + lm.list_by_state(StrategyState.ACTIVE)

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checked": len(strategies),
            "healthy": 0,
            "issues": [],
            "actions": [],
        }

        for strat in strategies:
            name = strat["name"]
            config = strat.get("config", {})

            # Check pair health if pairs strategy
            if strat.get("type") == "pairs":
                pair_a = config.get("pair_a", "")
                pair_b = config.get("pair_b", "")
                if pair_a and pair_b:
                    health = self.check_pair_health(pair_a, pair_b, exchange=config.get("exchange", "gate"))
                    if health["healthy"]:
                        report["healthy"] += 1
                    else:
                        report["issues"].append({
                            "strategy": name,
                            "type": "pair_health",
                            "details": health["issues"],
                        })

            # Check alpha decay for active strategies
            if strat["state"] == StrategyState.ACTIVE.value:
                sharpes = strat.get("rolling_sharpe", [])
                if sharpes:
                    decay = self.detect_alpha_decay(sharpes)
                    if decay["decaying"]:
                        report["actions"].append({
                            "strategy": name,
                            "action": "RETIRE",
                            "reason": "alpha decay detected",
                        })

        if apply_transitions:
            review_actions = lm.auto_review(health_issues=report["issues"])
            report["auto_review"] = review_actions

        # Log
        self._log(report)
        return report

    def _log(self, report: Dict[str, Any]):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")


__all__ = ["PerformanceMonitor"]
