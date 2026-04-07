"""Walk-Forward Validator — rolling out-of-sample backtesting.

Instead of a single train/test split, this rolls through time:
  Window 1: train[0:T], test[T:T+H]
  Window 2: train[S:T+S], test[T+S:T+S+H]
  ...
Only strategies that are profitable across ALL windows are trustworthy.

Usage:
    from workspace.walk_forward import WalkForwardValidator
    wf = WalkForwardValidator(train_bars=500, test_bars=200, step_bars=200)
    report = wf.validate(strategy_path="workspace/strategies/my.py")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class WindowResult:
    """Result of one walk-forward window."""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_bars: int
    test_bars: int
    backtest: Dict[str, Any]  # from run_backtest
    sharpe: float = 0.0
    profit_pct: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    max_dd_pct: float = 0.0
    ok: bool = False
    error: str = ""


@dataclass
class WalkForwardReport:
    """Aggregated walk-forward validation report."""
    strategy_path: str
    n_windows: int = 0
    n_successful: int = 0
    windows: List[WindowResult] = field(default_factory=list)

    # Aggregated stats (across all successful windows)
    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0
    mean_profit: float = 0.0
    mean_trades: float = 0.0
    mean_dd: float = 0.0
    pct_profitable_windows: float = 0.0

    # Verdict
    passed: bool = False
    verdict: str = ""

    def summary(self) -> str:
        lines = [
            f"Walk-Forward: {self.n_windows} windows, {self.n_successful} successful",
            f"  Mean Sharpe: {self.mean_sharpe:.4f} +/- {self.std_sharpe:.4f}",
            f"  Mean Profit: {self.mean_profit:.2f}%",
            f"  Mean Trades: {self.mean_trades:.0f}",
            f"  Mean MaxDD: {self.mean_dd:.2f}%",
            f"  Profitable windows: {self.pct_profitable_windows:.0%}",
            f"  VERDICT: {'PASS' if self.passed else 'FAIL'} — {self.verdict}",
        ]
        return "\n".join(lines)


class WalkForwardValidator:
    """Rolling walk-forward validator."""

    def __init__(
        self,
        *,
        train_bars: int = 500,
        test_bars: int = 200,
        step_bars: int = 200,
        min_windows: int = 3,
        pass_threshold_sharpe: float = 0.0,
        pass_threshold_pct_profitable: float = 0.5,
    ):
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.min_windows = min_windows
        self.pass_threshold_sharpe = pass_threshold_sharpe
        self.pass_threshold_pct_profitable = pass_threshold_pct_profitable

    def _compute_windows(self, dates: pd.Series) -> List[dict]:
        """Compute train/test date ranges for each window."""
        dates = pd.to_datetime(dates).sort_values().reset_index(drop=True)
        n = len(dates)
        windows = []
        start = 0

        while start + self.train_bars + self.test_bars <= n:
            train_start_idx = start
            train_end_idx = start + self.train_bars
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + self.test_bars, n)

            windows.append({
                "train_start": dates.iloc[train_start_idx].strftime("%Y%m%d"),
                "train_end": dates.iloc[train_end_idx - 1].strftime("%Y%m%d"),
                "test_start": dates.iloc[test_start_idx].strftime("%Y%m%d"),
                "test_end": dates.iloc[test_end_idx - 1].strftime("%Y%m%d"),
                "test_timerange": f"{dates.iloc[test_start_idx].strftime('%Y%m%d')}-{dates.iloc[test_end_idx - 1].strftime('%Y%m%d')}",
                "train_bars": self.train_bars,
                "test_bars": test_end_idx - test_start_idx,
            })
            start += self.step_bars

        return windows

    def validate(
        self,
        strategy_path: str | Path,
        strategy_name: Optional[str] = None,
        *,
        data_path: Optional[str | Path] = None,
        exchange: str = "kucoin",
        pair: str = "BTC/USDT",
        timeframe: str = "1h",
    ) -> WalkForwardReport:
        """Run walk-forward validation on a strategy."""
        from workspace.backtest_api import run_backtest

        strategy_path = Path(strategy_path)
        report = WalkForwardReport(strategy_path=str(strategy_path))

        # Load dates from data
        if data_path is None:
            slug = pair.replace("/", "_")
            data_path = ROOT / "user_data" / "data" / exchange / f"{slug}-{timeframe}.feather"
        data_path = Path(data_path)

        if not data_path.exists():
            report.verdict = f"Data not found: {data_path}"
            return report

        df = pd.read_feather(data_path)
        dates = pd.to_datetime(df["date"])

        # Compute windows
        windows = self._compute_windows(dates)
        report.n_windows = len(windows)

        if len(windows) < self.min_windows:
            report.verdict = (
                f"Only {len(windows)} windows (need {self.min_windows}). "
                f"Data has {len(df)} bars, need at least "
                f"{self.train_bars + self.test_bars + (self.min_windows - 1) * self.step_bars}."
            )
            return report

        # Run backtest for each window
        for i, w in enumerate(windows):
            print(f"  Window {i+1}/{len(windows)}: test {w['test_timerange']}")
            bt = run_backtest(
                str(strategy_path),
                strategy_name,
                timerange=w["test_timerange"],
            )

            wr = WindowResult(
                window_id=i + 1,
                train_start=w["train_start"],
                train_end=w["train_end"],
                test_start=w["test_start"],
                test_end=w["test_end"],
                train_bars=w["train_bars"],
                test_bars=w["test_bars"],
                backtest=bt,
            )

            if bt.get("ok"):
                wr.ok = True
                wr.sharpe = float(bt.get("sharpe", 0))
                wr.profit_pct = float(bt.get("profit_pct", 0))
                wr.trades = int(bt.get("trades", 0))
                wr.win_rate = float(bt.get("win_rate", 0))
                wr.max_dd_pct = float(bt.get("max_drawdown_pct", 0))
                report.n_successful += 1
            else:
                wr.error = str(bt.get("error", ""))[:200]

            report.windows.append(wr)

        # Aggregate stats
        successful = [w for w in report.windows if w.ok and w.trades > 0]
        if successful:
            sharpes = [w.sharpe for w in successful]
            report.mean_sharpe = float(np.mean(sharpes))
            report.std_sharpe = float(np.std(sharpes))
            report.mean_profit = float(np.mean([w.profit_pct for w in successful]))
            report.mean_trades = float(np.mean([w.trades for w in successful]))
            report.mean_dd = float(np.mean([w.max_dd_pct for w in successful]))
            n_profitable = sum(1 for w in successful if w.profit_pct > 0)
            report.pct_profitable_windows = n_profitable / len(successful) if successful else 0
        else:
            report.verdict = "No successful windows with trades"
            return report

        # Verdict
        if (
            report.mean_sharpe >= self.pass_threshold_sharpe
            and report.pct_profitable_windows >= self.pass_threshold_pct_profitable
        ):
            report.passed = True
            report.verdict = (
                f"PASS: mean Sharpe {report.mean_sharpe:.2f} >= {self.pass_threshold_sharpe}, "
                f"{report.pct_profitable_windows:.0%} profitable >= {self.pass_threshold_pct_profitable:.0%}"
            )
        else:
            report.verdict = (
                f"FAIL: mean Sharpe {report.mean_sharpe:.2f} (need >= {self.pass_threshold_sharpe}), "
                f"{report.pct_profitable_windows:.0%} profitable (need >= {self.pass_threshold_pct_profitable:.0%})"
            )

        # Save report
        results_dir = ROOT / "workspace" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        report_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy_path": str(strategy_path),
            "n_windows": report.n_windows,
            "n_successful": report.n_successful,
            "mean_sharpe": report.mean_sharpe,
            "std_sharpe": report.std_sharpe,
            "mean_profit": report.mean_profit,
            "pct_profitable": report.pct_profitable_windows,
            "passed": report.passed,
            "verdict": report.verdict,
            "windows": [
                {
                    "id": w.window_id,
                    "test_range": f"{w.test_start}-{w.test_end}",
                    "sharpe": w.sharpe,
                    "profit_pct": w.profit_pct,
                    "trades": w.trades,
                    "ok": w.ok,
                }
                for w in report.windows
            ],
        }
        rp = results_dir / f"walkforward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        rp.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

        return report


__all__ = ["WalkForwardValidator", "WalkForwardReport", "WindowResult"]
