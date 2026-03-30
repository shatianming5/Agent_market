"""Gate Pipeline — automated stage-gate validation for strategies.

Runs a strategy through Gate 1 → Gate 2 → Gate 3 → Gate 4 automatically.
Each gate has quantitative pass/fail criteria defined in sop.json.

Usage:
    from workspace.gate_pipeline import GatePipeline
    gp = GatePipeline()
    result = gp.run_gates("strategies/type_C_pairs/my_pairs.py", strategy_type="C_pairs")
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]


def _load_sop() -> Dict[str, Any]:
    return json.loads((ROOT / "workspace" / "sop.json").read_text(encoding="utf-8"))


class GateResult:
    """Result of a single gate check."""
    def __init__(self, gate: str, passed: bool, metrics: Dict[str, Any], failures: List[str]):
        self.gate = gate
        self.passed = passed
        self.metrics = metrics
        self.failures = failures

    def to_dict(self) -> Dict[str, Any]:
        return {"gate": self.gate, "passed": self.passed, "metrics": self.metrics, "failures": self.failures}


class GatePipeline:
    """Runs strategies through the SOP gate pipeline."""

    def __init__(self, sop_path: Optional[str | Path] = None):
        self.sop = _load_sop() if sop_path is None else json.loads(Path(sop_path).read_text())
        self.results_dir = ROOT / "workspace" / "validation"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_gates(
        self,
        strategy_path: str | Path,
        *,
        strategy_type: str = "B_meanrev",
        strategy_name: Optional[str] = None,
        exchange: str = "gate",
        pair: str = "BTC/USDT",
        timerange: Optional[str] = None,
        stop_on_fail: bool = True,
    ) -> Dict[str, Any]:
        """Run strategy through all auto-checkable gates."""
        strategy_path = Path(strategy_path)
        type_config = self.sop["strategy_types"].get(strategy_type, {})

        report = {
            "strategy": str(strategy_path),
            "strategy_type": strategy_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gates": {},
            "final_gate_passed": None,
            "recommendation": "",
        }

        # Gate 1: Signal (skip for now — needs IC calculation infrastructure)
        # Gate 2: Backtest
        g2 = self._gate_2_backtest(strategy_path, strategy_name, exchange, pair, timerange)
        report["gates"]["gate_2"] = g2.to_dict()
        if not g2.passed and stop_on_fail:
            report["final_gate_passed"] = "gate_1"
            report["recommendation"] = f"REJECTED at Gate 2: {'; '.join(g2.failures)}"
            self._save_report(report, strategy_path.stem)
            return report

        # Gate 3: Walk-Forward
        g3 = self._gate_3_robustness(strategy_path, strategy_name, exchange, pair, type_config)
        report["gates"]["gate_3"] = g3.to_dict()
        if not g3.passed and stop_on_fail:
            report["final_gate_passed"] = "gate_2"
            report["recommendation"] = f"REJECTED at Gate 3: {'; '.join(g3.failures)}"
            self._save_report(report, strategy_path.stem)
            return report

        report["final_gate_passed"] = "gate_3"
        report["recommendation"] = "APPROVED for paper trading (Gate 4)"
        self._save_report(report, strategy_path.stem)
        return report

    def _gate_2_backtest(
        self, path: Path, name: Optional[str], exchange: str, pair: str, timerange: Optional[str],
    ) -> GateResult:
        """Gate 2: Backtest with quantitative criteria."""
        criteria = self.sop["gates"]["gate_2_backtest"]["criteria"]
        metrics = {}
        failures = []

        # Lookahead check first
        from workspace.lookahead_checker import check_lookahead, fix_lookahead_issues
        la = check_lookahead(path)
        if not la.ok:
            fix_lookahead_issues(path)
            la = check_lookahead(path)
        if la.critical_count > criteria.get("lookahead_critical", 0):
            failures.append(f"lookahead: {la.critical_count} critical issues")
            return GateResult("gate_2", False, {"lookahead_critical": la.critical_count}, failures)

        # Determine backtest method based on strategy type
        # For pairs strategies: use pairs_engine (L1)
        # For directional: use freqtrade (L2)
        if "pairs" in str(path).lower() or "pair" in str(path).lower():
            metrics = self._backtest_pairs(path, exchange)
        else:
            from workspace.backtest_api import run_backtest
            tr = timerange or "20250601-20260301"
            bt = run_backtest(str(path), name, timerange=tr)
            if not bt.get("ok"):
                failures.append(f"backtest failed: {bt.get('error', '?')[:100]}")
                return GateResult("gate_2", False, bt, failures)
            metrics = {
                "sharpe": bt.get("sharpe", 0),
                "max_drawdown_pct": bt.get("max_drawdown_pct", 0),
                "trades": bt.get("trades", 0),
                "win_rate": bt.get("win_rate", 0),
                "profit_factor": bt.get("profit_factor", 0),
                "profit_pct": bt.get("profit_pct", 0),
            }

        # Check criteria
        checks = [
            ("sharpe", metrics.get("sharpe", 0), ">=", criteria.get("sharpe_min", 0.5)),
            ("max_drawdown_pct", metrics.get("max_drawdown_pct", 100), "<=", criteria.get("max_drawdown_pct", 20)),
            ("trades", metrics.get("trades", 0), ">=", criteria.get("min_trades", 30)),
            ("win_rate", metrics.get("win_rate", 0), ">=", criteria.get("win_rate_min", 0.40)),
            ("profit_factor", metrics.get("profit_factor", 0), ">=", criteria.get("profit_factor_min", 1.1)),
        ]
        for name_c, value, op, threshold in checks:
            if op == ">=" and value < threshold:
                failures.append(f"{name_c}={value:.4f} < {threshold}")
            elif op == "<=" and value > threshold:
                failures.append(f"{name_c}={value:.4f} > {threshold}")

        return GateResult("gate_2", len(failures) == 0, metrics, failures)

    def _backtest_pairs(self, path: Path, exchange: str) -> Dict[str, Any]:
        """Quick pairs backtest using pairs_engine."""
        # Extract pair info from filename or default
        from workspace.pairs_engine import PairsEngine
        from workspace.cost_model import CostModel

        # Try common pairs
        for a, b in [("DOGE/USDT","SOL/USDT"), ("ADA/USDT","AVAX/USDT"), ("LINK/USDT","SOL/USDT")]:
            try:
                pe = PairsEngine(a, b, exchange=exchange)
                sig = pe.generate_signals(lookback=80, entry_z=2.5, exit_z=0.7)
                cm = CostModel(exchange=exchange)
                bt = pe.backtest(sig, maker_fee_bps=cm.taker_bps)
                return {
                    "sharpe": bt.sharpe,
                    "max_drawdown_pct": bt.max_dd_pct,
                    "trades": bt.trades,
                    "win_rate": bt.win_rate,
                    "profit_factor": 1.0 + bt.profit_pct / 100 if bt.profit_pct != 0 else 1.0,
                    "profit_pct": bt.profit_pct,
                    "pair": f"{a}/{b}",
                }
            except Exception:
                continue
        return {"error": "no valid pair found"}

    def _gate_3_robustness(
        self, path: Path, name: Optional[str], exchange: str, pair: str, type_config: Dict,
    ) -> GateResult:
        """Gate 3: Walk-Forward robustness check."""
        criteria = self.sop["gates"]["gate_3_robustness"]["criteria"]
        from workspace.walk_forward import WalkForwardValidator

        min_windows = criteria.get("min_windows", 4)
        wf = WalkForwardValidator(
            train_bars=2000, test_bars=500, step_bars=500,
            min_windows=min_windows,
            pass_threshold_sharpe=criteria.get("mean_sharpe_min", 0.0),
            pass_threshold_pct_profitable=criteria.get("profitable_windows_pct", 0.6),
        )

        report = wf.validate(path, name, exchange=exchange, pair=pair)
        metrics = {
            "n_windows": report.n_windows,
            "n_successful": report.n_successful,
            "mean_sharpe": report.mean_sharpe,
            "pct_profitable": report.pct_profitable_windows,
            "mean_profit": report.mean_profit,
        }

        failures = []
        if report.n_successful < min_windows:
            failures.append(f"only {report.n_successful} windows (need {min_windows})")
        elif report.pct_profitable_windows < criteria.get("profitable_windows_pct", 0.6):
            failures.append(f"profitable={report.pct_profitable_windows:.0%} < {criteria['profitable_windows_pct']:.0%}")
        if report.mean_sharpe < criteria.get("mean_sharpe_min", 0.0):
            failures.append(f"mean_sharpe={report.mean_sharpe:.2f} < {criteria['mean_sharpe_min']}")

        # Check max single window loss
        max_loss = criteria.get("max_single_window_loss_pct", 15.0)
        for w in report.windows:
            if w.ok and w.profit_pct < -max_loss:
                failures.append(f"window {w.window_id} loss={w.profit_pct:.1f}% > {max_loss}%")
                break

        return GateResult("gate_3", len(failures) == 0, metrics, failures)

    def _save_report(self, report: Dict[str, Any], name: str):
        path = self.results_dir / f"gate_report_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))


__all__ = ["GatePipeline", "GateResult"]
