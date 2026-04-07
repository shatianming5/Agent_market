"""Continuous Runner — main loop that orchestrates the full research cycle.

Phases:
  1. Data Refresh — download latest candles
  2. Strategy Discovery — OpenCode scans + writes strategies
  3. Validation — walk-forward test new strategies
  4. Paper Trading — simulate validated strategies
  5. Performance Review — check health, promote/retire
  6. Report — generate daily summary

Usage:
    from workspace.continuous_runner import ContinuousRunner
    runner = ContinuousRunner(exchange="gate")
    report = runner.run_cycle()  # one full cycle
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]


class ContinuousRunner:
    """Main loop for continuous strategy research and management."""

    def __init__(
        self,
        *,
        exchange: str = "gate",
        pairs: Optional[List[str]] = None,
        timeframe: str = "1h",
        maker_fee_bps: float = 1.0,
    ):
        self.exchange = exchange
        self.pairs = pairs or ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT",
                                "XRP/USDT", "AVAX/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"]
        self.timeframe = timeframe
        self.maker_fee_bps = maker_fee_bps
        self.results_dir = ROOT / "workspace" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_cycle(self, *, skip_download: bool = False, skip_discovery: bool = False) -> Dict[str, Any]:
        """Run one complete research cycle."""
        cycle_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phases": {},
        }

        # Phase 1: Data Refresh
        print(f"\n{'='*60}")
        print("PHASE 1: Data Refresh")
        print(f"{'='*60}")
        if skip_download:
            cycle_report["phases"]["data_refresh"] = {"skipped": True}
            print("  Skipped")
        else:
            dr = self._phase_data_refresh()
            cycle_report["phases"]["data_refresh"] = dr
            print(f"  {dr.get('status', 'done')}")

        # Phase 2: Strategy Discovery
        print(f"\n{'='*60}")
        print("PHASE 2: Strategy Discovery")
        print(f"{'='*60}")
        if skip_discovery:
            cycle_report["phases"]["discovery"] = {"skipped": True}
            print("  Skipped")
        else:
            disc = self._phase_discovery()
            cycle_report["phases"]["discovery"] = disc
            print(f"  Found {disc.get('new_validated', 0)} new validated strategies")

        # Phase 3: Paper Trading Update
        print(f"\n{'='*60}")
        print("PHASE 3: Paper Trading")
        print(f"{'='*60}")
        paper = self._phase_paper_trading()
        cycle_report["phases"]["paper_trading"] = paper
        print(f"  {paper.get('n_paper', 0)} strategies in paper pool")

        # Phase 4: Performance Review
        print(f"\n{'='*60}")
        print("PHASE 4: Performance Review")
        print(f"{'='*60}")
        review = self._phase_performance_review()
        cycle_report["phases"]["review"] = review
        print(f"  Actions: {len(review.get('actions', []))}")

        # Phase 5: Report
        print(f"\n{'='*60}")
        print("PHASE 5: Report")
        print(f"{'='*60}")
        summary = self._phase_report(cycle_report)
        cycle_report["summary"] = summary
        print(f"  {summary.get('status', 'done')}")

        # Phase 6: Cleanup
        print(f"\n{'='*60}")
        print("PHASE 6: Cleanup")
        print(f"{'='*60}")
        cleanup_result = self._phase_cleanup()
        cycle_report["phases"]["cleanup"] = cleanup_result
        if cleanup_result.get("files_removed", 0) > 0:
            print(f"  Removed {cleanup_result['files_removed']} files, freed {cleanup_result.get('bytes_freed_mb', 0)} MB")
        else:
            print(f"  Nothing to clean")

        # Save cycle report
        report_path = self.results_dir / f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.write_text(json.dumps(cycle_report, indent=2, ensure_ascii=False, default=str))
        print(f"\nCycle report: {report_path}")

        return cycle_report

    def _phase_data_refresh(self) -> Dict[str, Any]:
        """Download latest data."""
        download_script = ROOT / "workspace" / "download_data.py"
        if not download_script.exists():
            return {"status": "no download script"}

        try:
            proc = subprocess.run(
                [sys.executable, str(download_script), "--exchange", self.exchange,
                 "--days", "30", "--outdir", str(ROOT / "user_data" / "data")],
                cwd=str(ROOT), capture_output=True, text=True, timeout=300,
            )
            return {"status": "ok" if proc.returncode == 0 else "error", "returncode": proc.returncode}
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            return {"status": "error", "error": str(e)[:200]}

    def _phase_discovery(self) -> Dict[str, Any]:
        """Scan for new pairs and validate through the central GatePipeline."""
        from workspace.pairs_engine import scan_pairs
        from workspace.strategy_lifecycle import LifecycleManager, StrategyState
        from workspace.gate_pipeline import GatePipeline
        from pathlib import Path
        import tempfile

        lm = LifecycleManager()
        gp = GatePipeline()
        pairs = scan_pairs(exchange=self.exchange, min_correlation=0.8)
        cointegrated = [p for p in pairs if p["cointegrated"]]

        new_validated = 0
        for p in cointegrated:
            name = f"pairs_{p['pair_a'].split('/')[0]}_{p['pair_b'].split('/')[0]}"
            existing = lm.get(name)
            if existing and existing["state"] != StrategyState.RETIRED.value:
                continue

            # Create a temporary strategy file with the pair config
            strategy_dir = ROOT / "workspace" / "strategies" / "type_C_pairs"
            strategy_dir.mkdir(parents=True, exist_ok=True)
            strategy_file = strategy_dir / f"{name}.py"
            strategy_file.write_text(
                f'"""Auto-discovered pairs strategy: {name}"""\n'
                f'PAIR_A = "{p["pair_a"]}"\n'
                f'PAIR_B = "{p["pair_b"]}"\n'
                f'LOOKBACK = 80\n'
                f'ENTRY_Z = 2.0\n'
                f'EXIT_Z = 0.5\n',
                encoding="utf-8",
            )

            # Route through the CENTRAL gate pipeline (Gate 1→2→3)
            try:
                result = gp.run_gates(
                    strategy_file,
                    strategy_type="C_pairs",
                    exchange=self.exchange,
                    stop_on_fail=True,
                )
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Gate validation failed for %s: %s", name, exc)
                continue

            passed_gate3 = result.get("final_gate_passed") == "gate_3"
            if passed_gate3:
                lm.register(name, strategy_type="pairs", config={
                    "pair_a": p["pair_a"], "pair_b": p["pair_b"],
                    "exchange": self.exchange, "correlation": p["correlation"],
                    "half_life": p["half_life"],
                }, source="auto_discovery")
                lm.promote(name, "Gate 1-3 passed via GatePipeline")
                lm.promote(name, "auto-promote to paper")
                new_validated += 1

        return {
            "scanned": len(pairs),
            "cointegrated": len(cointegrated),
            "new_validated": new_validated,
        }

    def _phase_paper_trading(self) -> Dict[str, Any]:
        """Simulate one day of paper trading for all PAPER strategies."""
        from workspace.strategy_lifecycle import LifecycleManager, StrategyState
        from workspace.pairs_engine import PairsEngine
        import numpy as np

        lm = LifecycleManager()
        paper_strategies = lm.list_by_state(StrategyState.PAPER)

        for strat in paper_strategies:
            config = strat.get("config", {})
            if strat["type"] == "pairs":
                pair_a = config.get("pair_a", "")
                pair_b = config.get("pair_b", "")
                if not pair_a or not pair_b:
                    continue

                # Simulate latest day PnL
                pe = PairsEngine(pair_a, pair_b, exchange=config.get("exchange", self.exchange))
                try:
                    df = pe.load_data()
                    # Use last 48 bars for a "day" simulation
                    pe_day = PairsEngine(pair_a, pair_b, exchange=config.get("exchange", self.exchange))
                    pe_day._df = df.tail(48).reset_index(drop=True)
                    signals = pe_day.generate_signals(lookback=30, entry_z=2.0, exit_z=0.5)
                    bt = pe_day.backtest(signals, maker_fee_bps=self.maker_fee_bps)
                    lm.record_paper_day(strat["name"], bt.profit_pct)
                except Exception:
                    lm.record_paper_day(strat["name"], 0.0)

        return {"n_paper": len(paper_strategies)}

    def _phase_performance_review(self) -> Dict[str, Any]:
        """Review all strategies and apply auto-transitions."""
        from workspace.performance_monitor import PerformanceMonitor
        from workspace.strategy_lifecycle import LifecycleManager

        lm = LifecycleManager()
        pm = PerformanceMonitor()

        review = pm.daily_check()
        actions = lm.auto_review()

        return {
            "healthy": review.get("healthy", 0),
            "issues": review.get("issues", []),
            "actions": actions,
        }

    def _phase_report(self, cycle_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cycle summary."""
        from workspace.strategy_lifecycle import LifecycleManager

        lm = LifecycleManager()
        summary = lm.summary()

        return {
            "status": "done",
            "total_strategies": summary["total"],
            "by_state": summary["by_state"],
            "cycle_phases_completed": len(cycle_report.get("phases", {})),
        }


    def _phase_cleanup(self) -> Dict[str, Any]:
        """Clean up old files to prevent disk bloat."""
        try:
            from workspace.cleanup import auto_cleanup, disk_usage
            before = disk_usage()
            result = auto_cleanup(max_run_files=100, max_model_files=3, max_age_days=30)
            after = disk_usage()
            result["disk_before_mb"] = before.get("total_mb", 0)
            result["disk_after_mb"] = after.get("total_mb", 0)
            return result
        except Exception as e:
            return {"error": str(e)[:200]}


__all__ = ["ContinuousRunner"]
