"""Continuous Runner — main loop that orchestrates the full research cycle.

Phases:
  1. Data Refresh — download latest candles
  2. Strategy Discovery — OpenCode scans + writes strategies
  3. Validation — walk-forward test new strategies
  4. Paper Trading — forward paper trading on fresh market data
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
        maker_fee_bps: float = 10.0,
        min_paper_pool: int = 3,
    ):
        self.exchange = exchange
        self.pairs = pairs or ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT",
                                "XRP/USDT", "AVAX/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"]
        self.timeframe = timeframe
        self.maker_fee_bps = maker_fee_bps
        self.min_paper_pool = max(1, int(min_paper_pool))
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
        self._write_latest_cycle_artifacts(cycle_report)
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
        from workspace.adaptive_params import AdaptiveEngine
        from workspace.pairs_engine import scan_pairs
        from workspace.strategy_lifecycle import LifecycleManager, StrategyState
        from workspace.gate_pipeline import GatePipeline

        lm = LifecycleManager()
        gp = GatePipeline()
        adaptive = AdaptiveEngine(exchange=self.exchange)
        pairs = scan_pairs(exchange=self.exchange, min_correlation=0.8)
        cointegrated = [p for p in pairs if p["cointegrated"]]

        paper_pool_before = self._paper_pool_size(lm)
        promoted_existing = self._top_up_paper_pool(lm)
        paper_pool_after_existing = self._paper_pool_size(lm)
        attempts_allowed = max(0, self.min_paper_pool - paper_pool_after_existing)
        strategy_dir = ROOT / "workspace" / "strategies" / "type_C_pairs"
        strategy_dir.mkdir(parents=True, exist_ok=True)

        candidate_rows: list[tuple[int, float, str, dict[str, Any], Optional[dict[str, Any]]]] = []
        skipped_existing: list[dict[str, Any]] = []
        for pair_spec in cointegrated:
            name = f"pairs_{pair_spec['pair_a'].split('/')[0]}_{pair_spec['pair_b'].split('/')[0]}"
            existing = lm.get(name)
            state = str((existing or {}).get("state") or "")
            if state in {StrategyState.PAPER.value, StrategyState.ACTIVE.value}:
                skipped_existing.append({"name": name, "state": state, "reason": "already_in_pool"})
                continue
            if state == StrategyState.VALIDATED.value:
                skipped_existing.append({"name": name, "state": state, "reason": "validated_backlog"})
                continue
            priority = 0 if state == StrategyState.DISCOVERED.value else (1 if state == StrategyState.RETIRED.value else 2)
            candidate_rows.append((priority, -float(pair_spec.get("correlation") or 0.0), name, pair_spec, existing))

        candidate_rows.sort(key=lambda item: (item[0], item[1], item[2]))

        new_validated = 0
        validated_names: list[str] = []
        promoted_new: list[str] = []
        attempts: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for _, _, name, pair_spec, existing in candidate_rows[:attempts_allowed]:
            adaptive_result = self._optimize_pairs_candidate(adaptive, pair_spec)
            params = self._resolve_pairs_params(adaptive_result)
            strategy_file = strategy_dir / f"{name}.py"
            strategy_file.write_text(
                self._render_pairs_strategy(name, pair_spec, params),
                encoding="utf-8",
            )

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
                errors.append({"name": name, "error": str(exc)[:200]})
                continue

            passed_gate3 = result.get("final_gate_passed") == "gate_3"
            attempts.append(
                {
                    "name": name,
                    "state_before": str((existing or {}).get("state") or "new"),
                    "passed_gate3": bool(passed_gate3),
                    "final_gate_passed": result.get("final_gate_passed"),
                    "params": params,
                }
            )
            if not passed_gate3:
                continue

            config = self._build_pairs_strategy_config(
                pair_spec=pair_spec,
                params=params,
                adaptive_result=adaptive_result,
                gate_result=result,
                strategy_file=strategy_file,
            )
            if existing:
                lm.update_entry(
                    name,
                    strategy_type="pairs",
                    config=config,
                    source="auto_discovery",
                    replace_config=True,
                )
                if existing.get("state") == StrategyState.RETIRED.value:
                    lm.reset_to_discovered(name, "rediscovered via auto discovery")
            else:
                lm.register(name, strategy_type="pairs", config=config, source="auto_discovery")

            if lm.get(name) and lm.get(name)["state"] == StrategyState.DISCOVERED.value:
                lm.promote(name, "Gate 1-3 passed via GatePipeline")
            if self._paper_pool_size(lm) < self.min_paper_pool and lm.get(name) and lm.get(name)["state"] == StrategyState.VALIDATED.value:
                if lm.promote(name, "auto-promote to paper to maintain minimum pool"):
                    promoted_new.append(name)
            validated_names.append(name)
            new_validated += 1

        return {
            "scanned": len(pairs),
            "cointegrated": len(cointegrated),
            "new_validated": new_validated,
            "validated_names": validated_names,
            "attempted": len(attempts),
            "attempts": attempts,
            "promoted_existing": promoted_existing,
            "promoted_new": promoted_new,
            "paper_pool_before": paper_pool_before,
            "paper_pool_after": self._paper_pool_size(lm),
            "paper_pool_target": self.min_paper_pool,
            "skipped_existing": skipped_existing,
            "errors": errors,
        }

    def _phase_paper_trading(self) -> Dict[str, Any]:
        """Run forward-only paper trading for all PAPER strategies."""
        from workspace.strategy_lifecycle import LifecycleManager, StrategyState
        from workspace.paper_cycle import PairsPaperCycle
        from workspace.freqtrade_paper_cycle import FreqtradePaperCycle

        lm = LifecycleManager()
        paper_strategies = lm.list_by_state(StrategyState.PAPER)
        paper_cycle = PairsPaperCycle(
            exchange=self.exchange,
            timeframe=self.timeframe,
            fee_bps=self.maker_fee_bps,
        )
        freqtrade_cycle = FreqtradePaperCycle(
            exchange=self.exchange,
            timeframe=self.timeframe,
        )
        updated = 0
        initialized = 0
        total_new_bars = 0
        strategy_results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for strat in paper_strategies:
            config = dict(strat.get("config") or {})
            paper_started_at = self._paper_started_at(strat)
            if paper_started_at and not config.get("paper_started_at"):
                config["paper_started_at"] = paper_started_at
            if strat["type"] == "pairs":
                try:
                    result = paper_cycle.run_pairs_strategy(strat["name"], config)
                    if not result.get("ok"):
                        continue
                    lm.sync_paper_tracking(
                        strat["name"],
                        daily_equity=result.get("daily_equity") or {},
                        last_processed_at=str(result.get("last_processed_at") or ""),
                        current_equity=result.get("current_equity"),
                        state_path=result.get("state_path"),
                    )
                    initialized += int(bool(result.get("initialized")))
                    updated += int(not bool(result.get("initialized")))
                    total_new_bars += int(result.get("new_bars") or 0)
                    strategy_results.append({
                        "name": strat["name"],
                        "type": strat["type"],
                        "state": strat.get("state"),
                        "initialized": bool(result.get("initialized")),
                        "bootstrapped": bool(result.get("bootstrapped")),
                        "new_bars": int(result.get("new_bars") or 0),
                        "orders_added": int(result.get("orders_added") or 0),
                        "paper_days": int(result.get("paper_days") or 0),
                        "current_equity": float(result.get("current_equity") or 0.0),
                        "cumulative_return_pct": float(result.get("cumulative_return_pct") or 0.0),
                        "latest_day_return_pct": result.get("latest_day_return_pct"),
                        "last_processed_at": result.get("last_processed_at"),
                        "state_path": result.get("state_path"),
                    })
                except Exception as exc:
                    errors.append({"name": strat["name"], "error": str(exc)[:200]})
                    continue
            elif strat["type"] in {"ml", "dl", "rl"}:
                try:
                    result = freqtrade_cycle.run_strategy(strat["name"], config)
                    if not result.get("ok"):
                        type_label = str(strat["type"])
                        errors.append({"name": strat["name"], "error": str(result.get("error") or f"{type_label} paper failed")[:200]})
                        continue
                    lm.sync_paper_tracking(
                        strat["name"],
                        daily_equity=result.get("daily_equity") or {},
                        last_processed_at=str(result.get("last_processed_at") or ""),
                        current_equity=result.get("current_equity"),
                        state_path=result.get("state_path"),
                    )
                    initialized += int(bool(result.get("initialized")))
                    updated += int(not bool(result.get("initialized")))
                    total_new_bars += int(result.get("new_days") or 0)
                    strategy_results.append({
                        "name": strat["name"],
                        "type": strat["type"],
                        "state": strat.get("state"),
                        "initialized": bool(result.get("initialized")),
                        "backfilled": bool(result.get("backfilled")),
                        "new_bars": int(result.get("new_days") or 0),
                        "orders_added": int(result.get("orders_added") or 0),
                        "paper_days": int(result.get("paper_days") or 0),
                        "current_equity": float(result.get("current_equity") or 0.0),
                        "cumulative_return_pct": float(result.get("cumulative_return_pct") or 0.0),
                        "latest_day_return_pct": result.get("latest_day_return_pct"),
                        "last_processed_at": result.get("last_processed_at"),
                        "state_path": result.get("state_path"),
                        "strategy_path": result.get("strategy_path"),
                        "timerange": result.get("timerange"),
                        "backtest_run_id": result.get("backtest_run_id"),
                        "backtest_zip": result.get("backtest_zip"),
                    })
                except Exception as exc:
                    errors.append({"name": strat["name"], "error": str(exc)[:200]})
                    continue

        return {
            "n_paper": len(paper_strategies),
            "initialized": initialized,
            "updated": updated,
            "new_bars": total_new_bars,
            "strategies": strategy_results,
            "errors": errors,
        }

    def _phase_performance_review(self) -> Dict[str, Any]:
        """Review all strategies and apply auto-transitions."""
        from workspace.performance_monitor import PerformanceMonitor
        from workspace.strategy_lifecycle import LifecycleManager

        lm = LifecycleManager()
        pm = PerformanceMonitor()

        review = pm.daily_check(apply_transitions=False)
        actions = lm.auto_review(health_issues=review.get("issues"))

        return {
            "healthy": review.get("healthy", 0),
            "issues": review.get("issues", []),
            "actions": actions,
        }

    def _phase_report(self, cycle_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cycle summary."""
        from workspace.report_generator import generate_daily_report
        from workspace.strategy_lifecycle import LifecycleManager

        lm = LifecycleManager()
        summary = lm.summary()
        paper_returns = [
            {
                "name": item["name"],
                "state": item["state"],
                "paper_days": item.get("paper_days", 0),
                "paper_dates": item.get("paper_dates", []),
                "paper_pnl": item.get("paper_pnl", []),
                "paper_daily_equity": item.get("paper_daily_equity", {}),
                "paper_last_equity": item.get("paper_last_equity"),
                "cumulative_return_pct": item.get("cumulative_return_pct", 0.0),
                "latest_day_return_pct": item.get("latest_day_return_pct"),
                "paper_last_processed_at": item.get("paper_last_processed_at"),
                "paper_state_path": item.get("paper_state_path"),
            }
            for item in summary["strategies"]
            if item.get("paper_days", 0) > 0 or item.get("paper_state_path")
        ]
        daily_report_generated = False
        daily_report_error = None
        daily_report_path = ROOT / "workspace" / "reports" / f"daily_{datetime.now().strftime('%Y%m%d')}.json"
        try:
            generate_daily_report()
            daily_report_generated = True
        except Exception as exc:
            daily_report_error = str(exc)[:200]

        return {
            "status": "done",
            "total_strategies": summary["total"],
            "by_state": summary["by_state"],
            "cycle_phases_completed": len(cycle_report.get("phases", {})),
            "paper_returns": paper_returns,
            "daily_report_generated": daily_report_generated,
            "daily_report_path": str(daily_report_path),
            "daily_report_error": daily_report_error,
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

    def _write_latest_cycle_artifacts(self, cycle_report: Dict[str, Any]) -> None:
        (self.results_dir / "last_cycle.json").write_text(
            json.dumps(cycle_report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        paper_returns = {
            "timestamp": cycle_report.get("timestamp"),
            "paper_trading": cycle_report.get("phases", {}).get("paper_trading", {}),
            "summary": cycle_report.get("summary", {}).get("paper_returns", []),
        }
        (self.results_dir / "paper_returns_latest.json").write_text(
            json.dumps(paper_returns, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def _paper_started_at(self, strategy: Dict[str, Any]) -> Optional[str]:
        for row in reversed(strategy.get("history") or []):
            if row.get("state") == "paper" and row.get("at"):
                return str(row["at"])
        return None

    def _paper_pool_size(self, lifecycle_manager: Any) -> int:
        from workspace.strategy_lifecycle import StrategyState

        return len(lifecycle_manager.list_by_state(StrategyState.PAPER)) + len(
            lifecycle_manager.list_by_state(StrategyState.ACTIVE)
        )

    def _top_up_paper_pool(self, lifecycle_manager: Any) -> list[str]:
        from workspace.strategy_lifecycle import StrategyState

        promoted: list[str] = []
        for entry in lifecycle_manager.list_by_state(StrategyState.VALIDATED):
            if self._paper_pool_size(lifecycle_manager) >= self.min_paper_pool:
                break
            if lifecycle_manager.promote(entry["name"], "auto-promote validated backlog to maintain minimum pool"):
                promoted.append(str(entry["name"]))
        return promoted

    def _optimize_pairs_candidate(self, adaptive_engine: Any, pair_spec: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = adaptive_engine.optimize_pairs(
                str(pair_spec["pair_a"]),
                str(pair_spec["pair_b"]),
                cost_bps=self.maker_fee_bps,
            )
        except Exception as exc:
            return {"error": str(exc)[:200]}
        return result if isinstance(result, dict) else {}

    def _resolve_pairs_params(self, adaptive_result: Dict[str, Any]) -> Dict[str, float]:
        best_params = adaptive_result.get("best_params") if isinstance(adaptive_result, dict) else {}
        return {
            "lookback": int((best_params or {}).get("lookback", 80)),
            "entry_z": float((best_params or {}).get("entry_z", 2.0)),
            "exit_z": float((best_params or {}).get("exit_z", 0.5)),
        }

    def _render_pairs_strategy(self, name: str, pair_spec: Dict[str, Any], params: Dict[str, float]) -> str:
        return (
            f'"""Auto-discovered pairs strategy: {name}"""\n'
            f'PAIR_A = "{pair_spec["pair_a"]}"\n'
            f'PAIR_B = "{pair_spec["pair_b"]}"\n'
            f'LOOKBACK = {int(params["lookback"])}\n'
            f'ENTRY_Z = {float(params["entry_z"])}\n'
            f'EXIT_Z = {float(params["exit_z"])}\n'
        )

    def _build_pairs_strategy_config(
        self,
        *,
        pair_spec: Dict[str, Any],
        params: Dict[str, float],
        adaptive_result: Dict[str, Any],
        gate_result: Dict[str, Any],
        strategy_file: Path,
    ) -> Dict[str, Any]:
        gate_2_metrics = (((gate_result.get("gates") or {}).get("gate_2") or {}).get("metrics") or {})
        gate_3_metrics = (((gate_result.get("gates") or {}).get("gate_3") or {}).get("metrics") or {})
        overfit_ratio = adaptive_result.get("overfit_ratio")
        if overfit_ratio is None:
            gate_2_sharpe = float(gate_2_metrics.get("sharpe") or 0.0)
            gate_3_sharpe = float(gate_3_metrics.get("mean_sharpe") or 0.0)
            overfit_ratio = abs(gate_2_sharpe) / (abs(gate_3_sharpe) + 0.01)
        return {
            "pair_a": str(pair_spec["pair_a"]),
            "pair_b": str(pair_spec["pair_b"]),
            "exchange": self.exchange,
            "timeframe": self.timeframe,
            "correlation": float(pair_spec.get("correlation") or 0.0),
            "half_life": float(pair_spec.get("half_life") or 0.0),
            "hedge_ratio": float(pair_spec.get("hedge_ratio") or 0.0),
            "lookback": int(params["lookback"]),
            "entry_z": float(params["entry_z"]),
            "exit_z": float(params["exit_z"]),
            "params": {
                "lookback": int(params["lookback"]),
                "entry_z": float(params["entry_z"]),
                "exit_z": float(params["exit_z"]),
            },
            "strategy_path": str(strategy_file.relative_to(ROOT)),
            "params_train": dict(adaptive_result.get("train") or {}),
            "params_validation": dict(gate_2_metrics),
            "walk_forward_validation": dict(gate_3_metrics),
            "params_overfit_ratio": float(overfit_ratio),
            "adaptive_result": dict(adaptive_result),
            "gate_result": {
                "final_gate_passed": gate_result.get("final_gate_passed"),
                "recommendation": gate_result.get("recommendation"),
            },
        }


__all__ = ["ContinuousRunner"]
