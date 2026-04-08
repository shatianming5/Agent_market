"""Strategy Lifecycle Manager — state machine for strategy promotion/retirement.

States: DISCOVERED → VALIDATED → PAPER → ACTIVE → RETIRED

Transitions:
  DISCOVERED → VALIDATED: passes walk-forward
  VALIDATED → PAPER:      enters paper trading pool
  PAPER → ACTIVE:         paper PnL positive for N days
  PAPER → RETIRED:        paper PnL negative for N days
  ACTIVE → RETIRED:       rolling Sharpe < threshold for N days
  any → RETIRED:          manual kill

Usage:
    from workspace.strategy_lifecycle import LifecycleManager
    lm = LifecycleManager()
    lm.register("my_strategy", strategy_type="pairs", config={...})
    lm.promote("my_strategy")  # DISCOVERED → VALIDATED
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]


class StrategyState(str, Enum):
    DISCOVERED = "discovered"   # just created, not yet validated
    VALIDATED = "validated"     # passed walk-forward
    PAPER = "paper"            # running in paper trading
    ACTIVE = "active"          # promoted to live/active
    RETIRED = "retired"        # decommissioned


VALID_TRANSITIONS = {
    StrategyState.DISCOVERED: [StrategyState.VALIDATED, StrategyState.RETIRED],
    StrategyState.VALIDATED: [StrategyState.PAPER, StrategyState.RETIRED],
    StrategyState.PAPER: [StrategyState.ACTIVE, StrategyState.RETIRED],
    StrategyState.ACTIVE: [StrategyState.RETIRED],
    StrategyState.RETIRED: [],
}


class LifecycleManager:
    """Manages strategy lifecycle state transitions."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = Path(db_path or ROOT / "workspace" / "results" / "lifecycle.json")
        self._strategies: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.db_path.exists():
            self._strategies = json.loads(self.db_path.read_text(encoding="utf-8"))
        changed = False
        for name, entry in list(self._strategies.items()):
            if self._normalize_entry(entry):
                changed = True
            self._strategies[name] = entry
        if changed:
            self._save()

    def _normalize_entry(self, entry: Dict[str, Any]) -> bool:
        changed = False
        if "paper_pnl" not in entry:
            entry["paper_pnl"] = []
            changed = True
        if "paper_days" not in entry:
            entry["paper_days"] = len(entry.get("paper_pnl") or [])
            changed = True
        if "active_days" not in entry:
            entry["active_days"] = 0
            changed = True
        if "rolling_sharpe" not in entry:
            entry["rolling_sharpe"] = []
            changed = True
        if "paper_dates" not in entry:
            entry["paper_dates"] = []
            changed = True
        if "paper_daily_equity" not in entry:
            entry["paper_daily_equity"] = {}
            changed = True
        if "paper_initial_equity" not in entry:
            entry["paper_initial_equity"] = 1000.0
            changed = True

        if entry.get("paper_daily_equity"):
            self._recompute_paper_metrics(entry)
            changed = True
        elif self._should_reset_legacy_paper_tracking(entry):
            entry["legacy_paper_tracking_reset_at"] = datetime.now(timezone.utc).isoformat()
            entry["legacy_paper_tracking_reset_reason"] = "legacy_loop_count_without_dates"
            entry["paper_days"] = 0
            entry["paper_pnl"] = []
            entry["paper_dates"] = []
            entry["paper_daily_equity"] = {}
            changed = True
        return changed

    def _should_reset_legacy_paper_tracking(self, entry: Dict[str, Any]) -> bool:
        paper_pnl = list(entry.get("paper_pnl") or [])
        paper_days = int(entry.get("paper_days") or 0)
        if not paper_pnl and paper_days <= 0:
            return False
        paper_started_at = self._state_started_at(entry, StrategyState.PAPER.value)
        if paper_started_at is None:
            return False
        now = datetime.now(timezone.utc)
        max_real_days = max(0, (now.date() - paper_started_at.date()).days + 1)
        return paper_days > max_real_days + 1 or len(paper_pnl) > max_real_days + 1

    def _state_started_at(self, entry: Dict[str, Any], state: str) -> Optional[datetime]:
        for row in reversed(entry.get("history") or []):
            if row.get("state") != state:
                continue
            raw = row.get("at")
            if not raw:
                return None
            try:
                parsed = datetime.fromisoformat(str(raw))
            except ValueError:
                return None
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
        return None

    def _recompute_paper_metrics(self, entry: Dict[str, Any]) -> None:
        daily_equity = {
            str(day): float(value)
            for day, value in sorted((entry.get("paper_daily_equity") or {}).items())
        }
        entry["paper_daily_equity"] = daily_equity
        entry["paper_dates"] = list(daily_equity.keys())
        prev_equity = float(entry.get("paper_initial_equity", 1000.0))
        pnl: List[float] = []
        for _, equity in daily_equity.items():
            daily_pct = 0.0 if prev_equity <= 0 else ((float(equity) / prev_equity) - 1.0) * 100.0
            pnl.append(round(float(daily_pct), 8))
            prev_equity = float(equity)
        entry["paper_pnl"] = pnl
        entry["paper_days"] = len(entry["paper_dates"])
        if daily_equity:
            last_day = entry["paper_dates"][-1]
            entry["paper_last_equity"] = float(daily_equity[last_day])

    def _save(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.write_text(
            json.dumps(self._strategies, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def register(
        self,
        name: str,
        *,
        strategy_type: str = "directional",
        config: Optional[Dict[str, Any]] = None,
        source: str = "manual",
    ) -> Dict[str, Any]:
        """Register a new strategy in DISCOVERED state."""
        now = datetime.now(timezone.utc).isoformat()
        entry = {
            "name": name,
            "state": StrategyState.DISCOVERED.value,
            "type": strategy_type,
            "config": config or {},
            "source": source,
            "created_at": now,
            "updated_at": now,
            "history": [{"state": "discovered", "at": now, "reason": "registered"}],
            "metrics": {},
            "paper_days": 0,
            "paper_pnl": [],
            "paper_dates": [],
            "paper_daily_equity": {},
            "paper_initial_equity": 1000.0,
            "active_days": 0,
            "rolling_sharpe": [],
        }
        self._strategies[name] = entry
        self._save()
        return entry

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self._strategies.get(name)

    def list_by_state(self, state: StrategyState) -> List[Dict[str, Any]]:
        return [s for s in self._strategies.values() if s["state"] == state.value]

    def transition(self, name: str, to_state: StrategyState, reason: str = "") -> bool:
        """Transition a strategy to a new state."""
        entry = self._strategies.get(name)
        if not entry:
            return False

        current = StrategyState(entry["state"])
        if to_state not in VALID_TRANSITIONS.get(current, []):
            return False

        now = datetime.now(timezone.utc).isoformat()
        entry["state"] = to_state.value
        entry["updated_at"] = now
        entry["history"].append({"state": to_state.value, "at": now, "reason": reason})
        self._save()
        return True

    def promote(self, name: str, reason: str = "manual") -> bool:
        """Promote to next state: DISCOVERED→VALIDATED→PAPER→ACTIVE."""
        entry = self._strategies.get(name)
        if not entry:
            return False
        current = StrategyState(entry["state"])
        next_map = {
            StrategyState.DISCOVERED: StrategyState.VALIDATED,
            StrategyState.VALIDATED: StrategyState.PAPER,
            StrategyState.PAPER: StrategyState.ACTIVE,
        }
        next_state = next_map.get(current)
        if not next_state:
            return False
        return self.transition(name, next_state, reason)

    def retire(self, name: str, reason: str = "manual") -> bool:
        """Retire a strategy from any state."""
        return self.transition(name, StrategyState.RETIRED, reason)

    def sync_paper_tracking(
        self,
        name: str,
        *,
        daily_equity: Dict[str, float],
        last_processed_at: str = "",
        current_equity: Optional[float] = None,
        state_path: Optional[str] = None,
    ) -> None:
        """Persist real forward paper-trading snapshots keyed by UTC day."""
        entry = self._strategies.get(name)
        if not entry or entry["state"] not in {StrategyState.PAPER.value, StrategyState.ACTIVE.value}:
            return
        merged = {str(day): float(value) for day, value in (entry.get("paper_daily_equity") or {}).items()}
        merged.update({str(day): float(value) for day, value in daily_equity.items()})
        entry["paper_daily_equity"] = dict(sorted(merged.items()))
        if current_equity is not None:
            entry["paper_last_equity"] = float(current_equity)
        if last_processed_at:
            entry["paper_last_processed_at"] = str(last_processed_at)
        if state_path:
            entry["paper_state_path"] = str(state_path)
        self._recompute_paper_metrics(entry)
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def record_paper_day(self, name: str, pnl_pct: float, *, as_of: Optional[str] = None) -> None:
        """Back-compat wrapper for day-level paper tracking."""
        entry = self._strategies.get(name)
        if not entry or entry["state"] != StrategyState.PAPER.value:
            return
        stamp = as_of or datetime.now(timezone.utc).isoformat()
        day = datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc).date().isoformat()
        daily_equity = dict(entry.get("paper_daily_equity") or {})
        base = float(entry.get("paper_initial_equity", 1000.0))
        previous_dates = sorted(daily_equity)
        previous_equity = float(daily_equity[previous_dates[-1]]) if previous_dates else base
        daily_equity[day] = previous_equity * (1.0 + (float(pnl_pct) / 100.0))
        self.sync_paper_tracking(
            name,
            daily_equity=daily_equity,
            last_processed_at=stamp,
            current_equity=daily_equity[day],
            state_path=entry.get("paper_state_path"),
        )

    def record_active_day(self, name: str, sharpe: float) -> None:
        """Record an active trading day's rolling Sharpe."""
        entry = self._strategies.get(name)
        if not entry or entry["state"] != StrategyState.ACTIVE.value:
            return
        entry["active_days"] += 1
        entry["rolling_sharpe"].append(sharpe)
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def auto_review(
        self,
        *,
        paper_promote_days: int = 7,
        paper_retire_days: int = 7,
        active_retire_days: int = 5,
        active_sharpe_threshold: float = 0.0,
        health_issues: Optional[List[Dict[str, Any]]] = None,
        paper_overfit_ratio_max: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """Auto-review all strategies and apply transitions.

        Returns list of actions taken.
        """
        actions = []
        health_by_strategy = {
            str(issue.get("strategy")): issue
            for issue in (health_issues or [])
            if issue.get("strategy")
        }

        # Paper → Active: if profitable for N consecutive days
        for entry in self.list_by_state(StrategyState.PAPER):
            name = str(entry["name"])
            health_issue = health_by_strategy.get(name) or {}
            details = [str(item) for item in (health_issue.get("details") or [])]
            if any("cointegration lost" in item.lower() for item in details):
                if self.retire(name, "paper health failed: cointegration lost"):
                    actions.append({"action": "retire", "name": name, "reason": "cointegration lost"})
                continue

            config = entry.get("config") or {}
            validation = config.get("params_validation") or {}
            overfit_ratio = config.get("params_overfit_ratio")
            validation_sharpe = validation.get("sharpe")
            validation_trades = validation.get("trades")
            if validation_trades is not None and float(validation_trades) <= 0:
                if self.retire(name, "paper guardrail: validation trades <= 0"):
                    actions.append({"action": "retire", "name": name, "reason": "validation trades <= 0"})
                continue
            if validation_sharpe is not None and float(validation_sharpe) <= 0:
                if self.retire(name, f"paper guardrail: validation sharpe <= 0 ({validation_sharpe})"):
                    actions.append({"action": "retire", "name": name, "reason": "validation sharpe <= 0"})
                continue
            if overfit_ratio is not None and float(overfit_ratio) > paper_overfit_ratio_max:
                if self.retire(name, f"paper guardrail: overfit ratio > {paper_overfit_ratio_max}"):
                    actions.append({"action": "retire", "name": name, "reason": "overfit ratio too high"})
                continue

            pnl = entry.get("paper_pnl", [])
            if len(pnl) >= paper_promote_days:
                recent = pnl[-paper_promote_days:]
                if sum(recent) > 0 and sum(1 for p in recent if p > 0) >= paper_promote_days * 0.6:
                    self.promote(name, f"paper profitable {paper_promote_days} days")
                    actions.append({"action": "promote", "name": name, "to": "active"})
                elif len(pnl) >= paper_retire_days and sum(pnl[-paper_retire_days:]) < 0:
                    self.retire(name, f"paper unprofitable {paper_retire_days} days")
                    actions.append({"action": "retire", "name": name, "reason": "paper loss"})

        # Active → Retired: if rolling Sharpe < threshold for N days
        for entry in self.list_by_state(StrategyState.ACTIVE):
            sharpes = entry.get("rolling_sharpe", [])
            if len(sharpes) >= active_retire_days:
                recent = sharpes[-active_retire_days:]
                if all(s < active_sharpe_threshold for s in recent):
                    self.retire(entry["name"], f"Sharpe < {active_sharpe_threshold} for {active_retire_days} days")
                    actions.append({"action": "retire", "name": entry["name"], "reason": "sharpe decay"})

        return actions

    def summary(self) -> Dict[str, Any]:
        """Summary of all strategies by state."""
        counts = {}
        for state in StrategyState:
            items = self.list_by_state(state)
            counts[state.value] = len(items)
        return {
            "total": len(self._strategies),
            "by_state": counts,
            "strategies": [
                {"name": s["name"], "state": s["state"], "type": s["type"],
                 "paper_days": s.get("paper_days", 0), "active_days": s.get("active_days", 0)}
                for s in self._strategies.values()
            ],
        }


__all__ = ["LifecycleManager", "StrategyState"]
