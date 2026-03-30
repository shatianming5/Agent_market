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

    def record_paper_day(self, name: str, pnl_pct: float) -> None:
        """Record a paper trading day result."""
        entry = self._strategies.get(name)
        if not entry or entry["state"] != StrategyState.PAPER.value:
            return
        entry["paper_days"] += 1
        entry["paper_pnl"].append(pnl_pct)
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save()

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
    ) -> List[Dict[str, Any]]:
        """Auto-review all strategies and apply transitions.

        Returns list of actions taken.
        """
        actions = []

        # Paper → Active: if profitable for N consecutive days
        for entry in self.list_by_state(StrategyState.PAPER):
            pnl = entry.get("paper_pnl", [])
            if len(pnl) >= paper_promote_days:
                recent = pnl[-paper_promote_days:]
                if sum(recent) > 0 and sum(1 for p in recent if p > 0) >= paper_promote_days * 0.6:
                    self.promote(entry["name"], f"paper profitable {paper_promote_days} days")
                    actions.append({"action": "promote", "name": entry["name"], "to": "active"})
                elif len(pnl) >= paper_retire_days and sum(pnl[-paper_retire_days:]) < 0:
                    self.retire(entry["name"], f"paper unprofitable {paper_retire_days} days")
                    actions.append({"action": "retire", "name": entry["name"], "reason": "paper loss"})

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
