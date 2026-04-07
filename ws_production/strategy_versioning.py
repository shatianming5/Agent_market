"""Strategy Version Manager — tracks every parameter change with git-like history.

Usage:
    from workspace.strategy_versioning import VersionManager
    vm = VersionManager()
    vm.save_version("doge_sol_pairs", params={...}, metrics={...}, notes="initial")
    history = vm.get_history("doge_sol_pairs")
    diff = vm.diff("doge_sol_pairs", v1=1, v2=3)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]


class VersionManager:
    """Git-like version tracking for strategy parameters and performance."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = Path(db_path or ROOT / "workspace" / "results" / "versions.json")
        self._versions: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    def _load(self):
        if self.db_path.exists():
            self._versions = json.loads(self.db_path.read_text(encoding="utf-8"))

    def _save(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.write_text(json.dumps(self._versions, indent=2, ensure_ascii=False, default=str))

    def _hash_params(self, params: Dict) -> str:
        return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]

    def save_version(
        self,
        strategy_name: str,
        *,
        params: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        notes: str = "",
        source: str = "manual",
    ) -> Dict[str, Any]:
        """Save a new version of a strategy."""
        if strategy_name not in self._versions:
            self._versions[strategy_name] = []

        history = self._versions[strategy_name]
        version = len(history) + 1
        param_hash = self._hash_params(params)

        # Check if params actually changed
        if history:
            last_hash = history[-1].get("param_hash", "")
            if last_hash == param_hash:
                return history[-1]  # no change

        entry = {
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": params,
            "param_hash": param_hash,
            "metrics": metrics or {},
            "notes": notes,
            "source": source,
        }
        history.append(entry)
        self._save()
        return entry

    def get_history(self, strategy_name: str) -> List[Dict[str, Any]]:
        """Get full version history for a strategy."""
        return self._versions.get(strategy_name, [])

    def get_latest(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get latest version."""
        history = self.get_history(strategy_name)
        return history[-1] if history else None

    def get_version(self, strategy_name: str, version: int) -> Optional[Dict[str, Any]]:
        """Get specific version."""
        history = self.get_history(strategy_name)
        for v in history:
            if v["version"] == version:
                return v
        return None

    def diff(self, strategy_name: str, v1: int, v2: int) -> Dict[str, Any]:
        """Compare two versions — show what parameters changed."""
        ver1 = self.get_version(strategy_name, v1)
        ver2 = self.get_version(strategy_name, v2)
        if not ver1 or not ver2:
            return {"error": f"version {v1} or {v2} not found"}

        p1 = ver1.get("params", {})
        p2 = ver2.get("params", {})
        m1 = ver1.get("metrics", {})
        m2 = ver2.get("metrics", {})

        param_changes = {}
        all_keys = set(list(p1.keys()) + list(p2.keys()))
        for k in all_keys:
            old = p1.get(k)
            new = p2.get(k)
            if old != new:
                param_changes[k] = {"v1": old, "v2": new}

        metric_changes = {}
        all_m_keys = set(list(m1.keys()) + list(m2.keys()))
        for k in all_m_keys:
            old = m1.get(k)
            new = m2.get(k)
            if old != new:
                metric_changes[k] = {"v1": old, "v2": new}

        return {
            "strategy": strategy_name,
            "v1": v1, "v2": v2,
            "param_changes": param_changes,
            "metric_changes": metric_changes,
            "params_changed": len(param_changes) > 0,
        }

    def rollback(self, strategy_name: str, to_version: int) -> Optional[Dict[str, Any]]:
        """Create a new version by rolling back to a previous one."""
        old = self.get_version(strategy_name, to_version)
        if not old:
            return None
        return self.save_version(
            strategy_name,
            params=old["params"],
            notes=f"rollback to v{to_version}",
            source="rollback",
        )

    def summary(self) -> Dict[str, Any]:
        """Summary of all versioned strategies."""
        return {
            name: {
                "total_versions": len(history),
                "latest_version": history[-1]["version"] if history else 0,
                "latest_hash": history[-1].get("param_hash", "") if history else "",
                "created": history[0]["timestamp"] if history else "",
                "updated": history[-1]["timestamp"] if history else "",
            }
            for name, history in self._versions.items()
        }


__all__ = ["VersionManager"]
