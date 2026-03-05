"""Knowledge base for elite strategies and failure patterns."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_ELITE = 20
_MAX_FAILURES = 50


class KnowledgeBase:
    """Persistent store for elite strategies and failure patterns.

    Stored as a single JSON file in the run directory.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._elites: List[Dict[str, Any]] = []
        self._failures: List[Dict[str, Any]] = []
        if path.exists():
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._elites = data.get("elites", [])
            self._failures = data.get("failures", [])
        except Exception as e:
            logger.warning("Failed to load knowledge base from %s: %s", self._path, e)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(
            {"elites": self._elites, "failures": self._failures},
            ensure_ascii=False,
            indent=2,
        )
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(data, encoding="utf-8")
        tmp.rename(self._path)

    def add_elite(
        self,
        name: str,
        code: str,
        reward: float,
        backtest_summary: Dict[str, Any],
        iteration: int,
    ) -> None:
        entry = {
            "name": name,
            "reward": reward,
            "iteration": iteration,
            "profit_pct": backtest_summary.get("profit_total_pct"),
            "trades": backtest_summary.get("trades"),
            "winrate": backtest_summary.get("winrate"),
            "max_drawdown": backtest_summary.get("max_drawdown_abs"),
            "code_snippet": code[:2000],
        }
        self._elites.append(entry)
        # Keep sorted by reward descending, trim to max
        self._elites.sort(key=lambda e: e.get("reward", 0), reverse=True)
        self._elites = self._elites[:_MAX_ELITE]
        self.save()

    def add_failure(
        self,
        name: str,
        iteration: int,
        failure_type: str,
        detail: str,
    ) -> None:
        entry = {
            "name": name,
            "iteration": iteration,
            "failure_type": failure_type,
            "detail": detail[:500],
        }
        self._failures.append(entry)
        self._failures = self._failures[-_MAX_FAILURES:]
        self.save()

    @property
    def elites(self) -> List[Dict[str, Any]]:
        return list(self._elites)

    @property
    def failures(self) -> List[Dict[str, Any]]:
        return list(self._failures)

    def top_elite_codes(self, n: int = 3) -> List[str]:
        return [e.get("code_snippet", "") for e in self._elites[:n]]

    def failure_summary(self, n: int = 10) -> str:
        if not self._failures:
            return "No recorded failures."
        recent = self._failures[-n:]
        lines = []
        for f in recent:
            lines.append(
                f"  - iter{f.get('iteration', '?')} [{f.get('failure_type', '?')}]: "
                f"{f.get('detail', '')[:100]}"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elites_count": len(self._elites),
            "failures_count": len(self._failures),
            "top_reward": self._elites[0]["reward"] if self._elites else None,
            "elites": self._elites,
            "failures": self._failures,
        }
