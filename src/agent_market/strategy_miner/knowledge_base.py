"""Knowledge base for elite strategies and failure patterns.

Upgraded schema (D7):
- strategy_cards: rich provenance + indicators + metrics
- failure_cards: structured failure taxonomy (D6)
- edges: lineage graph (parent → child)
- Backward compatible: loads old {elites, failures} format transparently.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_MAX_ELITE = 20
_MAX_FAILURES = 50
_MAX_STRATEGY_CARDS = 100
_MAX_FAILURE_CARDS = 100


def _extract_indicator_names(code: str) -> Set[str]:
    """Best-effort extraction of indicator names from strategy code."""
    indicators: Set[str] = set()
    # Match ta.xxx() and pta.xxx() calls
    for m in re.finditer(r"(?:ta|pta)\.(\w+)\s*\(", code):
        indicators.add(m.group(1).lower())
    # Match common indicator references like 'ema', 'rsi', 'macd', etc.
    for kw in ("ema", "sma", "rsi", "macd", "bbands", "atr", "adx", "cci",
               "stoch", "willr", "obv", "mfi", "keltner", "donchian", "vwap",
               "bollinger", "supertrend", "ichimoku", "aroon", "psar"):
        if re.search(rf'\b{kw}\b', code, re.IGNORECASE):
            indicators.add(kw)
    return indicators


def _indicator_overlap(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard similarity between two indicator sets."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


class KnowledgeBase:
    """Persistent store for elite strategies, failure patterns, and lineage.

    Schema v2 stores {elites, failures, strategy_cards, failure_cards, edges}.
    Loads v1 (elites+failures only) transparently.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._elites: List[Dict[str, Any]] = []
        self._failures: List[Dict[str, Any]] = []
        # D7: New collections
        self._strategy_cards: List[Dict[str, Any]] = []
        self._failure_cards: List[Dict[str, Any]] = []
        self._edges: List[Dict[str, Any]] = []  # {parent, child, mutation_type}
        if path.exists():
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._elites = data.get("elites", [])
            self._failures = data.get("failures", [])
            self._strategy_cards = data.get("strategy_cards", [])
            self._failure_cards = data.get("failure_cards", [])
            self._edges = data.get("edges", [])
        except Exception as e:
            logger.warning("Failed to load knowledge base from %s: %s", self._path, e)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(
            {
                "elites": self._elites,
                "failures": self._failures,
                "strategy_cards": self._strategy_cards,
                "failure_cards": self._failure_cards,
                "edges": self._edges,
            },
            ensure_ascii=False,
            indent=2,
        )
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(data, encoding="utf-8")
        tmp.rename(self._path)

    # -----------------------------------------------------------------------
    # Legacy API (unchanged)
    # -----------------------------------------------------------------------

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
            "strategy_cards_count": len(self._strategy_cards),
            "failure_cards_count": len(self._failure_cards),
            "edges_count": len(self._edges),
            "top_reward": self._elites[0]["reward"] if self._elites else None,
            "elites": self._elites,
            "failures": self._failures,
        }

    # -----------------------------------------------------------------------
    # D7: Strategy Cards
    # -----------------------------------------------------------------------

    def add_strategy_card(
        self,
        name: str,
        code: str,
        iteration: int,
        *,
        candidate_type: str = "rule",
        candidate_family: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        parent_name: str = "",
        mutation_description: str = "",
    ) -> None:
        """Add a rich strategy card with provenance and indicators."""
        indicators = sorted(_extract_indicator_names(code))
        card: Dict[str, Any] = {
            "name": name,
            "iteration": iteration,
            "candidate_type": candidate_type,
            "candidate_family": candidate_family,
            "indicators": indicators,
            "metrics": metrics or {},
            "parent_name": parent_name,
            "mutation_description": mutation_description,
        }
        self._strategy_cards.append(card)
        self._strategy_cards = self._strategy_cards[-_MAX_STRATEGY_CARDS:]
        # Record lineage edge if parent exists
        if parent_name:
            self._edges.append({
                "parent": parent_name,
                "child": name,
                "mutation_type": mutation_description[:100] or "derived",
            })
        self.save()

    @property
    def strategy_cards(self) -> List[Dict[str, Any]]:
        return list(self._strategy_cards)

    # -----------------------------------------------------------------------
    # D6: Failure Cards (structured taxonomy)
    # -----------------------------------------------------------------------

    def add_failure_card(
        self,
        name: str,
        iteration: int,
        *,
        phase: str = "",
        category: str = "",
        subcategory: str = "",
        detail: str = "",
        fix_applied: str = "",
        attempt: int = 0,
    ) -> None:
        """Add a structured failure card (D6: failure taxonomy)."""
        card: Dict[str, Any] = {
            "name": name,
            "iteration": iteration,
            "phase": phase,
            "category": category,
            "subcategory": subcategory,
            "detail": detail[:500],
            "fix_applied": fix_applied[:200],
            "attempt": attempt,
        }
        self._failure_cards.append(card)
        self._failure_cards = self._failure_cards[-_MAX_FAILURE_CARDS:]
        self.save()

    @property
    def failure_cards(self) -> List[Dict[str, Any]]:
        return list(self._failure_cards)

    # -----------------------------------------------------------------------
    # D7: Similarity-based retrieval
    # -----------------------------------------------------------------------

    def query_similar_strategies(
        self, code: str, n: int = 3, *, exclude_names: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-N strategy cards most similar to the given code by indicator overlap."""
        target_indicators = _extract_indicator_names(code)
        if not target_indicators:
            return self._strategy_cards[:n]
        exclude = exclude_names or set()
        scored = []
        for card in self._strategy_cards:
            if card.get("name") in exclude:
                continue
            card_indicators = set(card.get("indicators", []))
            sim = _indicator_overlap(target_indicators, card_indicators)
            scored.append((sim, card))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [card for _, card in scored[:n]]

    def query_similar_failures(
        self, code: str, n: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return failure cards for strategies with similar indicators."""
        target_indicators = _extract_indicator_names(code)
        if not target_indicators or not self._failure_cards:
            return self._failure_cards[-n:]
        # Match by name against strategy_cards to find indicators
        name_to_indicators: Dict[str, Set[str]] = {}
        for card in self._strategy_cards:
            name_to_indicators[card["name"]] = set(card.get("indicators", []))
        scored = []
        for fcard in self._failure_cards:
            fname = fcard.get("name", "")
            fi = name_to_indicators.get(fname, set())
            sim = _indicator_overlap(target_indicators, fi) if fi else 0.0
            scored.append((sim, fcard))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [card for _, card in scored[:n]]

    # -----------------------------------------------------------------------
    # D7: Lineage graph queries
    # -----------------------------------------------------------------------

    @property
    def edges(self) -> List[Dict[str, Any]]:
        return list(self._edges)

    def ancestors(self, name: str, max_depth: int = 5) -> List[str]:
        """Return ancestor names up the lineage chain."""
        result: List[str] = []
        current = name
        for _ in range(max_depth):
            parent = None
            for edge in self._edges:
                if edge.get("child") == current:
                    parent = edge.get("parent")
                    break
            if parent is None:
                break
            result.append(parent)
            current = parent
        return result
