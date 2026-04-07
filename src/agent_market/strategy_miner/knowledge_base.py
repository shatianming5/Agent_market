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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_MAX_ELITE = 20
_MAX_FAILURES = 50
_MAX_STRATEGY_CARDS = 500
_MAX_FAILURE_CARDS = 300


@dataclass
class StrategyRetrievalResult:
    context: str
    query: Dict[str, Any]
    strategy_cards: List[Dict[str, Any]]
    failure_cards: List[Dict[str, Any]]


def _infer_run_id_hint(path: Path) -> str:
    try:
        if path.name == "knowledge_base.json" and path.parent.name == "strategy_miner":
            return str(path.parent.parent.name or "")
    except Exception:
        return ""
    return ""


def _normalize_strategy_card(
    card: Dict[str, Any],
    *,
    run_id_hint: str = "",
) -> Optional[Dict[str, Any]]:
    name = str(card.get("name") or "").strip()
    if not name:
        return None

    iteration = int(card.get("iteration") or 0)
    run_id = str(card.get("run_id") or run_id_hint or "run").strip()
    metrics = dict(card.get("metrics") or {})
    refs = [
        dict(ref)
        for ref in list(card.get("downstream_strategy_references") or [])
        if isinstance(ref, dict)
    ]
    indicators = sorted(
        {
            str(item or "").strip().lower()
            for item in list(card.get("indicators") or [])
            if str(item or "").strip()
        }
    )
    return {
        "card_id": str(card.get("card_id") or f"{run_id}:{name}:{iteration}"),
        "created_at": str(card.get("created_at") or ""),
        "run_id": run_id,
        "name": name,
        "iteration": iteration,
        "candidate_type": str(card.get("candidate_type") or "rule"),
        "candidate_family": str(card.get("candidate_family") or ""),
        "timeframe": str(card.get("timeframe") or ""),
        "universe": list(card.get("universe") or []),
        "indicators": indicators,
        "metrics": metrics,
        "parent_name": str(card.get("parent_name") or ""),
        "mutation_description": str(card.get("mutation_description") or ""),
        "reuse_count": int(card.get("reuse_count", 0) or 0),
        "downstream_strategy_references": refs[-50:],
    }


def _normalize_failure_card(
    card: Dict[str, Any],
    *,
    run_id_hint: str = "",
) -> Optional[Dict[str, Any]]:
    name = str(card.get("name") or "").strip()
    if not name:
        return None

    iteration = int(card.get("iteration") or 0)
    run_id = str(card.get("run_id") or run_id_hint or "run").strip()
    category = str(card.get("category") or "")
    subcategory = str(card.get("subcategory") or "")
    failure_id = str(
        card.get("failure_id") or f"{run_id}:{name}:{iteration}:{category or subcategory or 'failure'}"
    )
    return {
        "failure_id": failure_id,
        "run_id": run_id,
        "name": name,
        "iteration": iteration,
        "phase": str(card.get("phase") or ""),
        "category": category,
        "subcategory": subcategory,
        "candidate_family": str(card.get("candidate_family") or ""),
        "candidate_type": str(card.get("candidate_type") or "rule"),
        "timeframe": str(card.get("timeframe") or ""),
        "universe": list(card.get("universe") or []),
        "detail": str(card.get("detail") or "")[:500],
        "fix_applied": str(card.get("fix_applied") or "")[:200],
        "attempt": int(card.get("attempt", 0) or 0),
    }


def _normalize_edge(edge: Dict[str, Any], *, run_id_hint: str = "") -> Optional[Dict[str, Any]]:
    parent = str(edge.get("parent") or "").strip()
    child = str(edge.get("child") or "").strip()
    if not parent or not child:
        return None
    return {
        "parent": parent,
        "child": child,
        "mutation_type": str(edge.get("mutation_type") or ""),
        "edge_type": str(edge.get("edge_type") or "lineage"),
        "run_id": str(edge.get("run_id") or run_id_hint or ""),
    }


def _strategy_card_rank(card: Dict[str, Any]) -> float:
    metrics = card.get("metrics") or {}
    score = 0.0
    sharpe = metrics.get("sharpe")
    profit = metrics.get("profit_pct")
    trades = metrics.get("trades")
    if isinstance(sharpe, (int, float)):
        score += float(sharpe) * 3.0
    if isinstance(profit, (int, float)):
        score += float(profit) * 0.2
    if isinstance(trades, (int, float)):
        score += min(float(trades) / 50.0, 2.0)
    score += min(int(card.get("reuse_count", 0) or 0), 5) * 0.2
    return score


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


def _base_assets(universe: List[str]) -> List[str]:
    assets: List[str] = []
    for raw in universe:
        text = str(raw or "").strip().upper()
        if not text:
            continue
        pair = text.split(":")[0]
        base = pair.split("/")[0]
        if base and base not in assets:
            assets.append(base)
    return assets


def _text_blob(*parts: Any) -> str:
    return " ".join(str(item or "") for item in parts).strip().lower()


def _family_match_tier(requested_family: str, card_family: str) -> int:
    requested = _text_blob(requested_family)
    candidate = _text_blob(card_family)
    if not requested:
        return 0
    if requested == candidate:
        return 3

    requested_parts = [part for part in requested.split("/") if part]
    candidate_parts = [part for part in candidate.split("/") if part]
    requested_type = requested_parts[0] if requested_parts else ""
    candidate_type = candidate_parts[0] if candidate_parts else ""
    requested_leaf = requested_parts[-1] if requested_parts else ""

    if requested_leaf and requested_leaf in candidate:
        return 2
    if requested_type and requested_type == candidate_type:
        return 1
    return 0


def format_strategy_retrieval_context(
    *,
    strategy_cards: List[Dict[str, Any]],
    failure_cards: List[Dict[str, Any]],
) -> str:
    if not strategy_cards and not failure_cards:
        return ""

    lines: List[str] = [
        "## Strategy Memory Retrieval",
        "Use these prior strategy outcomes as reusable patterns or guardrails. Recombine them; do not clone them.",
    ]
    if strategy_cards:
        lines.append("### Retrieved strategy cards")
        for card in strategy_cards[:4]:
            metrics = card.get("metrics") or {}
            lines.append(
                "- {name} [{card_id}] family={family} type={ctype} sharpe={sharpe} "
                "profit={profit}% trades={trades} reuse_count={reuse}".format(
                    name=card.get("name") or "?",
                    card_id=card.get("card_id") or "?",
                    family=card.get("candidate_family") or "?",
                    ctype=card.get("candidate_type") or "?",
                    sharpe=metrics.get("sharpe"),
                    profit=metrics.get("profit_pct"),
                    trades=metrics.get("trades"),
                    reuse=card.get("reuse_count", 0),
                )
            )
            mutation = str(card.get("mutation_description") or "").strip()
            if mutation:
                lines.append(f"  mutation: {mutation[:160]}")
    if failure_cards:
        lines.append("### Failure cards to avoid")
        for card in failure_cards[:3]:
            lines.append(
                "- {name} [{failure_id}] family={family} category={category}; detail={detail}".format(
                    name=card.get("name") or "?",
                    failure_id=card.get("failure_id") or "?",
                    family=card.get("candidate_family") or "?",
                    category=card.get("category") or card.get("subcategory") or "failure",
                    detail=str(card.get("detail") or "")[:180],
                )
            )
    return "\n".join(lines)


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
            self._elites = []
            self._failures = []
            self._strategy_cards = []
            self._failure_cards = []
            self._edges = []
            self.merge_payload(data, run_id_hint=_infer_run_id_hint(self._path), save=False)
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

    def merge_payload(
        self,
        payload: Dict[str, Any],
        *,
        run_id_hint: str = "",
        save: bool = True,
    ) -> None:
        changed = False

        elite_keys = {
            (
                str(item.get("name") or ""),
                int(item.get("iteration") or 0),
                str(item.get("failure_type") or ""),
                str(item.get("detail") or ""),
            )
            for item in self._failures
            if isinstance(item, dict)
        }
        failure_keys = set(elite_keys)
        elite_entry_keys = {
            (
                str(item.get("name") or ""),
                int(item.get("iteration") or 0),
                float(item.get("reward") or 0.0),
                float(item.get("profit_pct") or 0.0),
                float(item.get("trades") or 0.0),
            )
            for item in self._elites
            if isinstance(item, dict)
        }
        strategy_index = {
            str(card.get("card_id") or ""): idx
            for idx, card in enumerate(self._strategy_cards)
            if isinstance(card, dict) and str(card.get("card_id") or "")
        }
        failure_index = {
            str(card.get("failure_id") or ""): idx
            for idx, card in enumerate(self._failure_cards)
            if isinstance(card, dict) and str(card.get("failure_id") or "")
        }
        edge_keys = {
            (
                str(edge.get("parent") or ""),
                str(edge.get("child") or ""),
                str(edge.get("edge_type") or ""),
                str(edge.get("run_id") or ""),
            )
            for edge in self._edges
            if isinstance(edge, dict)
        }

        for item in list(payload.get("elites") or []):
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            key = (
                name,
                int(entry.get("iteration") or 0),
                float(entry.get("reward") or 0.0),
                float(entry.get("profit_pct") or 0.0),
                float(entry.get("trades") or 0.0),
            )
            if key in elite_entry_keys:
                continue
            self._elites.append(entry)
            elite_entry_keys.add(key)
            changed = True

        if self._elites:
            self._elites.sort(key=lambda item: float(item.get("reward") or 0.0), reverse=True)
            self._elites = self._elites[:_MAX_ELITE]

        for item in list(payload.get("failures") or []):
            if not isinstance(item, dict):
                continue
            entry = {
                "name": str(item.get("name") or ""),
                "iteration": int(item.get("iteration") or 0),
                "failure_type": str(item.get("failure_type") or ""),
                "detail": str(item.get("detail") or "")[:500],
            }
            if not entry["name"]:
                continue
            key = (
                entry["name"],
                entry["iteration"],
                entry["failure_type"],
                entry["detail"],
            )
            if key in failure_keys:
                continue
            self._failures.append(entry)
            failure_keys.add(key)
            changed = True

        if self._failures:
            self._failures = self._failures[-_MAX_FAILURES:]

        for raw_card in list(payload.get("strategy_cards") or []):
            if not isinstance(raw_card, dict):
                continue
            card = _normalize_strategy_card(raw_card, run_id_hint=run_id_hint)
            if card is None:
                continue
            card_id = str(card.get("card_id") or "")
            existing_idx = strategy_index.get(card_id)
            if existing_idx is None:
                self._strategy_cards.append(card)
                strategy_index[card_id] = len(self._strategy_cards) - 1
                changed = True
                continue

            existing = dict(self._strategy_cards[existing_idx])
            merged = dict(existing)
            merged.update(card)
            merged["reuse_count"] = max(
                int(existing.get("reuse_count", 0) or 0),
                int(card.get("reuse_count", 0) or 0),
            )
            refs = list(existing.get("downstream_strategy_references") or [])
            for ref in list(card.get("downstream_strategy_references") or []):
                if ref not in refs:
                    refs.append(ref)
            merged["downstream_strategy_references"] = refs[-50:]
            if merged != existing:
                self._strategy_cards[existing_idx] = merged
                changed = True

        if self._strategy_cards:
            self._strategy_cards.sort(key=_strategy_card_rank, reverse=True)
            self._strategy_cards = self._strategy_cards[:_MAX_STRATEGY_CARDS]

        for raw_card in list(payload.get("failure_cards") or []):
            if not isinstance(raw_card, dict):
                continue
            card = _normalize_failure_card(raw_card, run_id_hint=run_id_hint)
            if card is None:
                continue
            failure_id = str(card.get("failure_id") or "")
            existing_idx = failure_index.get(failure_id)
            if existing_idx is None:
                self._failure_cards.append(card)
                failure_index[failure_id] = len(self._failure_cards) - 1
                changed = True
                continue
            existing = dict(self._failure_cards[existing_idx])
            if card != existing:
                self._failure_cards[existing_idx] = card
                changed = True

        if self._failure_cards:
            self._failure_cards = self._failure_cards[-_MAX_FAILURE_CARDS:]

        for raw_edge in list(payload.get("edges") or []):
            if not isinstance(raw_edge, dict):
                continue
            edge = _normalize_edge(raw_edge, run_id_hint=run_id_hint)
            if edge is None:
                continue
            edge_key = (
                str(edge.get("parent") or ""),
                str(edge.get("child") or ""),
                str(edge.get("edge_type") or ""),
                str(edge.get("run_id") or ""),
            )
            if edge_key in edge_keys:
                continue
            self._edges.append(edge)
            edge_keys.add(edge_key)
            changed = True

        if changed and save:
            self.save()

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
        run_id: str = "",
        timeframe: str = "",
        universe: Optional[List[str]] = None,
    ) -> None:
        """Add a rich strategy card with provenance and indicators."""
        indicators = sorted(_extract_indicator_names(code))
        card_id = f"{run_id or 'run'}:{name}:{iteration}"
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        card: Dict[str, Any] = {
            "card_id": card_id,
            "created_at": created_at,
            "run_id": run_id,
            "name": name,
            "iteration": iteration,
            "candidate_type": candidate_type,
            "candidate_family": candidate_family,
            "timeframe": timeframe,
            "universe": list(universe or []),
            "indicators": indicators,
            "metrics": metrics or {},
            "parent_name": parent_name,
            "mutation_description": mutation_description,
            "reuse_count": 0,
            "downstream_strategy_references": [],
        }
        existing_idx = next(
            (idx for idx, item in enumerate(self._strategy_cards) if str(item.get("card_id") or "") == card_id),
            None,
        )
        if existing_idx is not None:
            existing = dict(self._strategy_cards[existing_idx])
            if existing.get("created_at"):
                card["created_at"] = existing.get("created_at")
            card["reuse_count"] = int(existing.get("reuse_count", 0) or 0)
            card["downstream_strategy_references"] = list(existing.get("downstream_strategy_references") or [])
            self._strategy_cards[existing_idx] = card
        else:
            self._strategy_cards.append(card)
            self._strategy_cards = self._strategy_cards[-_MAX_STRATEGY_CARDS:]
        # Record lineage edge if parent exists
        if parent_name:
            edge = {
                "parent": parent_name,
                "child": name,
                "mutation_type": mutation_description[:100] or "derived",
                "edge_type": "lineage",
                "run_id": run_id,
            }
            if edge not in self._edges:
                self._edges.append(edge)
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
        run_id: str = "",
        candidate_family: str = "",
        candidate_type: str = "",
        timeframe: str = "",
        universe: Optional[List[str]] = None,
    ) -> None:
        """Add a structured failure card (D6: failure taxonomy)."""
        failure_id = f"{run_id or 'run'}:{name}:{iteration}:{category or subcategory or 'failure'}"
        card: Dict[str, Any] = {
            "failure_id": failure_id,
            "run_id": run_id,
            "name": name,
            "iteration": iteration,
            "phase": phase,
            "category": category,
            "subcategory": subcategory,
            "candidate_family": candidate_family,
            "candidate_type": candidate_type,
            "timeframe": timeframe,
            "universe": list(universe or []),
            "detail": detail[:500],
            "fix_applied": fix_applied[:200],
            "attempt": attempt,
        }
        existing_idx = next(
            (idx for idx, item in enumerate(self._failure_cards) if str(item.get("failure_id") or "") == failure_id),
            None,
        )
        if existing_idx is not None:
            self._failure_cards[existing_idx] = card
        else:
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

    def query_strategy_cards_for_generation(
        self,
        *,
        top_n: int = 3,
        family: str = "",
        timeframe: str = "",
        universe: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        requested_assets = set(_base_assets(list(universe or [])))
        family_blob = _text_blob(family)
        results: List[tuple[int, float, Dict[str, Any]]] = []
        for card in self._strategy_cards:
            metrics = card.get("metrics") or {}
            score = 0.0
            sharpe = metrics.get("sharpe")
            profit = metrics.get("profit_pct")
            trades = metrics.get("trades")
            if isinstance(sharpe, (int, float)):
                score += float(sharpe) * 3.0
            if isinstance(profit, (int, float)):
                score += float(profit) * 0.2
            if isinstance(trades, (int, float)):
                score += min(float(trades) / 50.0, 2.0)
            score += min(int(card.get("reuse_count", 0) or 0), 5) * 0.2

            card_family = _text_blob(card.get("candidate_family"), card.get("candidate_type"))
            family_tier = _family_match_tier(family_blob, card_family)
            if family_tier == 3:
                score += 2.0
            elif family_tier == 2:
                score += 1.0
            elif family_tier == 1:
                score += 0.5

            card_timeframe = str(card.get("timeframe") or "").strip()
            if timeframe and card_timeframe == str(timeframe).strip():
                score += 0.8

            card_assets = set(_base_assets(list(card.get("universe") or [])))
            if requested_assets:
                score += len(requested_assets & card_assets) * 0.4

            results.append((family_tier, score, dict(card)))
        results.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [card for _, _, card in results[: max(0, int(top_n))]]

    def query_failure_cards_for_generation(
        self,
        *,
        top_n: int = 3,
        family: str = "",
        timeframe: str = "",
        universe: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        requested_assets = set(_base_assets(list(universe or [])))
        family_blob = _text_blob(family)
        results: List[tuple[int, float, Dict[str, Any]]] = []
        for card in self._failure_cards:
            score = 0.0
            card_family = _text_blob(card.get("candidate_family"), card.get("candidate_type"))
            family_tier = _family_match_tier(family_blob, card_family)
            if family_tier == 3:
                score += 2.0
            elif family_tier == 2:
                score += 1.0
            elif family_tier == 1:
                score += 0.5
            card_timeframe = str(card.get("timeframe") or "").strip()
            if timeframe and card_timeframe == str(timeframe).strip():
                score += 0.8
            card_assets = set(_base_assets(list(card.get("universe") or [])))
            if requested_assets:
                score += len(requested_assets & card_assets) * 0.4
            results.append((family_tier, score, dict(card)))
        results.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [card for _, _, card in results[: max(0, int(top_n))]]

    def retrieve_for_generation(
        self,
        *,
        family: str = "",
        timeframe: str = "",
        universe: Optional[List[str]] = None,
        top_n: int = 3,
        recent_n: int = 0,
    ) -> StrategyRetrievalResult:
        strategy_cards = self.query_strategy_cards_for_generation(
            top_n=top_n,
            family=family,
            timeframe=timeframe,
            universe=universe,
        )
        # Deterministic recency slice: helps cross-run reuse even when a newly-added
        # card doesn't rank in `top_n` due to score weights.
        try:
            recent_n_int = max(0, int(recent_n or 0))
        except Exception:
            recent_n_int = 0
        if recent_n_int > 0 and self._strategy_cards:
            seen: set[str] = set()
            for card in strategy_cards:
                cid = str(card.get("card_id") or "").strip()
                seen.add(cid or f"{card.get('run_id','')}::{card.get('name','')}")
            by_recency = sorted(
                self._strategy_cards,
                key=lambda c: str((c or {}).get("created_at") or ""),
                reverse=True,
            )
            for raw in by_recency:
                cid = str(raw.get("card_id") or "").strip()
                key = cid or f"{raw.get('run_id','')}::{raw.get('name','')}"
                if key in seen:
                    continue
                strategy_cards.append(dict(raw))
                seen.add(key)
                if len(strategy_cards) >= max(0, int(top_n or 0)) + recent_n_int:
                    break
        failure_cards = self.query_failure_cards_for_generation(
            top_n=min(max(1, top_n), 3),
            family=family,
            timeframe=timeframe,
            universe=universe,
        )
        return StrategyRetrievalResult(
            context=format_strategy_retrieval_context(
                strategy_cards=strategy_cards,
                failure_cards=failure_cards,
            ),
            query={
                "family": family,
                "timeframe": timeframe,
                "universe": list(universe or []),
                "top_n": int(top_n),
                "recent_n": int(recent_n_int),
            },
            strategy_cards=strategy_cards,
            failure_cards=failure_cards,
        )

    def register_strategy_references(
        self,
        *,
        card_ids: List[str],
        strategy_name: str,
        strategy_run_id: str,
        candidate_family: str = "",
    ) -> None:
        card_id_set = {str(card_id or "").strip() for card_id in card_ids if str(card_id or "").strip()}
        if not card_id_set:
            return
        changed = False
        for idx, card in enumerate(self._strategy_cards):
            card_id = str(card.get("card_id") or "").strip()
            if card_id not in card_id_set:
                continue
            refs = list(card.get("downstream_strategy_references") or [])
            ref = {
                "strategy_name": strategy_name,
                "strategy_run_id": strategy_run_id,
                "candidate_family": candidate_family,
            }
            if ref not in refs:
                refs.append(ref)
            card["downstream_strategy_references"] = refs[-50:]
            card["reuse_count"] = int(card.get("reuse_count", 0) or 0) + 1
            self._strategy_cards[idx] = card
            edge = {
                "parent": card_id,
                "child": strategy_name,
                "edge_type": "strategy_reference",
                "run_id": strategy_run_id,
            }
            if edge not in self._edges:
                self._edges.append(edge)
            changed = True
        if changed:
            self.save()

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
