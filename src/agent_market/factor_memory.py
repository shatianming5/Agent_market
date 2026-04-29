from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from filelock import FileLock, Timeout as FileLockTimeout

logger = logging.getLogger(__name__)


FACTOR_MEMORY_REQUIRED_FIELDS: tuple[str, ...] = (
    "source_run_id",
    "timeframe",
    "universe",
    "target",
    "metrics.train_ic",
    "metrics.validation_ic",
    "metrics.blind_ic",
    "metrics.turnover",
    "capacity_slippage_proxy",
    "regime_tags",
    "snoop_level",
)

FACTOR_MEMORY_STATUS_CLEAN = "clean"
FACTOR_MEMORY_STATUS_LEGACY = "legacy"
FACTOR_MEMORY_STATUS_OOS_SNOOPED = "oos_snooped"
FACTOR_MEMORY_STATUS_TRADEABLE = "tradeable_candidate"


def _nested_get(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in str(path).split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if not _is_missing(value):
            return value
    return None


def _as_float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _factor_memory_missing_fields(card: dict[str, Any]) -> list[str]:
    return [field for field in FACTOR_MEMORY_REQUIRED_FIELDS if _is_missing(_nested_get(card, field))]


def _factor_tradeability_gate(card: dict[str, Any]) -> dict[str, Any]:
    """Lightweight non-destructive gate for factor-to-strategy readiness."""
    metrics = card.get("metrics") if isinstance(card.get("metrics"), dict) else {}
    reasons: list[str] = []
    if not bool(card.get("gate_pass")):
        reasons.append("factor_gate_not_passed")

    snoop_level = str(card.get("snoop_level") or "").strip().lower()
    if snoop_level and snoop_level not in {"clean", "unknown"}:
        reasons.append(f"snoop_level={snoop_level}")

    validation_ic = _as_float_or_none(
        _first_present(metrics.get("validation_ic"), metrics.get("oos_ic"), metrics.get("ic"))
    )
    blind_ic = _as_float_or_none(metrics.get("blind_ic"))
    rank_ic = _as_float_or_none(_first_present(metrics.get("rank_ic"), metrics.get("ic")))
    if validation_ic is None and rank_ic is None:
        reasons.append("missing_validation_or_rank_ic")
    elif validation_ic is not None and abs(validation_ic) < 0.01 and (rank_ic is None or abs(rank_ic) < 0.01):
        reasons.append("weak_validation_or_rank_ic")

    turnover = _as_float_or_none(metrics.get("turnover"))
    if turnover is None:
        reasons.append("missing_turnover")
    elif turnover > 5.0:
        reasons.append("turnover_too_high")

    corr = _as_float_or_none(metrics.get("corr_to_library_max"))
    if corr is not None and abs(corr) >= 0.95:
        reasons.append("duplicate_or_highly_correlated")

    transfer_score = _as_float_or_none(
        _first_present(metrics.get("strategy_transfer_score"), metrics.get("rank_portfolio_transfer_score"))
    )
    if transfer_score is not None and transfer_score <= 0:
        reasons.append("negative_strategy_transfer")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "checks": {
            "validation_ic": validation_ic,
            "blind_ic": blind_ic,
            "rank_ic": rank_ic,
            "turnover": turnover,
            "corr_to_library_max": corr,
            "strategy_transfer_score": transfer_score,
        },
    }


def classify_factor_memory_card(card: dict[str, Any]) -> str:
    """Classify a factor card for memory hygiene and downstream strategy use."""
    snoop_level = str(card.get("snoop_level") or "").strip().lower()
    if snoop_level in {"oos_snooped", "snooped", "test_snooped", "leaky"}:
        return FACTOR_MEMORY_STATUS_OOS_SNOOPED

    missing = _factor_memory_missing_fields(card)
    critical_missing = {field for field in missing if field in {"source_run_id", "timeframe", "universe", "target", "snoop_level"}}
    if critical_missing or snoop_level in {"legacy", "unknown", ""}:
        return FACTOR_MEMORY_STATUS_LEGACY

    if _factor_tradeability_gate(card).get("passed"):
        return FACTOR_MEMORY_STATUS_TRADEABLE
    return FACTOR_MEMORY_STATUS_CLEAN


def _annotate_factor_memory_card(card: dict[str, Any]) -> dict[str, Any]:
    out = dict(card)
    missing = _factor_memory_missing_fields(out)
    out["memory_status"] = classify_factor_memory_card(out)
    out["audit_missing_fields"] = missing
    out["quality_gate"] = _factor_tradeability_gate(out)
    return out


def audit_factor_memory_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cards = [dict(card) for card in list(payload.get("factor_cards") or []) if isinstance(card, dict)]
    failures = [dict(card) for card in list(payload.get("failure_cards") or []) if isinstance(card, dict)]
    annotated = [_annotate_factor_memory_card(card) for card in cards]

    coverage: dict[str, dict[str, Any]] = {}
    for field in FACTOR_MEMORY_REQUIRED_FIELDS:
        present = sum(0 if _is_missing(_nested_get(card, field)) else 1 for card in cards)
        total = len(cards)
        coverage[field] = {
            "present": present,
            "missing": total - present,
            "coverage_pct": round((present / total * 100.0) if total else 100.0, 2),
        }

    status_counts: dict[str, int] = {}
    snoop_counts: dict[str, int] = {}
    for card in annotated:
        status = str(card.get("memory_status") or FACTOR_MEMORY_STATUS_LEGACY)
        status_counts[status] = status_counts.get(status, 0) + 1
        snoop = str(card.get("snoop_level") or "missing").strip().lower() or "missing"
        snoop_counts[snoop] = snoop_counts.get(snoop, 0) + 1

    sig_clusters: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        signature = str(card.get("signature") or "").strip()
        if not signature:
            continue
        sig_clusters.setdefault(signature, []).append(card)
    duplicate_clusters = [
        {
            "signature": signature,
            "count": len(cluster),
            "card_ids": [str(item.get("card_id") or "") for item in cluster[:20]],
            "names": [str(item.get("name") or "") for item in cluster[:20]],
        }
        for signature, cluster in sig_clusters.items()
        if len(cluster) > 1
    ]
    duplicate_clusters.sort(key=lambda item: int(item["count"]), reverse=True)

    missing_by_card = [
        {
            "card_id": card.get("card_id"),
            "name": card.get("name"),
            "memory_status": card.get("memory_status"),
            "missing_fields": card.get("audit_missing_fields") or [],
        }
        for card in annotated
        if card.get("audit_missing_fields")
    ]
    missing_by_card.sort(key=lambda item: len(item.get("missing_fields") or []), reverse=True)

    total = len(cards)
    legacy_or_snooped = status_counts.get(FACTOR_MEMORY_STATUS_LEGACY, 0) + status_counts.get(FACTOR_MEMORY_STATUS_OOS_SNOOPED, 0)
    return {
        "version": "factor-memory-audit-v1",
        "total_factor_cards": total,
        "total_failure_cards": len(failures),
        "status_counts": status_counts,
        "snoop_level_counts": snoop_counts,
        "legacy_or_oos_snooped_ratio": round((legacy_or_snooped / total) if total else 0.0, 6),
        "coverage": coverage,
        "missing_required_count": len(missing_by_card),
        "missing_by_card": missing_by_card[:100],
        "duplicate_cluster_count": len(duplicate_clusters),
        "duplicate_factor_count": sum(int(item["count"]) - 1 for item in duplicate_clusters),
        "duplicate_clusters": duplicate_clusters[:50],
        "tradeable_candidates": [
            {
                "card_id": card.get("card_id"),
                "name": card.get("name"),
                "metrics": card.get("metrics") or {},
                "quality_gate": card.get("quality_gate") or {},
            }
            for card in annotated
            if card.get("memory_status") == FACTOR_MEMORY_STATUS_TRADEABLE
        ][:100],
        "required_fields": list(FACTOR_MEMORY_REQUIRED_FIELDS),
    }


def audit_factor_memory_path(path: Path, *, write_tags: bool = False) -> dict[str, Any]:
    store = FactorMemoryStore(Path(path))
    payload = {
        "factor_cards": store.factor_cards,
        "failure_cards": store.failure_cards,
        "edges": store.edges,
    }
    audit = audit_factor_memory_payload(payload)
    audit["path"] = str(Path(path).resolve())
    if write_tags:
        store._payload["factor_cards"] = [_annotate_factor_memory_card(card) for card in store.factor_cards]
        store.save()
        store._sync_sidecar_exports_if_present()
        audit["write_tags"] = True
    else:
        audit["write_tags"] = False
    return audit

def _read_json(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _factor_signature(
    *,
    expression: str,
    timeframe: Any,
    universe: list[Any],
    target_col: str,
) -> str:
    """Stable signature used to deduplicate factors across runs."""
    uni = sorted({str(x).strip().upper() for x in (universe or []) if str(x).strip()})
    payload = {
        "expression": str(expression or "").strip(),
        "timeframe": str(timeframe or "").strip(),
        "universe": list(uni),
        "target_col": str(target_col or "").strip(),
    }
    return _sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _parse_iso8601(value: Any) -> float:
    raw = str(value or "").strip()
    if not raw:
        return 0.0
    try:
        # `fromisoformat` supports "+00:00" but not "Z".
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _failed_gate_reasons(item: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    gates = item.get("gates") or {}

    nan_ratio = item.get("nan_ratio")
    max_nan_ratio = gates.get("max_nan_ratio")
    if isinstance(nan_ratio, (int, float)) and isinstance(max_nan_ratio, (int, float)):
        if float(nan_ratio) > float(max_nan_ratio):
            reasons.append("nan_ratio")

    turnover = item.get("turnover")
    max_turnover = gates.get("max_turnover")
    if isinstance(turnover, (int, float)) and isinstance(max_turnover, (int, float)):
        if float(turnover) > float(max_turnover):
            reasons.append("turnover")

    corr_max = item.get("corr_to_library_max")
    max_corr = gates.get("max_corr_to_library")
    if isinstance(corr_max, (int, float)) and isinstance(max_corr, (int, float)):
        if float(corr_max) > float(max_corr):
            reasons.append("corr_to_library")

    if not reasons and not bool(item.get("gate_pass")):
        reasons.append("gate_fail")
    return reasons


@dataclass
class FactorMemoryArtifacts:
    factor_memory_json: str
    factor_cards_json: str
    factor_failure_cards_json: str
    factor_lineage_json: str


@dataclass
class FactorRetrievalResult:
    context: str
    query: dict[str, Any]
    factor_cards: list[dict[str, Any]]
    failure_cards: list[dict[str, Any]]


_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "breakout": ("breakout", "donchian", "range break", "squeeze", "impulse"),
    "trend-following": ("trend", "momentum", "macd", "adx", "follow"),
    "trend-pullback": ("pullback", "retest", "dip buy", "trend pullback"),
    "mean-reversion": ("reversion", "mean revert", "boll", "vwap", "stoch", "cci", "rsi"),
    "basket-rotation": ("rotation", "cross sectional", "ranking", "basket"),
    "pair-relative": ("pair", "relative", "spread", "ratio"),
    "ml": ("lightgbm", "xgboost", "prediction", "regression", "model"),
    "dl": ("mlp", "neural", "deep", "torch"),
    "rl": ("ppo", "policy", "reward shaping", "reinforcement"),
}

_REGIME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "trend": ("trend", "momentum", "follow"),
    "range": ("range", "reversion", "mean revert"),
    "volatility_expand": ("breakout", "squeeze", "impulse", "expansion"),
    "volatility_contract": ("compression", "contract", "squeeze"),
}


def _text_blob(*parts: Any) -> str:
    return " ".join(str(item or "") for item in parts).strip().lower()


def _base_assets(universe: Iterable[str]) -> list[str]:
    assets: list[str] = []
    for raw in universe:
        text = str(raw or "").strip().upper()
        if not text:
            continue
        pair = text.split(":")[0]
        base = pair.split("/")[0]
        if base and base not in assets:
            assets.append(base)
    return assets


def _infer_tags(*, name: str, hypothesis: str, expression: str) -> tuple[list[str], list[str]]:
    blob = _text_blob(name, hypothesis, expression)
    category_tags = [
        tag for tag, keywords in _CATEGORY_KEYWORDS.items()
        if any(keyword in blob for keyword in keywords)
    ]
    regime_tags = [
        tag for tag, keywords in _REGIME_KEYWORDS.items()
        if any(keyword in blob for keyword in keywords)
    ]
    return category_tags, regime_tags


def _repair_recipe_from_failure_reasons(reasons: list[str]) -> dict[str, list[str]]:
    avoid_rules: list[str] = []
    mutation_hints: list[str] = []
    exclusion_constraints: list[str] = []
    if "nan_ratio" in reasons:
        avoid_rules.append("Avoid factors that require heavy backfill or sparse informative joins.")
        mutation_hints.append("Prefer close/volume-derived features or shorter warmup windows.")
        exclusion_constraints.append("reject_nan_ratio_above_gate")
    if "turnover" in reasons:
        avoid_rules.append("Avoid hyper-reactive signals that flip on every bar.")
        mutation_hints.append("Add smoothing, regime filters, or wider thresholds.")
        exclusion_constraints.append("reject_turnover_above_gate")
    if "corr_to_library" in reasons:
        avoid_rules.append("Do not resubmit library-clone factors with only cosmetic expression edits.")
        mutation_hints.append("Combine the signal with an orthogonal volatility or volume term.")
        exclusion_constraints.append("reject_corr_to_library_above_gate")
    if "gate_fail" in reasons and not avoid_rules:
        avoid_rules.append("Do not reuse this exact factor shape without changing its core driver.")
        mutation_hints.append("Mutate horizon, normalization, or cross-term composition.")
        exclusion_constraints.append("reject_prior_gate_fail_clone")
    return {
        "avoid_rules": avoid_rules,
        "mutation_hints": mutation_hints,
        "exclusion_constraints": exclusion_constraints,
    }


def _card_preview(card: dict[str, Any]) -> dict[str, Any]:
    metrics = card.get("metrics") or {}
    return {
        "card_id": card.get("card_id"),
        "name": card.get("name"),
        "expression": card.get("expression"),
        "timeframe": card.get("timeframe"),
        "universe": list(card.get("universe") or []),
        "gate_pass": bool(card.get("gate_pass")),
        "pareto": bool(card.get("pareto")),
        "weighted_score": metrics.get("weighted_score"),
        "category_tags": list(card.get("category_tags") or []),
        "regime_tags": list(card.get("regime_tags") or []),
    }


def format_factor_retrieval_context(
    *,
    factor_cards: list[dict[str, Any]],
    failure_cards: list[dict[str, Any]],
) -> str:
    if not factor_cards and not failure_cards:
        return ""

    lines: list[str] = [
        "## Factor Memory Retrieval",
        "Use these factor findings as building blocks or guardrails. Recombine them; do not copy them blindly.",
    ]
    if factor_cards:
        lines.append("### Retrieved factor cards")
        for card in factor_cards[:4]:
            metrics = card.get("metrics") or {}
            tags = list(card.get("category_tags") or []) + list(card.get("regime_tags") or [])
            tag_text = ", ".join(tags[:4]) if tags else "none"
            lines.append(
                "- {name} [{card_id}] tf={tf} gate_pass={gate_pass} pareto={pareto} "
                "score={score} tags={tags}".format(
                    name=card.get("name") or "?",
                    card_id=card.get("card_id") or "?",
                    tf=card.get("timeframe") or "?",
                    gate_pass=bool(card.get("gate_pass")),
                    pareto=bool(card.get("pareto")),
                    score=metrics.get("weighted_score"),
                    tags=tag_text,
                )
            )
            expr = str(card.get("expression") or "").strip()
            if expr:
                lines.append(f"  expression: {expr[:220]}")
            hypothesis = str(card.get("hypothesis") or "").strip()
            if hypothesis:
                lines.append(f"  hypothesis: {hypothesis[:180]}")
    if failure_cards:
        lines.append("### Failure cards to avoid")
        for failure in failure_cards[:3]:
            recipe = failure.get("repair_recipe") or {}
            hints = ", ".join((recipe.get("mutation_hints") or [])[:2]) or "mutate the factor more aggressively"
            lines.append(
                "- {name} [{failure_id}] reasons={reasons}; next mutation hints: {hints}".format(
                    name=failure.get("name") or "?",
                    failure_id=failure.get("failure_id") or "?",
                    reasons=failure.get("subcategory") or failure.get("category") or "gate_fail",
                    hints=hints,
                )
            )
    return "\n".join(lines)


class FactorMemoryStore:
    """Persistent factor memory plane for factor-mining runs."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._payload: dict[str, Any] = {
            "schema_version": "1.0",
            "factor_cards": [],
            "failure_cards": [],
            "edges": [],
        }
        if self._path.exists():
            loaded = _read_json(self._path)
            if loaded:
                self._payload = {
                    "schema_version": str(loaded.get("schema_version") or "1.0"),
                    "factor_cards": list(loaded.get("factor_cards") or []),
                    "failure_cards": list(loaded.get("failure_cards") or []),
                    "edges": list(loaded.get("edges") or []),
                }

    @property
    def factor_cards(self) -> list[dict[str, Any]]:
        return list(self._payload.get("factor_cards") or [])

    @property
    def failure_cards(self) -> list[dict[str, Any]]:
        return list(self._payload.get("failure_cards") or [])

    @property
    def edges(self) -> list[dict[str, Any]]:
        return list(self._payload.get("edges") or [])

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def merge_payload(self, payload: dict[str, Any]) -> None:
        def _factor_key(card: dict[str, Any]) -> str:
            sig = str(card.get("signature") or "").strip()
            if not sig:
                try:
                    sig = _factor_signature(
                        expression=str(card.get("expression") or ""),
                        timeframe=card.get("timeframe"),
                        universe=list(card.get("universe") or []),
                        target_col=str(card.get("target_col") or ""),
                    )
                except Exception:
                    sig = ""
            return sig or str(card.get("card_id") or "").strip()

        factor_index = {
            _factor_key(card): idx
            for idx, card in enumerate(self._payload.get("factor_cards") or [])
            if _factor_key(card)
        }
        failure_index = {
            str(card.get("failure_id") or ""): idx
            for idx, card in enumerate(self._payload.get("failure_cards") or [])
            if str(card.get("failure_id") or "")
        }
        edge_keys = {
            (
                str(edge.get("parent") or ""),
                str(edge.get("child") or ""),
                str(edge.get("edge_type") or ""),
                str(edge.get("run_id") or ""),
            )
            for edge in (self._payload.get("edges") or [])
        }

        for card in list(payload.get("factor_cards") or []):
            card = _annotate_factor_memory_card(dict(card))
            merge_key = _factor_key(card)
            if not merge_key:
                continue
            if merge_key in factor_index:
                existing = dict((self._payload.get("factor_cards") or [])[factor_index[merge_key]])
                merged = dict(existing)
                merged.update(card)
                # Preserve original card_id but remember aliases when de-duping by signature.
                existing_card_id = str(existing.get("card_id") or "").strip()
                incoming_card_id = str(card.get("card_id") or "").strip()
                if existing_card_id and incoming_card_id and existing_card_id != incoming_card_id:
                    aliases = list(existing.get("source_card_ids") or [])
                    for item in (incoming_card_id,):
                        if item not in aliases:
                            aliases.append(item)
                    merged["source_card_ids"] = aliases[-20:]
                merged["seen_count"] = int(existing.get("seen_count", 1) or 1) + 1
                merged["last_seen_run_id"] = str(card.get("run_id") or merged.get("run_id") or "")

                refs = list(existing.get("downstream_strategy_references") or [])
                for ref in list(card.get("downstream_strategy_references") or []):
                    if ref not in refs:
                        refs.append(ref)
                merged["downstream_strategy_references"] = refs[-50:]
                merged["reuse_count"] = max(
                    int(existing.get("reuse_count", 0) or 0),
                    int(card.get("reuse_count", 0) or 0),
                )
                # Keep the best metrics snapshot (by weighted_score).
                try:
                    existing_score = (existing.get("metrics") or {}).get("weighted_score")
                    incoming_score = (card.get("metrics") or {}).get("weighted_score")
                    es = float(existing_score) if existing_score is not None else float("-inf")
                    ins = float(incoming_score) if incoming_score is not None else float("-inf")
                    if ins > es:
                        merged["metrics"] = dict(card.get("metrics") or {})
                        merged["gate_pass"] = bool(card.get("gate_pass"))
                        merged["pareto"] = bool(card.get("pareto"))
                except Exception:
                    pass

                self._payload["factor_cards"][factor_index[merge_key]] = _annotate_factor_memory_card(merged)
            else:
                self._payload["factor_cards"].append(card)
                factor_index[merge_key] = len(self._payload["factor_cards"]) - 1

        for card in list(payload.get("failure_cards") or []):
            failure_id = str(card.get("failure_id") or "").strip()
            if not failure_id:
                continue
            if failure_id in failure_index:
                self._payload["failure_cards"][failure_index[failure_id]] = dict(card)
            else:
                self._payload["failure_cards"].append(dict(card))
                failure_index[failure_id] = len(self._payload["failure_cards"]) - 1

        for edge in list(payload.get("edges") or []):
            edge_key = (
                str(edge.get("parent") or ""),
                str(edge.get("child") or ""),
                str(edge.get("edge_type") or ""),
                str(edge.get("run_id") or ""),
            )
            if edge_key in edge_keys:
                continue
            self._payload["edges"].append(dict(edge))
            edge_keys.add(edge_key)

    def prune(
        self,
        *,
        max_factor_cards: int = 2000,
        max_failure_cards: int = 800,
        max_edges: int = 5000,
    ) -> None:
        """Best-effort cap memory growth for long-running global stores."""
        try:
            max_factor_cards_i = max(1, int(max_factor_cards))
        except Exception:
            max_factor_cards_i = 2000
        try:
            max_failure_cards_i = max(1, int(max_failure_cards))
        except Exception:
            max_failure_cards_i = 800
        try:
            max_edges_i = max(1, int(max_edges))
        except Exception:
            max_edges_i = 5000

        cards = list(self._payload.get("factor_cards") or [])
        if len(cards) > max_factor_cards_i:
            def _rank(card: dict[str, Any]) -> tuple:
                metrics = card.get("metrics") or {}
                ws = metrics.get("weighted_score")
                try:
                    ws_f = float(ws) if ws is not None else float("-inf")
                except Exception:
                    ws_f = float("-inf")
                return (
                    1 if bool(card.get("gate_pass")) else 0,
                    1 if bool(card.get("pareto")) else 0,
                    ws_f,
                    int(card.get("reuse_count", 0) or 0),
                    _parse_iso8601(card.get("generated_at")),
                )

            cards.sort(key=_rank, reverse=True)
            kept = cards[:max_factor_cards_i]
            kept_ids = {str(c.get("card_id") or "") for c in kept if str(c.get("card_id") or "")}
            kept_names = {str(c.get("name") or "") for c in kept if str(c.get("name") or "")}
            kept_spec = {str(c.get("spec_name") or "") for c in kept if str(c.get("spec_name") or "")}
            keep_tokens = kept_ids | kept_names | kept_spec
            self._payload["factor_cards"] = kept

            edges = list(self._payload.get("edges") or [])
            filtered: list[dict[str, Any]] = []
            for edge in edges:
                parent = str(edge.get("parent") or "")
                child = str(edge.get("child") or "")
                if parent in keep_tokens or child in keep_tokens:
                    filtered.append(edge)
            self._payload["edges"] = filtered[:max_edges_i]

        failures = list(self._payload.get("failure_cards") or [])
        if len(failures) > max_failure_cards_i:
            failures.sort(key=lambda c: _parse_iso8601(c.get("generated_at")), reverse=True)
            self._payload["failure_cards"] = failures[:max_failure_cards_i]

    def merge_from_store(self, other: "FactorMemoryStore") -> None:
        self.merge_payload(other._payload)

    def merge_from_path(self, path: Path) -> None:
        payload = _read_json(Path(path).resolve())
        if payload:
            self.merge_payload(payload)

    def _sync_sidecar_exports_if_present(self) -> None:
        out_dir = self._path.parent
        factor_cards_path = out_dir / "factor_cards.json"
        factor_failures_path = out_dir / "factor_failure_cards.json"
        factor_lineage_path = out_dir / "factor_lineage.json"

        if factor_cards_path.exists():
            factor_cards_path.write_text(
                json.dumps({"items": self.factor_cards}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if factor_failures_path.exists():
            factor_failures_path.write_text(
                json.dumps({"items": self.failure_cards}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if factor_lineage_path.exists():
            factor_lineage_path.write_text(
                json.dumps({"edges": self.edges}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def query_factor_cards(
        self,
        *,
        top_n: int = 3,
        family: str = "",
        timeframe: str = "",
        universe: Optional[Iterable[str]] = None,
        require_gate_pass: bool = True,
        pareto_only: bool = False,
    ) -> list[dict[str, Any]]:
        family_blob = _text_blob(family)
        requested_assets = set(_base_assets(universe or []))
        results: list[tuple[float, dict[str, Any]]] = []
        for card in self.factor_cards:
            if require_gate_pass and not bool(card.get("gate_pass")):
                continue
            if pareto_only and not bool(card.get("pareto")):
                continue
            if timeframe and str(card.get("timeframe") or "").strip() != str(timeframe).strip():
                continue

            score = 0.0
            metrics = card.get("metrics") or {}
            weighted = metrics.get("weighted_score")
            if isinstance(weighted, (int, float)):
                score += float(weighted)
            if bool(card.get("gate_pass")):
                score += 2.0
            if bool(card.get("pareto")):
                score += 1.0

            card_assets = set(_base_assets(card.get("universe") or []))
            overlap = len(requested_assets & card_assets)
            if requested_assets:
                score += overlap * 0.4

            family_tags = set(card.get("category_tags") or []) | set(card.get("regime_tags") or [])
            if family_blob:
                if any(tag in family_blob or family_blob in tag for tag in family_tags):
                    score += 1.5
                card_blob = _text_blob(card.get("name"), card.get("hypothesis"), card.get("expression"))
                if family_blob and family_blob.split("/")[-1] in card_blob:
                    score += 0.8

            results.append((score, card))

        results.sort(key=lambda item: item[0], reverse=True)
        return [_card_preview(card) | {"metrics": dict(card.get("metrics") or {}), "hypothesis": card.get("hypothesis")} for _, card in results[: max(0, int(top_n))]]

    def query_failure_cards(
        self,
        *,
        top_n: int = 2,
        timeframe: str = "",
        universe: Optional[Iterable[str]] = None,
        failure_reason: str = "",
    ) -> list[dict[str, Any]]:
        requested_assets = set(_base_assets(universe or []))
        reason_blob = _text_blob(failure_reason)
        results: list[tuple[float, dict[str, Any]]] = []
        for card in self.failure_cards:
            detail = card.get("detail") or {}
            if timeframe and str(detail.get("timeframe") or card.get("timeframe") or "").strip() not in {"", str(timeframe).strip()}:
                continue
            score = 0.0
            detail_assets = set(_base_assets((detail.get("universe") or card.get("universe") or [])))
            if requested_assets:
                score += len(requested_assets & detail_assets) * 0.4
            if reason_blob:
                sub = _text_blob(card.get("subcategory"))
                if reason_blob in sub:
                    score += 1.5
            results.append((score, card))
        results.sort(key=lambda item: item[0], reverse=True)
        return [dict(card) for _, card in results[: max(0, int(top_n))]]

    def retrieve_for_strategy(
        self,
        *,
        family: str = "",
        timeframe: str = "",
        universe: Optional[Iterable[str]] = None,
        top_n: int = 3,
    ) -> FactorRetrievalResult:
        factor_cards = self.query_factor_cards(
            top_n=top_n,
            family=family,
            timeframe=timeframe,
            universe=universe,
            require_gate_pass=True,
        )
        if not factor_cards and timeframe:
            factor_cards = self.query_factor_cards(
                top_n=top_n,
                family=family,
                timeframe="",
                universe=universe,
                require_gate_pass=True,
            )
        failure_cards = self.query_failure_cards(
            top_n=min(3, max(1, top_n)),
            timeframe=timeframe,
            universe=universe,
            failure_reason=family,
        )
        if not failure_cards and timeframe:
            failure_cards = self.query_failure_cards(
                top_n=min(3, max(1, top_n)),
                timeframe="",
                universe=universe,
                failure_reason=family,
            )
        query = {
            "family": family,
            "timeframe": timeframe,
            "universe": list(universe or []),
            "top_n": int(top_n),
        }
        return FactorRetrievalResult(
            context=format_factor_retrieval_context(
                factor_cards=factor_cards,
                failure_cards=failure_cards,
            ),
            query=query,
            factor_cards=factor_cards,
            failure_cards=failure_cards,
        )

    def register_strategy_references(
        self,
        *,
        card_ids: Iterable[str],
        strategy_name: str,
        strategy_run_id: str,
        candidate_family: str = "",
    ) -> None:
        card_id_set = {str(card_id or "").strip() for card_id in card_ids if str(card_id or "").strip()}
        if not card_id_set:
            return
        changed = False
        for idx, card in enumerate(self._payload.get("factor_cards") or []):
            card_id = str(card.get("card_id") or "")
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
            card["downstream_strategy_references"] = refs[-20:]
            card["reuse_count"] = int(card.get("reuse_count", 0) or 0) + 1
            self._payload["factor_cards"][idx] = card
            changed = True
            edge = {
                "parent": card_id,
                "child": strategy_name,
                "edge_type": "strategy_reference",
                "run_id": strategy_run_id,
            }
            if edge not in (self._payload.get("edges") or []):
                self._payload["edges"].append(edge)
        if changed:
            self.save()
            self._sync_sidecar_exports_if_present()

    def ingest_evaluation(
        self,
        *,
        run_id: str,
        factor_spec: Optional[dict[str, Any]],
        factor_eval_meta: Optional[dict[str, Any]],
        score_payload: dict[str, Any],
        source_artifacts: dict[str, str],
    ) -> None:
        spec = factor_spec or {}
        eval_meta = factor_eval_meta or {}
        spec_meta = spec.get("meta") or {}
        spec_name = str(spec.get("name") or "compiled_factor")
        hypothesis = str(spec.get("hypothesis") or "")
        expression = str(eval_meta.get("expression") or spec.get("expression") or "")
        target_col = str(score_payload.get("target_col") or "")
        generated_at = str(score_payload.get("generated_at") or "")
        timeframe = spec_meta.get("timeframe")
        universe = list(spec_meta.get("universe") or [])
        data_sources = list(spec_meta.get("data_sources") or [])
        category_tags, regime_tags = _infer_tags(
            name=spec_name,
            hypothesis=hypothesis,
            expression=expression,
        )

        card_index = {
            str(card.get("card_id") or ""): idx
            for idx, card in enumerate(self._payload.get("factor_cards") or [])
        }
        failure_index = {
            str(card.get("failure_id") or ""): idx
            for idx, card in enumerate(self._payload.get("failure_cards") or [])
        }
        edge_keys = {
            (str(edge.get("parent") or ""), str(edge.get("child") or ""), str(edge.get("edge_type") or ""))
            for edge in (self._payload.get("edges") or [])
        }

        for item in list(score_payload.get("items") or []):
            name = str(item.get("name") or spec_name)
            card_id = f"{run_id}:{spec_name}:{name}"
            signature = _factor_signature(
                expression=expression,
                timeframe=timeframe,
                universe=universe,
                target_col=target_col,
            )
            card = {
                "card_id": card_id,
                "signature": signature,
                "run_id": run_id,
                "source_run_id": run_id,
                "name": name,
                "spec_name": spec_name,
                "hypothesis": hypothesis,
                "expression": expression,
                "target_col": target_col,
                "target": target_col,
                "timeframe": timeframe,
                "universe": universe,
                "data_sources": data_sources,
                "gate_pass": bool(item.get("gate_pass")),
                "pareto": bool(item.get("pareto")),
                "category_tags": category_tags,
                "regime_tags": regime_tags,
                "regime_coverage": item.get("regime_consistency"),
                "capacity_proxy": item.get("capacity_proxy"),
                "capacity_slippage_proxy": _first_present(
                    item.get("capacity_slippage_proxy"),
                    item.get("capacity_proxy"),
                    item.get("slippage_reduction_bps"),
                ),
                "snoop_level": str(_first_present(item.get("snoop_level"), score_payload.get("snoop_level"), spec_meta.get("snoop_level"), "unknown")),
                "reuse_count": int((self._payload.get("factor_cards") or [{}])[card_index.get(card_id, 0)].get("reuse_count", 0) or 0) if card_id in card_index else 0,
                "downstream_strategy_references": list((self._payload.get("factor_cards") or [{}])[card_index.get(card_id, 0)].get("downstream_strategy_references", []) or []) if card_id in card_index else [],
                "metrics": {
                    "weighted_score": item.get("weighted_score"),
                    "gate_pass": item.get("gate_pass"),
                    "pareto": item.get("pareto"),
                    "ic": item.get("ic"),
                    "train_ic": _first_present(item.get("train_ic"), item.get("in_sample_ic")),
                    "validation_ic": _first_present(item.get("validation_ic"), item.get("oos_ic")),
                    "blind_ic": item.get("blind_ic"),
                    "rank_ic": item.get("rank_ic"),
                    "sharpe_net": item.get("sharpe_net"),
                    "sortino_net": item.get("sortino_net"),
                    "mdd": item.get("mdd"),
                    "turnover": item.get("turnover"),
                    "nan_ratio": item.get("nan_ratio"),
                    "corr_to_library_max": item.get("corr_to_library_max"),
                    "capacity_proxy": item.get("capacity_proxy"),
                    "regime_consistency": item.get("regime_consistency"),
                    "train_test_gap": item.get("train_test_gap"),
                    "slippage_reduction_bps": item.get("slippage_reduction_bps"),
                    "fill_rate": item.get("fill_rate"),
                    "adverse_selection_proxy": item.get("adverse_selection_proxy"),
                    "rank_portfolio_transfer_score": _first_present(
                        item.get("rank_portfolio_transfer_score"),
                        item.get("rank_portfolio_transfer"),
                    ),
                    "strategy_transfer_score": item.get("strategy_transfer_score"),
                },
                "generated_at": generated_at,
                "source_artifacts": source_artifacts,
                "seen_count": int((self._payload.get("factor_cards") or [{}])[card_index.get(card_id, 0)].get("seen_count", 1) or 1) if card_id in card_index else 1,
                "last_seen_run_id": run_id,
            }
            card = _annotate_factor_memory_card(card)
            if card_id in card_index:
                self._payload["factor_cards"][card_index[card_id]] = card
            else:
                self._payload["factor_cards"].append(card)

            edge = (spec_name, name, "factor_eval")
            if edge not in edge_keys:
                self._payload["edges"].append(
                    {"parent": spec_name, "child": name, "edge_type": "factor_eval", "run_id": run_id}
                )
                edge_keys.add(edge)

            if bool(item.get("gate_pass")):
                continue

            reasons = _failed_gate_reasons(item)
            failure_id = f"{run_id}:{spec_name}:{name}:{','.join(reasons) or 'gate_fail'}"
            failure_card = {
                "failure_id": failure_id,
                "run_id": run_id,
                "source_run_id": run_id,
                "name": name,
                "spec_name": spec_name,
                "signature": signature,
                "category": "gate_fail",
                "subcategory": ",".join(reasons) or "gate_fail",
                "timeframe": timeframe,
                "universe": universe,
                "category_tags": category_tags,
                "regime_tags": regime_tags,
                "detail": {
                    "metrics": item,
                    "expression": expression,
                    "target_col": target_col,
                    "timeframe": timeframe,
                    "universe": universe,
                },
                "repair_recipe": _repair_recipe_from_failure_reasons(reasons),
                "source_artifacts": source_artifacts,
                "generated_at": generated_at,
            }
            if failure_id in failure_index:
                self._payload["failure_cards"][failure_index[failure_id]] = failure_card
            else:
                self._payload["failure_cards"].append(failure_card)

    def write_exports(self, out_dir: Path) -> FactorMemoryArtifacts:
        out_dir.mkdir(parents=True, exist_ok=True)
        factor_cards_path = (out_dir / "factor_cards.json").resolve()
        factor_failures_path = (out_dir / "factor_failure_cards.json").resolve()
        factor_lineage_path = (out_dir / "factor_lineage.json").resolve()

        factor_cards_path.write_text(
            json.dumps({"items": self.factor_cards}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        factor_failures_path.write_text(
            json.dumps({"items": self.failure_cards}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        factor_lineage_path.write_text(
            json.dumps({"edges": self.edges}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.save()
        return FactorMemoryArtifacts(
            factor_memory_json=str(self._path.resolve()),
            factor_cards_json=str(factor_cards_path),
            factor_failure_cards_json=str(factor_failures_path),
            factor_lineage_json=str(factor_lineage_path),
        )


def build_factor_memory_artifacts(
    *,
    run_id: str,
    memory_dir: Path,
    factor_spec_path: Optional[Path],
    factor_eval_meta_path: Optional[Path],
    factor_scores_path: Path,
) -> FactorMemoryArtifacts:
    memory_dir = Path(memory_dir).resolve()
    store = FactorMemoryStore(memory_dir / "factor_memory.json")
    factor_spec = _read_json(factor_spec_path.resolve()) if factor_spec_path else {}
    factor_eval_meta = _read_json(factor_eval_meta_path.resolve()) if factor_eval_meta_path else {}
    score_payload = _read_json(factor_scores_path.resolve())
    store.ingest_evaluation(
        run_id=run_id,
        factor_spec=factor_spec,
        factor_eval_meta=factor_eval_meta,
        score_payload=score_payload,
        source_artifacts={
            "factor_spec_json": str(factor_spec_path.resolve()) if factor_spec_path else "",
            "factor_eval_meta": str(factor_eval_meta_path.resolve()) if factor_eval_meta_path else "",
            "factor_scores_json": str(factor_scores_path.resolve()),
        },
    )
    return store.write_exports(memory_dir)


def build_factor_memory_artifacts_from_expression_output(
    *,
    run_id: str,
    memory_dir: Path,
    expressions_path: Path,
    scored_candidates_path: Optional[Path] = None,
) -> FactorMemoryArtifacts:
    memory_dir = Path(memory_dir).resolve()
    expressions_payload = _read_json(expressions_path.resolve())
    scored_payload = _read_json(scored_candidates_path.resolve()) if scored_candidates_path else {}

    selected_items = list(expressions_payload.get("expressions") or [])
    scored_candidates = list(scored_payload.get("candidates") or [])
    scored_by_expr = {
        str(item.get("expression") or "").strip(): item
        for item in scored_candidates
        if str(item.get("expression") or "").strip()
    }

    timeframe = str(expressions_payload.get("timeframe") or "")
    universe = list(expressions_payload.get("pairs") or [])
    feature_file = str(expressions_payload.get("feature_file") or "")
    exchange = str(expressions_payload.get("exchange") or "")
    label_period = expressions_payload.get("label_period")
    generated_at = str(expressions_payload.get("generated_at") or "")
    source_artifacts = {
        "expression_output_json": str(expressions_path.resolve()),
        "scored_candidates_json": str(scored_candidates_path.resolve()) if scored_candidates_path else "",
    }

    factor_cards: list[dict[str, Any]] = []
    failure_cards: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    selected_exprs: set[str] = set()

    for idx, item in enumerate(selected_items, start=1):
        expr = str(item.get("expression") or "").strip()
        if not expr:
            continue
        selected_exprs.add(expr)
        name = str(item.get("name") or f"f{idx:03d}")
        score_item = scored_by_expr.get(expr, {})
        hypothesis = str(item.get("description") or score_item.get("description") or "")
        category_tags, regime_tags = _infer_tags(
            name=name,
            hypothesis=hypothesis,
            expression=expr,
        )
        weighted_score = item.get("score")
        if weighted_score is None:
            weighted_score = score_item.get("score")
        metric_abs_ic = item.get("metric_abs_ic")
        if metric_abs_ic is None:
            metric_abs_ic = score_item.get("metric_abs_ic")
        per_pair = item.get("per_pair")
        if per_pair is None:
            per_pair = score_item.get("per_pair") or {}
        transfer_audit = item.get("transfer_audit")
        if not isinstance(transfer_audit, dict):
            transfer_audit = score_item.get("transfer_audit") if isinstance(score_item.get("transfer_audit"), dict) else {}
        agent_tags = [
            str(tag)
            for tag in list(item.get("agent_tags") or score_item.get("agent_tags") or [])
            if str(tag).strip()
        ]
        memory_scope = str(item.get("memory_scope") or ("pending_review" if agent_tags else "legacy_expression_mining"))

        target_col = f"future_return_{label_period}" if label_period is not None else ""
        signature = _factor_signature(
            expression=expr,
            timeframe=timeframe,
            universe=universe,
            target_col=target_col,
        )
        card_id = f"{run_id}:expression:{name}"
        legacy_card = {
                "card_id": card_id,
                "signature": signature,
                "run_id": run_id,
                "source_run_id": run_id,
                "name": name,
                "spec_name": "legacy_expression_mining",
                "hypothesis": hypothesis,
                "expression": expr,
                "target_col": target_col,
                "target": target_col,
                "timeframe": timeframe,
                "universe": universe,
                "data_sources": [item for item in ["freqai_expression_mining", feature_file] if item],
                "gate_pass": True,
                "pareto": True,
                "category_tags": list(dict.fromkeys([str(item.get("category") or "").strip()] + category_tags)),
                "regime_tags": regime_tags,
                "regime_coverage": None,
                "capacity_proxy": _first_present(
                    transfer_audit.get("capacity_proxy"),
                    transfer_audit.get("capacity_slippage_proxy"),
                ),
                "capacity_slippage_proxy": transfer_audit.get("capacity_slippage_proxy"),
                "snoop_level": str(item.get("snoop_level") or score_item.get("snoop_level") or expressions_payload.get("snoop_level") or "legacy"),
                "reuse_count": 0,
                "downstream_strategy_references": [],
                "memory_scope": memory_scope,
                "agent_tags": agent_tags,
                "critic_review": item.get("critic_review") or score_item.get("critic_review") or {},
                "transfer_audit": transfer_audit,
                "promotion_eligible": False,
                "metrics": {
                    "weighted_score": weighted_score,
                    "gate_pass": True,
                    "pareto": True,
                    "ic": metric_abs_ic,
                    "train_ic": score_item.get("train_ic"),
                    "validation_ic": score_item.get("validation_ic"),
                    "blind_ic": score_item.get("blind_ic"),
                    "rank_ic": transfer_audit.get("rank_ic_proxy"),
                    "sharpe_net": None,
                    "sortino_net": None,
                    "mdd": None,
                    "turnover": _first_present(
                        item.get("turnover"),
                        score_item.get("turnover"),
                        transfer_audit.get("turnover_proxy"),
                    ),
                    "nan_ratio": None,
                    "corr_to_library_max": None,
                    "capacity_proxy": _first_present(
                        transfer_audit.get("capacity_proxy"),
                        transfer_audit.get("capacity_slippage_proxy"),
                    ),
                    "regime_consistency": None,
                    "train_test_gap": None,
                    "slippage_reduction_bps": transfer_audit.get("slippage_reduction_bps"),
                    "fill_rate": None,
                    "adverse_selection_proxy": None,
                    "per_pair": per_pair,
                    "complexity": item.get("complexity", score_item.get("complexity")),
                    "metric_abs_ic": metric_abs_ic,
                    "rank_portfolio_transfer_score": _first_present(
                        transfer_audit.get("rank_portfolio_transfer_score"),
                        score_item.get("rank_portfolio_transfer_score"),
                    ),
                    "strategy_transfer_score": _first_present(
                        transfer_audit.get("strategy_transfer_score"),
                        score_item.get("strategy_transfer_score"),
                    ),
                },
                "generated_at": generated_at,
                "source_artifacts": source_artifacts,
                "exchange": exchange,
                "origin": item.get("origin") or score_item.get("origin") or "",
                "seen_count": 1,
                "last_seen_run_id": run_id,
            }
        factor_cards.append(_annotate_factor_memory_card(legacy_card))
        edges.append(
            {
                "parent": "legacy_expression_mining",
                "child": name,
                "edge_type": "factor_eval",
                "run_id": run_id,
            }
        )

    rejected_count = 0
    for idx, item in enumerate(scored_candidates, start=1):
        expr = str(item.get("expression") or "").strip()
        if not expr or expr in selected_exprs:
            continue
        if rejected_count >= 12:
            break
        rejected_count += 1
        name = f"rejected_{idx:03d}"
        hypothesis = str(item.get("description") or "")
        category_tags, regime_tags = _infer_tags(
            name=name,
            hypothesis=hypothesis,
            expression=expr,
        )
        failure_cards.append(
            {
                "failure_id": f"{run_id}:legacy_expression:{name}:low_score",
                "run_id": run_id,
                "source_run_id": run_id,
                "name": name,
                "spec_name": "legacy_expression_mining",
                "signature": _factor_signature(
                    expression=expr,
                    timeframe=timeframe,
                    universe=universe,
                    target_col=f"future_return_{label_period}" if label_period is not None else "",
                ),
                "category": "selection_reject",
                "subcategory": "low_score",
                "timeframe": timeframe,
                "universe": universe,
                "category_tags": list(dict.fromkeys([str(item.get("category") or "").strip()] + category_tags)),
                "regime_tags": regime_tags,
                "detail": {
                    "expression": expr,
                    "score": item.get("score"),
                    "metric_abs_ic": item.get("metric_abs_ic"),
                    "complexity": item.get("complexity"),
                    "per_pair": item.get("per_pair") or {},
                },
                "repair_recipe": {
                    "avoid_rules": [
                        "Do not keep low-scoring mined expressions unchanged in the next batch."
                    ],
                    "mutation_hints": [
                        "Reduce complexity or combine the signal with a more orthogonal feature."
                    ],
                    "exclusion_constraints": [
                        "prefer_higher_expression_score"
                    ],
                },
                "source_artifacts": source_artifacts,
                "generated_at": generated_at,
            }
        )

    for idx, failure in enumerate(list(expressions_payload.get("multiagent_failures") or []), start=1):
        if not isinstance(failure, dict):
            continue
        expr = str(failure.get("expression") or "").strip()
        subcategory = str(failure.get("subcategory") or failure.get("category") or "multiagent_failure")
        name = str(failure.get("name") or f"multiagent_rejected_{idx:03d}")
        category_tags, regime_tags = _infer_tags(name=name, hypothesis=subcategory, expression=expr)
        failure_cards.append(
            {
                "failure_id": f"{run_id}:legacy_expression:{name}:{subcategory}",
                "run_id": run_id,
                "source_run_id": run_id,
                "name": name,
                "spec_name": "legacy_expression_mining",
                "signature": _factor_signature(
                    expression=expr,
                    timeframe=timeframe,
                    universe=universe,
                    target_col=f"future_return_{label_period}" if label_period is not None else "",
                ),
                "category": "multiagent_review",
                "subcategory": subcategory,
                "timeframe": timeframe,
                "universe": universe,
                "category_tags": category_tags,
                "regime_tags": regime_tags,
                "detail": {
                    "expression": expr,
                    "role": failure.get("role"),
                    "promotion_eligible": False,
                },
                "repair_recipe": {
                    "avoid_rules": ["Do not reuse rejected multi-agent factor output unchanged."],
                    "mutation_hints": ["Fix the critic rejection reason before rescoring this factor."],
                    "exclusion_constraints": ["multiagent_rejected_factor"],
                },
                "source_artifacts": source_artifacts,
                "generated_at": generated_at,
            }
        )

    store = FactorMemoryStore(memory_dir / "factor_memory.json")
    store._payload = {
        "schema_version": "1.0",
        "factor_cards": factor_cards,
        "failure_cards": failure_cards,
        "edges": edges,
    }
    return store.write_exports(memory_dir)


def merge_factor_memory_artifacts(
    *,
    source_memory_path: Path,
    target_memory_dir: Path,
) -> FactorMemoryArtifacts:
    target_memory_dir = Path(target_memory_dir).resolve()
    lock_path = target_memory_dir / ".factor_memory.lock"
    timeout = float(os.environ.get("AGENT_MARKET_FACTOR_MEMORY_LOCK_TIMEOUT_SEC", "15") or "15")
    max_factor_cards = int(os.environ.get("AGENT_MARKET_FACTOR_MEMORY_MAX_FACTOR_CARDS", "2000") or "2000")
    max_failure_cards = int(os.environ.get("AGENT_MARKET_FACTOR_MEMORY_MAX_FAILURE_CARDS", "800") or "800")
    max_edges = int(os.environ.get("AGENT_MARKET_FACTOR_MEMORY_MAX_EDGES", "5000") or "5000")

    lock = FileLock(str(lock_path))
    try:
        with lock.acquire(timeout=timeout):
            target_store = FactorMemoryStore(target_memory_dir / "factor_memory.json")
            target_store.merge_from_path(source_memory_path)
            target_store.prune(
                max_factor_cards=max_factor_cards,
                max_failure_cards=max_failure_cards,
                max_edges=max_edges,
            )
            return target_store.write_exports(target_memory_dir)
    except FileLockTimeout as exc:
        logger.warning("Global factor memory lock timeout: %s", lock_path)
        raise RuntimeError(f"factor_memory_lock_timeout:{lock_path}") from exc
