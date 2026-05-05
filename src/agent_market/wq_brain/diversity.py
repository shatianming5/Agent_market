"""Family rotation + semantic similarity for alpha expression diversity.

Token jaccard (currently used by ``_check_local_jaccard_vs_active``) treats
``rank(ts_rank(close, 252))`` and ``rank(ts_rank(vwap, 252))`` as different
because the field token differs — even though the operator skeleton is
identical. WQ's self-correlation gate only cares about realized returns, so
those two often correlate ≥ 0.9. We need a *semantic* similarity that picks
up on operator-stack identity even when fields swap.

This module exposes three primitives:

  * ``op_field_multiset(expr)`` — break an expression into Counter of
    operators and Counter of fields. Counters (not sets) preserve
    multiplicity so ``rank(rank(...))`` ≠ ``rank(...)``.
  * ``semantic_jaccard(a, b)`` — weighted multiset Jaccard over those two
    Counters; default weights 0.7 ops + 0.3 fields, so operator skeleton
    drives the score.
  * ``family_distribution`` / ``next_target_family`` — small helpers around
    ``crossover.infer_family`` that tell callers which canonical family is
    under-represented in a pool, used by the agent prompt rotation policy.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional

from .crossover import infer_family

# Canonical 8 families that ``agent_brief.md`` calls out as acceptable.
# Order is the rotation order when all are equally rare (front first).
CANONICAL_FAMILIES: tuple[str, ...] = (
    "ts_corr_pv", "intraday_range", "vwap_dev", "volume_rank",
    "open_gap", "humped", "multi_signal", "sector_relative",
)

_OP_PATTERN = re.compile(r"([a-z_][a-z0-9_]*)\s*\(")
_WORD_PATTERN = re.compile(r"\b([a-z_][a-z0-9_]*)\b")

# Words that are grammar / boolean tokens but match the field regex
_GRAMMAR_TOKENS = frozenset({"if", "else", "and", "or", "true", "false", "not"})

# Fields known to WQ FASTEXPR. Anything matching the word regex but not in
# this set + not an operator is treated as a free variable and ignored,
# avoiding pollution from misspelled operators or non-data identifiers.
KNOWN_FIELDS: frozenset[str] = frozenset({
    "close", "open", "high", "low", "volume", "vwap", "returns", "cap",
    "adv5", "adv10", "adv20", "adv30", "adv60", "adv90", "adv120", "adv180",
    "sector", "industry", "subindustry", "country", "exchange",
    "market", "industry_group", "subsector",
    "earnings", "pe_ratio", "book_value", "shares_outstanding",
})


def op_field_multiset(expr: str) -> tuple[Counter, Counter]:
    """Decompose an expression into (operator multiset, field multiset).

    Multiplicity matters: ``rank(rank(close))`` has ``rank → 2``, not 1.
    Numeric literals and unknown identifiers are dropped.
    """
    if not expr:
        return Counter(), Counter()
    expr_l = expr.lower()
    ops = Counter(_OP_PATTERN.findall(expr_l))
    fields: Counter = Counter()
    for w in _WORD_PATTERN.findall(expr_l):
        if w in ops or w in _GRAMMAR_TOKENS:
            continue
        if w.replace(".", "").replace("_", "").isdigit():
            continue
        if w in KNOWN_FIELDS:
            fields[w] += 1
    return ops, fields


def _multiset_jaccard(a: Counter, b: Counter) -> float:
    """Jaccard on counter intersection / union (handles empty case)."""
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return inter / union if union > 0 else 0.0


def semantic_jaccard(expr_a: str, expr_b: str, *, op_weight: float = 0.75) -> float:
    """Multiset-Jaccard on (operators, fields), default 0.75×ops + 0.25×fields.

    Returns float in [0, 1]. ``op_weight`` controls how much of the score is
    driven by the operator skeleton — 0.75 means a pure field swap
    (close→vwap) while keeping the operator stack identical scores 0.75,
    which is above the 0.65 default block threshold for ``cmd_submit``.
    """
    if not 0.0 <= op_weight <= 1.0:
        raise ValueError(f"op_weight must be in [0, 1], got {op_weight}")
    ops_a, fields_a = op_field_multiset(expr_a)
    ops_b, fields_b = op_field_multiset(expr_b)
    op_sim = _multiset_jaccard(ops_a, ops_b)
    field_sim = _multiset_jaccard(fields_a, fields_b)
    return op_weight * op_sim + (1.0 - op_weight) * field_sim


def family_distribution(exprs: Iterable[str]) -> Counter:
    """Counter of family names across a list of expressions (skips empty)."""
    return Counter(infer_family(e or "") for e in exprs if e)


def next_target_family(
    exprs: Iterable[str],
    *,
    canonical: tuple[str, ...] = CANONICAL_FAMILIES,
) -> Optional[str]:
    """Pick the canonical family with the lowest count in the pool.

    Returns the first family in ``canonical`` order at the minimum count,
    or ``None`` if the pool is empty or already perfectly uniform.
    """
    exprs_list = [e for e in exprs if e]
    if not exprs_list:
        return canonical[0] if canonical else None
    counts = family_distribution(exprs_list)
    pairs = [(counts.get(f, 0), idx, f) for idx, f in enumerate(canonical)]
    pairs.sort()  # by count asc, then by canonical-order asc
    min_count = pairs[0][0]
    max_count = max(p[0] for p in pairs)
    if min_count == max_count:
        return None
    return pairs[0][2]


def max_semantic_similarity(
    candidate: str,
    pool: Iterable[str],
    *,
    op_weight: float = 0.75,
) -> tuple[float, Optional[str]]:
    """Return (max similarity, blocking expression) of candidate vs pool.

    Useful for the pre-submit gate: if max sim ≥ threshold, reject.
    Returns ``(0.0, None)`` for an empty pool.
    """
    best = 0.0
    blocker: Optional[str] = None
    for other in pool:
        if not other:
            continue
        s = semantic_jaccard(candidate, other, op_weight=op_weight)
        if s > best:
            best = s
            blocker = other
    return best, blocker
