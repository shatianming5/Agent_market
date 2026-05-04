"""Genetic mutation engine for WQ Brain alpha pool."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .dtypes import AlphaCandidate, AlphaSettings
from .operators import validate_expression

if TYPE_CHECKING:
    from .dtypes import AlphaPoolEntry

_WINDOW_VALUES = [5, 10, 20, 30, 60, 120, 252]

_TS_FAMILIES: dict[str, list[str]] = {
    "ts_mean":  ["ts_mean", "ts_rank", "ts_zscore"],
    "ts_rank":  ["ts_mean", "ts_rank", "ts_zscore"],
    "ts_zscore": ["ts_mean", "ts_rank", "ts_zscore"],
    "ts_sum":   ["ts_mean", "ts_sum"],
    "ts_delta": ["ts_delta", "ts_delay"],
    "ts_delay": ["ts_delta", "ts_delay"],
    "ts_max":   ["ts_max"],
    "ts_corr":  ["ts_corr"],
    "ts_decay_linear": ["ts_decay_linear"],
}

_PRICE_FIELDS = ["close", "vwap", "open", "high", "low"]

_INVALID = frozenset(["adv60", "adv120", "adv180", "cap"])


def _is_valid(expr: str) -> bool:
    return not any("Unavailable" in e for e in validate_expression(expr))


def mutate_window(expr: str) -> list[str]:
    """Replace numeric window args in ts_* calls with alternatives."""
    # [^()]* (no nested parens) avoids matching across nested calls like ts_mean(ts_rank(x,5),20)
    pattern = re.compile(r"(ts_\w+\s*\([^()]*?,\s*)(\d+)(\s*\))")
    results: list[str] = []
    for m in pattern.finditer(expr):
        orig = int(m.group(2))
        for w in _WINDOW_VALUES:
            if w != orig:
                new = expr[:m.start(2)] + str(w) + expr[m.end(2):]
                if _is_valid(new):
                    results.append(new)
    return list(dict.fromkeys(results))


def mutate_op(expr: str) -> list[str]:
    """Replace ts_* operators with alternatives from the same family."""
    results: list[str] = []
    for op, alts in _TS_FAMILIES.items():
        if re.search(r"\b" + op + r"\b", expr):
            for alt in alts:
                if alt != op:
                    new = re.sub(r"\b" + op + r"\b", alt, expr)
                    if _is_valid(new):
                        results.append(new)
    return list(dict.fromkeys(results))


def mutate_field(expr: str) -> list[str]:
    """Replace price fields with alternatives."""
    results: list[str] = []
    for fld in _PRICE_FIELDS:
        if re.search(r"\b" + fld + r"\b", expr):
            for alt in _PRICE_FIELDS:
                if alt != fld:
                    new = re.sub(r"\b" + fld + r"\b", alt, expr)
                    if _is_valid(new) and not any(t in new for t in _INVALID):
                        results.append(new)
    return list(dict.fromkeys(results))


def _strip_rank(expr: str) -> str:
    m = re.match(r"^(?:group_)?rank\s*\((.+)\)$", expr.strip())
    return m.group(1).strip() if m else expr.strip()


def mutate_crossover(expr_a: str, expr_b: str) -> list[str]:
    """Combine two expressions arithmetically."""
    a, b = _strip_rank(expr_a), _strip_rank(expr_b)
    results: list[str] = []
    for op in ["+", "-", "*"]:
        new = f"rank({a} {op} {b})"
        if _is_valid(new):
            results.append(new)
    return results


def generate_mutations(
    pool_entries: list,
    settings: AlphaSettings,
    *,
    top_n: int = 5,
    variants_per_parent: int = 4,
    crossover: bool = True,
) -> list[AlphaCandidate]:
    """Produce mutated candidates from the pool's top-N entries."""
    parents = pool_entries[:top_n]
    if not parents:
        return []

    seen: set[str] = {p.expr for p in parents}
    candidates: list[str] = []

    for p in parents:
        pool = (
            mutate_window(p.expr)
            + mutate_op(p.expr)
            + mutate_field(p.expr)
        )
        count = 0
        for v in pool:
            if v not in seen:
                seen.add(v)
                candidates.append(v)
                count += 1
                if count >= variants_per_parent:
                    break

    if crossover and len(parents) >= 2:
        for i in range(min(3, len(parents))):
            for j in range(i + 1, min(4, len(parents))):
                for cx in mutate_crossover(parents[i].expr, parents[j].expr):
                    if cx not in seen:
                        seen.add(cx)
                        candidates.append(cx)

    return [AlphaCandidate(expr=e, settings=settings) for e in candidates]
