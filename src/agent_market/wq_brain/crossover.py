"""Crossover engine — extract top fragments from iteration history for recombination.

Adapted from QuantGPT (Apache/MIT). Where QuantGPT uses an LLM to recombine,
we surface the top fragments as a structured markdown block injected into the
agent's prompt so the existing opencode/hermes session does the synthesis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Segment:
    expr: str
    score: float
    sharpe: Optional[float] = None
    fitness: Optional[float] = None
    turnover: Optional[float] = None
    family: str = ""  # rough family label inferred from expr structure


# ── Family inference (heuristic, used to diversify the surfaced segments) ──

_FAMILY_PATTERNS: dict[str, re.Pattern] = {
    "ts_corr_pv":         re.compile(r"ts_corr\s*\(\s*(close|vwap|open|high|low)\s*,\s*(volume|adv\d+)"),
    "ts_corr_other":      re.compile(r"ts_corr\("),
    "ts_rank_close":      re.compile(r"ts_rank\s*\(\s*close"),
    "ts_delta_close":     re.compile(r"ts_delta\s*\(\s*close"),
    "intraday_range":     re.compile(r"\(\s*high\s*-\s*low\s*\)"),
    "vwap_dev":           re.compile(r"close\s*/\s*vwap|vwap\s*-\s*close|close\s*-\s*vwap"),
    "decay_linear":       re.compile(r"ts_decay_linear\("),
    "humped":             re.compile(r"hump\("),
    "group_neutral":      re.compile(r"group_(zscore|neutralize|rank|mean)\("),
    "volume_rank":        re.compile(r"ts_rank\s*\(\s*volume"),
    "open_gap":           re.compile(r"open\s*-\s*ts_delay|ts_delay\s*\([^)]*close[^)]*\)"),
}


def infer_family(expr: str) -> str:
    """Tag an expression with the most-specific matched family label."""
    e = expr.lower()
    for name, pat in _FAMILY_PATTERNS.items():
        if pat.search(e):
            return name
    return "other"


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _quick_score(sh: Optional[float], fi: Optional[float], to: Optional[float]) -> float:
    """Synthetic 0-100 score from sh/fi/to. Same formula as mutation.FailureContext."""
    sh_v = max(0.0, min(2.0, sh or 0.0))
    fi_v = max(0.0, min(1.5, fi or 0.0))
    to_penalty = max(0.0, (to or 0.5) - 0.2) * 50
    return max(0.0, min(100.0, 30 * sh_v + 40 * fi_v - to_penalty + 10))


def extract_top_segments(
    records: list[dict[str, Any]],
    *,
    min_score: float = 30.0,
    top_n: int = 5,
    diversify_by_family: bool = True,
) -> list[Segment]:
    """Pick the top-N highest-scoring tried_log records, optionally one per family.

    Args:
        records: tried_log JSONL rows or pool entries.
        min_score: minimum quick_score (0-100) to qualify.
        top_n: max segments returned.
        diversify_by_family: if True, prefer one segment per family before picking
            duplicates from the same family (encourages crossover diversity).
    """
    segments: list[Segment] = []
    for r in records:
        expr = r.get("expr") or r.get("expression") or ""
        if not expr:
            continue
        if r.get("status") and r.get("status") != "COMPLETE":
            continue
        sh = _to_float(r.get("sharpe"))
        fi = _to_float(r.get("fitness"))
        to = _to_float(r.get("turnover"))
        score = _quick_score(sh, fi, to)
        if score < min_score:
            continue
        segments.append(Segment(
            expr=expr, score=score,
            sharpe=sh, fitness=fi, turnover=to,
            family=infer_family(expr),
        ))

    segments.sort(key=lambda s: -s.score)
    if not diversify_by_family:
        return segments[:top_n]

    seen_families: set[str] = set()
    diverse: list[Segment] = []
    for s in segments:
        if s.family in seen_families:
            continue
        diverse.append(s)
        seen_families.add(s.family)
        if len(diverse) >= top_n:
            break
    if len(diverse) < top_n:
        for s in segments:
            if s in diverse:
                continue
            diverse.append(s)
            if len(diverse) >= top_n:
                break
    return diverse


def format_crossover_block(segments: list[Segment]) -> str:
    """Render the top fragments as a markdown block for the agent prompt."""
    if not segments:
        return ""
    lines = [
        "## Cross-Over Candidates (MANDATORY recombination source)",
        "",
        "Top fragments from prior iterations, grouped by family.",
        "",
        "### 🚫 RULE — at least 2 of your simulated alphas this session MUST",
        "use a family DIFFERENT from any single source family in this table.",
        "If 4 of the 5 below are `ts_rank_close`, your first 2 alphas MUST be",
        "`ts_corr_pv`, `intraday_range`, `vwap_dev`, `volume_rank`, or a",
        "completely novel family. NO MORE single-family parameter tuning.",
        "",
        "### How to use this table",
        "Recombine — extract the *winning piece* of each (window choice,",
        "normalization shape, signal direction, liquidity term) and synthesize",
        "a NEW expression that fuses ≥2 of them. Do NOT just re-submit any of",
        "these alphas; do NOT just tweak a single window.",
        "",
        "| # | family | sh | fi | to | quick | expr |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(segments, 1):
        sh = f"{s.sharpe:.2f}" if s.sharpe is not None else "-"
        fi = f"{s.fitness:.2f}" if s.fitness is not None else "-"
        to = f"{s.turnover:.2f}" if s.turnover is not None else "-"
        expr = s.expr.replace("|", "/")[:90]
        lines.append(f"| {i} | {s.family} | {sh} | {fi} | {to} | {s.score:.0f} | `{expr}` |")
    lines.append("")
    lines.append("**Recombination patterns** to consider:")
    lines.append("- Window-graft: take family A's window length and apply to family B's operator stack")
    lines.append("- Normalization swap: replace family A's outer `rank()` with family B's `group_zscore(_, sector)`")
    lines.append("- Sign mix: subtract two families: `rank(family_A) - 0.5 * rank(family_B)`")
    lines.append("- Liquidity overlay: multiply a directional family by `volume / adv60` from a humped family")
    return "\n".join(lines)
