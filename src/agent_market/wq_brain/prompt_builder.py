"""Prompt-block builders for the agentic loop.

Extracted from ``agent_runner.py`` so the prompt orchestration can be
unit-tested + reused (e.g. by a future "preview-prompt" CLI) without
pulling in the LLM/CLI machinery.

Public-ish API (kept underscore-prefixed for back-compat with tests that
import them directly):

  * ``_family_diversity_hint`` — "next candidate must come from family X"
    based on the ACTIVE pool composition.
  * ``_operator_skeleton`` — operator multiset signature; two expressions
    collapse to the same skeleton when they share an operator stack.
  * ``_tried_skeleton_concentration_hint`` — fires when ≥ 50% of recent
    attempts share an operator skeleton (catches "same family, same
    skeleton, different fields" — invisible to family rotation).
  * ``_tried_family_concentration_hint`` — fires when ≥ 60% of recent
    attempts share a single family (forces rotation).
  * ``_build_prior_knowledge_block`` — top-level renderer. Combines
    pool snapshot + tried_log tail + cross-over fragments + mutation
    hints into a single Markdown block injected into the system prompt.

This module is import-cheap (no subprocess, no LLM CLI dependencies) so
tests can render prompts in a few ms.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from .crossover import extract_top_segments, format_crossover_block, infer_family
from .mutation import render_top_failures_block
from .paths import alpha_pool_path, tried_exprs_path
from .pool import AlphaPool
from .tried_log import find_recent_revisits, format_for_prompt, read_tried


# Full canonical family roster — agent_brief lists these 9 as acceptable;
# crossover.infer_family adds finer subdivisions but the prompt-level
# rotation policy uses this top-level list.
_CANONICAL_FAMILIES: tuple[str, ...] = (
    "ts_corr_pv", "intraday_range", "vwap_dev", "volume_rank",
    "open_gap", "humped", "multi_signal", "sector_relative",
    "fundamental_ratio",
)


def _family_diversity_hint(active_families: list[str]) -> str:
    """Render the 'next candidate must come from family X' hint based on
    what's already in ACTIVE pool. Returns empty string when pool is fresh."""
    if not active_families:
        return ""
    counts = Counter(active_families)
    seen = set(counts)
    missing = [f for f in _CANONICAL_FAMILIES if f not in seen]
    dominant = ", ".join(f"`{name}` (×{n})" for name, n in counts.most_common())
    if missing:
        miss_str = ", ".join(f"`{f}`" for f in missing)
        return (
            f"_⚠️ Pool currently dominated by {dominant}. Your NEXT candidate MUST "
            f"come from a family NOT yet present: {miss_str}._"
        )
    return f"_⚠️ Pool covers {dominant}. Pick the family with the LOWEST count next._"


# Family-specific anti-recommendations — what to try when stuck
_FAMILY_ANTI_EXAMPLES: dict[str, str] = {
    "sector_relative":  "`rank(sales/assets)`, `hump(rank(vwap/close))`, `rank((high - low)/close)`",
    "ts_corr_pv":       "`rank(sales/assets)`, `rank((vwap - close)/close)`, `group_zscore(_, subindustry)`",
    "multi_signal":     "`rank(sales/assets)`, `ts_corr(close, volume, 20)`, `rank((high - low)/close)`",
    "ts_rank_close":    "`rank(sales/assets)`, `rank((vwap - close)/close)`, `rank((high - low)/close)`",
    "ts_delta_close":   "`rank(sales/assets)`, `ts_corr(_, _, _)`, `(high - low) / close`",
    "decay_linear":     "swap to a different inner shape — `rank(sales/assets)`, `hump(rank(...))`, `vwap - close`",
    "humped":           "`rank(sales/assets)`, `ts_corr(close, volume, 20)`, `(high - low)/close`",
    "intraday_range":   "`rank(sales/assets)`, `rank(vwap/close)`, `ts_corr(close, volume, 20)`",
    "open_gap":         "`rank(sales/assets)`, `rank(vwap/close)`, `(high - low)/close`",
    "vwap_dev":         "`rank(sales/assets)`, `rank(ts_rank(volume, 20))`, `ts_corr(close, volume, 20)`",
    "volume_rank":      "`rank(sales/assets)`, `rank(vwap/close)`, `(high - low)/close`",
    "group_neutral":    "`rank(sales/assets)`, `hump(rank(...))`, `ts_corr(close, volume, 20)`",
    "ts_corr_other":    "`rank(sales/assets)`, `group_zscore(_, sector)`, `(high - low)/close`",
    "fundamental_ratio": "TRY DIFFERENT FUNDAMENTAL SKELETONS (not just `rank(F/G) * ts_decay_linear(...)`): "
                         "(1) sector-neutral: `rank(group_zscore(sales/assets, sector))`; "
                         "(2) time-trend: `rank(ts_zscore(sales/assets, 252))`; "
                         "(3) regime-conditional: `if_else(ts_corr(close, volume, 20) > 0, rank(sales/assets), -rank(sales/assets))`; "
                         "(4) cross-fundamental ratio: `rank(sales/assets) - rank(debt/equity)`. "
                         "OR step out entirely: `rank((vwap - close)/close)`, `ts_corr(close, volume, 20)`, `hump(rank(...))`.",
}


# ── Concrete examples for every whitelisted operator + field ──────────
# Used by `_unexplored_space_hint` to force the agent to cover the whole
# action space. Each example is hand-tuned to be a syntactically correct
# FASTEXPR that exercises the named atom and is structurally distinct
# from the current top-fitness ACTIVE alphas (so even successful trials
# break out of the dominant family).
_OP_EXAMPLES: dict[str, str] = {
    # TS ops
    "ts_mean":          "rank(close - ts_mean(close, 60))",
    "ts_rank":          "rank(ts_rank(volume, 252))",
    "ts_zscore":        "ts_zscore(returns, 60)",
    "ts_sum":           "rank(ts_sum(returns, 20))",
    "ts_max":           "rank(close / ts_max(close, 60))",
    "ts_delta":         "rank(-ts_delta(close, 5) / close)",
    "ts_delay":         "rank(close / ts_delay(close, 20) - 1)",
    "ts_decay_linear":  "ts_decay_linear(rank(returns), 20)",
    "ts_corr":          "ts_corr(close, volume, 20)",
    # CS ops
    "rank":             "rank(close)",
    "group_rank":       "group_rank(rank(volume / adv60), industry)",
    "group_mean":       "rank(close - group_mean(close, sector))",
    "group_sum":        "rank(group_sum(volume, sector))",
    "group_std":        "rank(returns / group_std(returns, sector))",
    "group_zscore":     "group_zscore(close - vwap, sector)",
    "group_neutralize": "group_neutralize(rank(returns), industry)",
    "group_count":      "rank(volume / group_count(close, subindustry))",
    "scale":            "scale(rank(sales/assets) - rank(returns))",
    "signed_power":     "signed_power(returns, 0.5)",
    # MATH ops (the under-explored bench)
    "abs":              "rank(abs(returns))",
    "log":              "rank(log(volume))",
    "sqrt":             "rank(sqrt(abs(returns)))",
    "sign":             "sign(rank(close) - rank(open))",
    "power":            "power(rank(returns), 2)",
    "exp":              "rank(exp(-abs(returns)))",
    "min":              "min(rank(volume), rank(returns))",
    "max":              "max(rank(close - vwap), rank(close - open))",
    # `sum` / `mean` arity is (1,1) and (1,N) — they reduce a single panel,
    # NOT a 2-arg combiner. Use `+` for explicit 2-signal combination.
    "sum":              "rank(sum(rank(close)))",
    "mean":             "rank(mean(rank(close)))",
    "if_else":          "if_else(volume > adv60, rank(returns), -rank(returns))",
    "clamp":            "clamp(rank(returns), -2, 2)",
    # `correlation` / `covariance` are 3-arg with explicit window — they're
    # cross-sectional stats over a rolling time window, NOT 2-arg pairs.
    "correlation":      "correlation(rank(volume), rank(returns), 20)",
    "covariance":       "covariance(returns, ts_delta(volume, 5), 20)",
    # TRF ops — `hump` is 1-arg ONLY on free tier (2-arg form rejects with
    # "Invalid number of inputs: 2")
    "hump":             "hump(rank(close - vwap))",
}

_FIELD_EXAMPLES: dict[str, str] = {
    # P/V
    "open":     "rank(close - open)",
    "high":     "rank((high - low) / close)",
    "low":      "rank((close - low) / (high - low))",
    "close":    "rank(close / vwap)",
    "volume":   "rank(volume / adv20)",
    "vwap":     "rank((vwap - close) / close)",
    "returns":  "rank(ts_zscore(returns, 60))",
    "adv20":    "rank(volume / adv20)",
    "adv60":    "rank(volume / adv60)",
    "adv120":   "rank(volume / adv120)",
    "adv180":   "rank(volume / adv180)",
    # Group
    "sector":       "group_zscore(close - vwap, sector)",
    "industry":     "group_neutralize(rank(volume), industry)",
    "subindustry":  "group_zscore(rank(returns), subindustry)",
    # Fundamentals (5 are completely unused — the gap)
    "sales":            "rank(sales / assets)",
    "assets":           "rank(sales / assets)",
    "equity":           "rank(group_zscore(revenue/equity, sector))",
    "debt":             "rank(debt / cap)",
    "revenue":          "rank(revenue / equity)",
    "earnings":         "rank(earnings / cap)",
    "ebit":             "rank(ebit / assets)",
    "ebitda":           "rank(ebitda / cap)",
    "operating_income": "rank(operating_income / equity)",
    "net_income":       "rank(net_income / equity)",
    "cash":             "rank(cash / assets)",
    "fcf":              "rank(fcf / cap)",
    "shares":           "rank(volume / shares)",
    "cap":              "rank(earnings / cap)",
}


_OP_FAMILY: dict[str, str] = {
    **{op: "TS"  for op in (
        "ts_mean", "ts_rank", "ts_zscore", "ts_sum", "ts_max",
        "ts_delta", "ts_delay", "ts_decay_linear", "ts_corr")},
    **{op: "CS"  for op in (
        "rank", "group_rank", "group_mean", "group_sum", "group_std",
        "group_zscore", "group_neutralize", "group_count", "scale",
        "signed_power")},
    **{op: "MATH" for op in (
        "abs", "log", "sqrt", "sign", "power", "exp", "min", "max",
        "sum", "mean", "if_else", "clamp", "correlation", "covariance")},
    "hump": "TRF",
}

_FIELD_KIND: dict[str, str] = {
    **{f: "P/V" for f in (
        "open", "high", "low", "close", "volume", "vwap", "returns",
        "adv20", "adv60", "adv120", "adv180")},
    **{f: "GRP" for f in ("sector", "industry", "subindustry")},
    **{f: "FUND" for f in (
        "sales", "assets", "equity", "debt", "revenue", "earnings",
        "ebit", "ebitda", "operating_income", "net_income",
        "cash", "fcf", "shares", "cap")},
}


def _count_atoms(tried_records: list[dict]) -> tuple[Counter, Counter]:
    """Return (operator_usage, field_usage) Counters from tried_log rows."""
    op_pat = re.compile(r"([a-z_][a-z0-9_]*)\s*\(")
    word_pat = re.compile(r"\b([a-z_][a-z0-9_]*)\b")
    op_use, fi_use = Counter(), Counter()
    known_fields = set(_FIELD_EXAMPLES)
    for r in tried_records:
        ex = (r.get("expr") or "").lower()
        if not ex:
            continue
        for op in op_pat.findall(ex):
            op_use[op] += 1
        for w in word_pat.findall(ex):
            if w in known_fields:
                fi_use[w] += 1
    return op_use, fi_use


def _unexplored_space_hint(
    tried_records: list[dict],
    *,
    min_uses_per_op: int = 3,
    min_uses_per_field: int = 3,
    max_examples: int = 12,
) -> str:
    """Generate a hint listing whitelisted atoms with < ``min_uses`` mentions.

    The agent must drive coverage to 100% — every operator + field used at
    least N times. This block names the laggards and gives a concrete
    1-line example for each so the LLM can paste-and-mutate.
    """
    op_use, fi_use = _count_atoms(tried_records)
    unused_ops = [
        op for op in _OP_EXAMPLES
        if op_use.get(op, 0) < min_uses_per_op
    ]
    unused_fields = [
        f for f in _FIELD_EXAMPLES
        if fi_use.get(f, 0) < min_uses_per_field
    ]
    if not unused_ops and not unused_fields:
        return ""

    total_ops = len(_OP_EXAMPLES)
    total_fields = len(_FIELD_EXAMPLES)
    covered_ops = total_ops - len(unused_ops)
    covered_fields = total_fields - len(unused_fields)
    op_pct = covered_ops / total_ops * 100
    fi_pct = covered_fields / total_fields * 100

    lines = [
        "### 🎯 ACTION SPACE COVERAGE — drive these to 100%",
        "",
        f"_Currently exploring **{covered_ops}/{total_ops}** operators "
        f"({op_pct:.0f}%) and **{covered_fields}/{total_fields}** fields "
        f"({fi_pct:.0f}%). Goal: every atom used ≥ {min_uses_per_op} times._",
        "",
        "**Hard rule for the next 3 simulate calls**: pick at least ONE under-",
        "explored operator AND ONE under-explored field. Mutating an existing",
        "fundamental_ratio expr that re-uses the same atoms COUNTS AS A FAILURE.",
        "",
    ]

    if unused_ops:
        # Group by family for readability
        by_fam: dict[str, list[str]] = {}
        for op in unused_ops:
            by_fam.setdefault(_OP_FAMILY.get(op, "other"), []).append(op)
        lines.append("#### Operators with < " + str(min_uses_per_op) + " uses (try these)")
        lines.append("")
        for fam in ("MATH", "CS", "TS", "TRF", "other"):
            ops = by_fam.get(fam) or []
            if not ops:
                continue
            lines.append(f"- **{fam}** ({len(ops)}): " + ", ".join(f"`{op}`" for op in ops))
        lines.append("")
        # 1-line examples for the worst-covered handful
        worst = sorted(unused_ops, key=lambda op: op_use.get(op, 0))[:max_examples]
        if worst:
            lines.append("Concrete starter expressions (paste then mutate):")
            for op in worst:
                lines.append(f"- `{op}` → `{_OP_EXAMPLES[op]}`")
            lines.append("")

    if unused_fields:
        by_kind: dict[str, list[str]] = {}
        for f in unused_fields:
            by_kind.setdefault(_FIELD_KIND.get(f, "other"), []).append(f)
        lines.append("#### Fields with < " + str(min_uses_per_field) + " uses (try these)")
        lines.append("")
        for kind in ("FUND", "GRP", "P/V", "other"):
            flds = by_kind.get(kind) or []
            if not flds:
                continue
            lines.append(f"- **{kind}** ({len(flds)}): " + ", ".join(f"`{f}`" for f in flds))
        lines.append("")
        worst_f = sorted(unused_fields, key=lambda f: fi_use.get(f, 0))[:max_examples]
        if worst_f:
            lines.append("Concrete starter expressions (paste then mutate):")
            for f in worst_f:
                lines.append(f"- `{f}` → `{_FIELD_EXAMPLES[f]}`")
            lines.append("")

    lines.append(
        "_If your next 3 candidates don't introduce ≥ 1 fresh operator AND "
        "≥ 1 fresh field, the diversity hint will keep firing on every iteration._"
    )
    return "\n".join(lines)


_SKELETON_OP_PATTERN = re.compile(r"([a-z_][a-z0-9_]*)\s*\(")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_WORD_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\b")


def _extract_numbers(expr: str) -> tuple[float, ...]:
    """Sorted tuple of numeric literals — window sizes, scales, signed_power
    exponents. Used by :func:`_classify_edit_altitude` to distinguish L3
    (calibration) from L2 (skeleton) edits."""
    if not expr:
        return ()
    return tuple(sorted(float(n) for n in _NUMBER_RE.findall(expr.lower())))


def _extract_field_set(expr: str) -> frozenset[str]:
    """Set of canonical fields (close/vwap/sales/...) referenced in ``expr``.

    Drives L4-refinement detection: two exprs sharing skeleton + numbers but
    differing in field-set are field-swap edits."""
    if not expr:
        return frozenset()
    return frozenset(w for w in _WORD_RE.findall(expr.lower()) if w in _FIELD_EXAMPLES)


def _classify_edit_altitude(prev_expr: str, curr_expr: str) -> str:
    """Classify the magnitude of a ``prev → curr`` edit.

    Returns one of:

      * ``"L1"`` — family swap (different :func:`infer_family` bucket).
      * ``"L2"`` — skeleton swap (same family, different operator multiset).
      * ``"L3"`` — calibration (same skeleton, numeric constants differ).
      * ``"L4"`` — refinement (same skeleton + constants, fields differ).
      * ``"—"``  — duplicate / unclassifiable.

    Used by :func:`_recent_altitude_distribution_hint` to detect "stuck at
    L4" patterns that family / skeleton concentration checks miss because
    they only look at distributions, not transitions.
    """
    if not prev_expr or not curr_expr or prev_expr == curr_expr:
        return "—"
    if infer_family(prev_expr) != infer_family(curr_expr):
        return "L1"
    if _operator_skeleton(prev_expr) != _operator_skeleton(curr_expr):
        return "L2"
    if _extract_numbers(prev_expr) != _extract_numbers(curr_expr):
        return "L3"
    if _extract_field_set(prev_expr) != _extract_field_set(curr_expr):
        return "L4"
    return "—"


def _altitude_taxonomy_block() -> str:
    """Static reference block: what the 4 edit altitudes mean + bubble-up rule.

    Rendered once per prompt so the agent has L1-L4 definitions before
    reading the dynamic concentration / distribution hints below.

    Codex review #8: examples MUST match the classifier
    (:func:`_classify_edit_altitude`), which checks family → skeleton →
    numbers → fields in that order. A normalization swap that *also*
    changes the family bucket (e.g. ``rank(...)`` → ``group_zscore(...,
    sector)``) lands as L1, not L3 — so we describe L3 strictly as
    intra-skeleton numeric / scale tweaks.
    """
    return (
        "### 🪜 EDIT ALTITUDE — taxonomy (declare this for every simulate)\n"
        "\n"
        "Every edit picks ONE of four altitudes (L1 highest, L4 lowest). "
        "**Higher altitudes preempt lower ones**: an edit that changes the "
        "family bucket is always L1, even if your intent was a calibration "
        "tweak. Plan top-down.\n"
        "\n"
        "- **L1 family swap** — switch among the 9 canonical families "
        "(ts_corr_pv / intraday_range / vwap_dev / volume_rank / open_gap / "
        "humped / multi_signal / sector_relative / fundamental_ratio). "
        "Largest signal change. Use when stuck or the pool's dominant family "
        "is saturated. Adding a `group_*(_, sector)` wrapper around a previously "
        "non-grouped expression typically lands here (it changes the family "
        "to `sector_relative`).\n"
        "- **L2 skeleton swap** — same family, different operator multiset "
        "(`ts_corr(close, volume, 20)` → `ts_corr(close, volume, 20) * "
        "ts_zscore(volume, 20)`). Bypasses WQ self-correlation rejection "
        "on near-identical operator multisets — often the cheapest way "
        "out of `STUCK IN OPERATOR SKELETON` warnings.\n"
        "- **L3 calibration** — same family AND same operator multiset, "
        "change ONLY numeric constants (windows / exponents / clamp bounds): "
        "`ts_rank(close, 20)` → `ts_rank(close, 60)`. Tunes signal-to-noise "
        "without restructuring.\n"
        "- **L4 refinement** — same family + skeleton + numbers, swap fields "
        "(`close` → `vwap`). Smallest edit. Note: WQ self-correlation will "
        "treat L4 edits as near-duplicates of the parent and likely reject.\n"
        "\n"
        "**Bubble-up rule**: 2 consecutive L4 edits MUST be followed by L2 "
        "or L1 — pure field swaps in the same operator stack produce signals "
        "that WQ self-correlation rejects as duplicates."
    )


def _recent_altitude_distribution_hint(
    tried_records: list[dict],
    *,
    window: int = 8,
) -> str:
    """Distribution of altitude transitions across the last N attempts.

    Fires two structured advisories:

      * **bubble-up advisory** — last 2+ classified transitions were L4.
        The agent must escalate to L2 / L1 next.
      * **no-high-altitude advisory** — 0 L1 + 0 L2 in the entire window.
        Forces at least one family swap in the next 3 attempts.

    Empty string when fewer than 3 records — not enough signal."""
    if len(tried_records) < 3:
        return ""
    recent = sorted(tried_records, key=lambda r: r.get("ts", 0))[-window:]
    altitudes: list[str] = []
    for prev, curr in zip(recent, recent[1:]):
        altitudes.append(_classify_edit_altitude(
            prev.get("expr") or "", curr.get("expr") or "",
        ))
    if not altitudes:
        return ""
    counts = Counter(altitudes)
    last_two = altitudes[-2:]
    bubble_up = len(last_two) >= 2 and all(a == "L4" for a in last_two)
    no_high = counts.get("L1", 0) == 0 and counts.get("L2", 0) == 0

    lines = [
        f"### 🪜 RECENT EDIT-ALTITUDE DISTRIBUTION (last {len(altitudes)} transitions)",
        "",
        f"L1 family-swap × {counts.get('L1', 0)} | "
        f"L2 skeleton-swap × {counts.get('L2', 0)} | "
        f"L3 calibration × {counts.get('L3', 0)} | "
        f"L4 refinement × {counts.get('L4', 0)} | "
        f"— dup × {counts.get('—', 0)}",
    ]
    if bubble_up:
        lines.extend([
            "",
            "🛑 **BUBBLE-UP ADVISORY**: last 2+ edits were L4 (field swap "
            "only). Your next candidate **MUST** be L2 (different operator "
            "skeleton) or L1 (different family). Continuing L4 leaves the "
            "operator stack identical and WQ self-correlation will reject.",
        ])
    if no_high:
        lines.extend([
            "",
            "⚠️ **No L1/L2 edits in this window** — you've been only "
            "re-tuning windows / fields inside one structural skeleton. "
            "Force at least 1 family swap (L1) in the next 3 attempts.",
        ])
    return "\n".join(lines)


def _operator_skeleton(expr: str) -> str:
    """Reduce an expression to its operator multiset signature.

    Drops fields, numbers, and infix operators — keeps only the function
    names. Two expressions collapse to the same skeleton if they call
    exactly the same operators in the same multiplicities, even if fields
    or constants differ.
    """
    if not expr:
        return ""
    ops = _SKELETON_OP_PATTERN.findall(expr.lower())
    if not ops:
        return ""
    counts = Counter(ops)
    return "|".join(f"{op}x{n}" for op, n in sorted(counts.items()))


def _slot_key(expr: str) -> Optional[tuple[str, str, frozenset[str], frozenset[str]]]:
    """A "slot" = ``(family, operator_skeleton, field_kinds, fields)``.

    Codex review R2-#6: original ``(skeleton, field_kinds)`` collapsed
    *every* fundamental ratio (sales/assets, debt/equity, fcf/cap) to
    the same slot, falsely flagging legitimate FUND-family exploration
    as anti-flip oscillation. Including ``infer_family`` and the
    canonical-field set distinguishes:

      * ``rank(sales/assets)``      → (`fundamental_ratio`, `rankx1`, {FUND}, {sales, assets})
      * ``rank(debt/equity)``       → (`fundamental_ratio`, `rankx1`, {FUND}, {debt, equity})
      * ``rank(close - vwap)``      → (`vwap_dev`,           `rankx1`, {P/V},  {close, vwap})

    Two expressions collapse only when they share BOTH operator multiset
    AND family AND the same canonical-field set — much closer to
    "actual signal duplicate".

    Returns ``None`` when ``expr`` has no operators.
    """
    skel = _operator_skeleton(expr)
    if not skel:
        return None
    fields = _extract_field_set(expr)
    kinds = frozenset(_FIELD_KIND.get(f, "other") for f in fields)
    family = infer_family(expr)
    return (family, skel, kinds, fields)


def _record_slot_key(r: dict) -> Optional[tuple[str, str, frozenset[str], frozenset[str]]]:
    """Slot-key adapter for :func:`find_recent_revisits` — operates on
    a tried_log record dict."""
    return _slot_key(r.get("expr") or "")


def _slot_cooldown_hint(
    tried_records: list[dict],
    *,
    window: int = 5,
    min_revisits: int = 3,
    epsilon: float = 0.05,
    consecutive_non_improvements: int = 2,
) -> str:
    """Anti-flip cool-down: flag slots stuck in noisy-no-progress local search.

    Codex review #9 + #10 hardening — replaces the prior naive
    ``latest_fi ≤ prior_max`` rule which was too eager:

    A slot is considered "in cool-down" when **all** of the following hold:

      1. It was visited ≥ ``min_revisits`` times in the last ``window``
         records (flip-back-in pattern, even if visits aren't strictly
         consecutive).
      2. Visits with finite ``fitness`` and ``status == "COMPLETE"`` are
         used; rows without metrics or failed simulates are ignored
         (they don't count as "no progress" — the agent never had data).
      3. The last ``consecutive_non_improvements`` visits each fall
         **at or below** the running best-so-far minus ``epsilon`` —
         i.e. the noisy bouncing has plateaued, not just dipped once.

    The default ``min_revisits=3`` deliberately tolerates legitimate
    fundamental-ratio exploration (sales/assets → debt/equity → fcf/cap
    is a 3-visit pattern that should NOT be frozen unless the *pattern*
    of fitness regressions is real). Slots are still listed worst-first.
    """
    revisits = find_recent_revisits(
        tried_records, _record_slot_key,
        window=window, min_revisits=min_revisits,
    )
    if not revisits:
        return ""

    def _completed_fi(r: dict) -> Optional[float]:
        """Return fitness only when row is COMPLETE and metric is finite."""
        if r.get("status") and r.get("status") != "COMPLETE":
            return None
        v = r.get("fitness")
        if not isinstance(v, (int, float)):
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if f != f or f in (float("inf"), float("-inf")):  # NaN / inf
            return None
        return f

    flagged: list[tuple[tuple[str, frozenset[str]], list[dict], list[float]]] = []
    for slot, visits in revisits.items():
        completed_fi = [_completed_fi(v) for v in visits]
        # Drop None-fitness rows from the trend check — they neither prove
        # nor disprove progress.
        fi_series = [f for f in completed_fi if f is not None]
        if len(fi_series) < min_revisits:
            continue
        # Best-so-far at each step
        running_best: list[float] = []
        for f in fi_series:
            running_best.append(f if not running_best else max(running_best[-1], f))
        # Last K visits each ≤ best_at_that_step - epsilon (plateau)
        K = min(consecutive_non_improvements, len(fi_series))
        if K < 1:
            continue
        plateau = all(
            f <= running_best[-1] - epsilon or
            (i > 0 and f <= running_best[i - 1] - epsilon)
            for i, f in enumerate(fi_series[-K:], start=len(fi_series) - K)
        )
        # Codex review R2 answer #3: replace exact-equality `flat` clause
        # with an epsilon-band so floating-point noise (1.50000001 vs
        # 1.49999999) doesn't silently disable the flat-stuck detection.
        last_K = fi_series[-K:]
        flat_stuck = (
            (max(last_K) - min(last_K)) <= epsilon
            and last_K[-1] >= running_best[-1] - epsilon
        )
        if plateau or (flat_stuck and len(fi_series) >= min_revisits + 1):
            flagged.append((slot, visits, fi_series))

    if not flagged:
        return ""

    # Show worst (most-revisited) slots first; tie-break by largest gap
    # between best and latest.
    flagged.sort(
        key=lambda sv: (
            -len(sv[1]),
            -(max(sv[2]) - sv[2][-1]) if sv[2] else 0.0,
            sv[0][0],
        )
    )

    lines = [
        f"### 🧊 SLOT COOL-DOWN — anti-flip guard ({len(flagged)} slot(s) frozen)",
        "",
        f"_A slot is `(family, operator-skeleton, field-kinds, fields)`. Slots "
        f"below were revisited ≥ {min_revisits}× in the last {window} attempts "
        f"and the last {consecutive_non_improvements} completed fitness values "
        f"failed to beat their running best by ε={epsilon:.2f}. **Bubble up to "
        f"L2 (different skeleton) or L1 (different family) for these slots.**_",
        "",
        "| family | skeleton | field-kinds | fields | visits | best fi | latest fi |",
        "|---|---|---|---|---|---|---|",
    ]
    for (family, skel, kinds, fields), visits, fi_series in flagged[:5]:
        kinds_str = "+".join(sorted(kinds)) or "—"
        fields_str = ",".join(sorted(fields))[:40] or "—"
        skel_short = skel.replace("|", " + ")[:50]
        best_fi = max(fi_series) if fi_series else float("-inf")
        latest_fi = fi_series[-1] if fi_series else float("-inf")
        best_s = f"{best_fi:.2f}" if best_fi != float("-inf") else "-"
        latest_s = f"{latest_fi:.2f}" if latest_fi != float("-inf") else "-"
        lines.append(
            f"| `{family}` | `{skel_short}` | {kinds_str} | `{fields_str}` | "
            f"{len(visits)} | {best_s} | {latest_s} |"
        )
    return "\n".join(lines)


def _tried_skeleton_concentration_hint(
    tried_records: list[dict],
    *,
    window: int = 10,
    threshold: float = 0.5,
) -> str:
    """Hint when ≥ ``threshold`` of recent attempts share one operator skeleton.

    Catches the failure mode that the family-rotation hint misses:
    same family, different fields, but **identical operator stack** —
    those will all hit WQ self-correlation rejection on submit.
    """
    if len(tried_records) < window:
        return ""
    recent = sorted(tried_records, key=lambda r: r.get("ts", 0), reverse=True)[:window]
    skeletons = [_operator_skeleton(r.get("expr") or "") for r in recent]
    skeletons = [s for s in skeletons if s]
    if not skeletons:
        return ""
    counts = Counter(skeletons)
    top_sk, top_n = counts.most_common(1)[0]
    if top_n / len(skeletons) < threshold:
        return ""
    pretty = top_sk.replace("|", " + ")
    return (
        f"_🛑 STUCK IN OPERATOR SKELETON `{pretty}` ({top_n}/{len(skeletons)} "
        f"of recent attempts share this exact multiset). Even with different "
        f"fields, WQ's self-correlation will see the same signal. CHANGE THE "
        f"OPERATOR STACK — drop / add an operator, swap multiplication to "
        f"subtraction, wrap with `if_else(...)` for regime-conditional, or "
        f"compose with `group_zscore(_, sector)` / `ts_zscore(_, 252)` for a "
        f"completely different signal shape._"
    )


def _tried_family_concentration_hint(
    tried_records: list[dict],
    *,
    window: int = 10,
    threshold: float = 0.6,
) -> str:
    """Hard rotation hint: if recent attempts are dominated by one family,
    forbid that family for the next handful of candidates.
    """
    if len(tried_records) < window:
        return ""
    recent = sorted(tried_records, key=lambda r: r.get("ts", 0), reverse=True)[:window]
    families = [infer_family(r.get("expr") or "") for r in recent if r.get("expr")]
    if not families:
        return ""
    counts = Counter(families)
    top_fam, top_count = counts.most_common(1)[0]
    if top_count / window < threshold:
        return ""
    examples = _FAMILY_ANTI_EXAMPLES.get(
        top_fam, "any of the 8 canonical families OTHER than this one"
    )
    return (
        f"_🛑 STUCK IN `{top_fam}` ({top_count}/{window} of recent attempts). The "
        f"mutation engine is over-twiddling parameters inside this family. "
        f"YOUR NEXT 5 simulate calls MUST come from a DIFFERENT family. "
        f"Try: {examples}. Re-entering `{top_fam}` before 5 cross-family "
        f"simulations will be flagged as a session-rule violation._"
    )


def _build_prior_knowledge_block(
    tag: str,
    *,
    max_pool: int = 20,
    max_tried: int = 60,
) -> str:
    """Render cross-loop knowledge: passing pool + recent tried_exprs +
    cross-over candidates + mutation hints + submit rejections."""
    parts: list[str] = []

    pool = AlphaPool(alpha_pool_path(tag))
    if len(pool):
        # Codex review R3-#1: bucket statuses explicitly so blocked alphas
        # land in the DO-NOT-REPEAT table, not "Submission Pending". Without
        # this the agent reads LOCAL_BLOCKED / SELF_CORR_BLOCKED structures
        # as still-pending and re-tries the same shape — breaks the learning
        # loop the gate states were supposed to enable.
        _FAILURE_STATUSES = {
            "REJECTED", "LOCAL_BLOCKED", "SELF_CORR_BLOCKED",
            "VERIFICATION_FAILED",
        }
        active = [
            e for e in pool.entries
            if getattr(e, "verified_status", "QUEUED") == "ACTIVE"
        ]
        rejected = [
            e for e in pool.entries
            if (
                getattr(e, "verified_status", "QUEUED") in _FAILURE_STATUSES
                or (
                    getattr(e, "verified_status", "QUEUED") == "UNSUBMITTED"
                    and getattr(e, "rejection_reasons", [])
                )
            )
        ]
        queued = [
            e for e in pool.entries
            if e not in active and e not in rejected
        ]

        if active:
            top_active = sorted(active, key=lambda e: -e.fitness)[:max_pool]
            active_families = [infer_family(e.expr or "") for e in top_active]
            hint = _family_diversity_hint(active_families)
            lines = [
                "### ✅ ACTIVE Submitted Alphas (TAG=" + tag + ")",
                "",
                f"_{len(active)} alpha(s) verified ACTIVE on WQ. These earn rewards. Avoid"
                f" submitting near-duplicates — WQ self-correlation will reject._",
            ]
            if hint:
                lines.extend(["", hint])
            lines.extend([
                "",
                "| alpha_id | family | expr | sh | fi | to |",
                "|---|---|---|---|---|---|",
            ])
            for e, fam in zip(top_active, active_families):
                expr = (e.expr or "")[:75].replace("|", "/")
                lines.append(
                    f"| {e.alpha_id} | `{fam}` | `{expr}` | {e.sharpe:.2f} | "
                    f"{e.fitness:.2f} | {e.turnover:.2f} |"
                )
            parts.append("\n".join(lines))

        if rejected:
            recent_rejected = sorted(
                rejected, key=lambda e: -getattr(e, "verified_at", 0),
            )[:max_pool]
            lines = [
                "### 🚫 SUBMIT FAILURES — DO NOT REPEAT THESE STRUCTURES",
                "",
                f"_{len(rejected)} alpha(s) passed local quality gate (sh≥1.25 fi≥1.0)_"
                f" _BUT WERE REJECTED BY WQ. The most common reason is_"
                f" _SELF_CORRELATION ≥ 0.7 against an ACTIVE alpha. If your new_"
                f" _candidate is similar to any of these, IT WILL ALSO BE REJECTED._",
                "",
                "| alpha_id | family | expr | sh | fi | to | failed_check |",
                "|---|---|---|---|---|---|---|",
            ]
            for e in recent_rejected:
                expr = (e.expr or "")[:65].replace("|", "/")
                fam = infer_family(e.expr or "")
                fails = getattr(e, "rejection_reasons", []) or []
                fail_str = ", ".join(
                    f"{r.get('name')}={r.get('value')}" if isinstance(r, dict) else str(r)
                    for r in fails[:2]
                ) or "?"
                lines.append(
                    f"| {e.alpha_id} | `{fam}` | `{expr}` | {e.sharpe:.2f} | "
                    f"{e.fitness:.2f} | {e.turnover:.2f} | {fail_str} |"
                )
            parts.append("\n".join(lines))

        if queued:
            top_queued = sorted(queued, key=lambda e: -e.fitness)[:max_pool]
            lines = [
                "### ⏳ Submission Pending (verification not yet run)",
                "",
                "| alpha_id | expr | sh | fi | to |",
                "|---|---|---|---|---|",
            ]
            for e in top_queued:
                expr = (e.expr or "")[:80].replace("|", "/")
                lines.append(
                    f"| {e.alpha_id} | `{expr}` | {e.sharpe:.2f} | "
                    f"{e.fitness:.2f} | {e.turnover:.2f} |"
                )
            parts.append("\n".join(lines))

    tried = read_tried(tried_exprs_path(tag), tail=max_tried * 4)
    if tried:
        # 100%-action-space push — name the under-explored operators / fields
        # ABOVE the family/skeleton hints so the agent reads "what to add"
        # before "what to avoid".
        coverage_hint = _unexplored_space_hint(tried)
        if coverage_hint:
            parts.append(coverage_hint)
        # Altitude taxonomy + recent-edit distribution come BEFORE concentration
        # hints so the agent has L1-L4 definitions when reading "STUCK IN
        # OPERATOR SKELETON" / "STUCK IN family" warnings below.
        parts.append(_altitude_taxonomy_block())
        altitude_hint = _recent_altitude_distribution_hint(tried)
        if altitude_hint:
            parts.append(altitude_hint)
        # Slot cool-down catches anti-flip oscillation in a tighter window
        # (5 records vs the 10 used by skeleton/family concentration). Fires
        # only when the same slot was revisited without fitness improvement —
        # the actionable signal that escalation is required.
        cooldown_hint = _slot_cooldown_hint(tried)
        if cooldown_hint:
            parts.append(cooldown_hint)
        rotation_hint = _tried_family_concentration_hint(tried)
        if rotation_hint:
            parts.append(rotation_hint)
        skeleton_hint = _tried_skeleton_concentration_hint(tried)
        if skeleton_hint:
            parts.append(skeleton_hint)

        parts.append(
            "### Recently Attempted Expressions (latest result per expr)\n\n"
            + format_for_prompt(tried, max_rows=max_tried)
        )

        segments = extract_top_segments(
            tried, min_score=30, top_n=5, diversify_by_family=True,
        )
        cross_block = format_crossover_block(segments)
        if cross_block:
            parts.append(cross_block)

        mutation_block = render_top_failures_block(tried, top_n=3)
        if mutation_block:
            parts.append(mutation_block)

    if not parts:
        return "_(no prior loop data — fresh start)_"
    return "\n\n".join(parts)
