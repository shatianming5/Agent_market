"""Polished LLM factor miner with iterative generation + review loop.

Key improvements over the original single-pass llm.py:
  1. MULTI-ROUND LOOP: round N uses the top performers from rounds 0..N-1
     as in-context exemplars, and the worst failures as "avoid patterns".
  2. REVIEW PASS: a second LLM call critiques each generated factor for
     (a) semantic meaning, (b) overfit risk, (c) redundancy with existing.
  3. PERFORMANCE-AWARE PROMPTING: injects concrete IC / gain statistics of
     previously-validated factors so LLM sees what actually works.
  4. STRUCTURED FAILURE PATTERNS: feeds back explicit descriptions of factor
     types that historically degraded OOS (e.g. sign-flipping, noise).

This module is intended to be driven by an outer orchestrator
(`scripts/mine_factors_llm.py`) that integrates scoring + review feedback
back into the prompt, round over round.
"""
from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class FactorExample:
    """Canonical representation of a validated factor used in prompts."""
    name: str
    expression: str
    category: str
    abs_ic: float
    oos_ic: float
    lgb_gain: Optional[float] = None
    description: str = ""
    raw_ic: Optional[float] = None
    clean_ic: Optional[float] = None
    neutralized_ic: Optional[float] = None
    residual_ic_ratio: Optional[float] = None
    exposure_r2: Optional[float] = None
    max_exposure_corr: Optional[float] = None


@dataclass
class FailurePattern:
    """A pattern that historically underperformed; avoid in new generations."""
    description: str
    example_expression: str
    why_it_failed: str


# ============================================================
# Failure knowledge base (seeded from our v1-v4 mining experience)
# ============================================================
KNOWN_FAILURES: List[FailurePattern] = [
    FailurePattern(
        description="Highly correlated variants of the same base idea",
        example_expression="z(ema_spread) - z(rsi_28)  vs  z(ema_spread) - z(mfi_28)",
        why_it_failed="Correlation > 0.9 across 30 candidates led to effective rank 5.2 in v3; "
                       "LightGBM treats them as redundant, hurting strategy performance.",
    ),
    FailurePattern(
        description="Factors with OOS-period selection bias",
        example_expression="rolling_min(roll_mean(tema_pct_12, 24), 12)  mined on OOS=full data",
        why_it_failed="f001-f008 claimed IC=0.20 but pooled actual OOS IC was -0.02 after honest "
                       "temporal split; selecting top-K by OOS performance = overfitting the test set.",
    ),
    FailurePattern(
        description="Complex nested expressions beyond depth 5",
        example_expression="ifelse(adx>25, ifelse(adx<20, rolling_min(roll_mean(rolling_min(ema_pct_12, 3), 72), 12), ...), ...)",
        why_it_failed="Complex depth-20+ factors survived IC screening but added noise to LightGBM; "
                       "30 deep v3 factors gave +0.15% strategy vs 13 simple g-factors giving +6.31%.",
    ),
    FailurePattern(
        description="Z-threshold signal simulation misleads model-driven strategies",
        example_expression="Signal simulator: (z > +1) → long; (z < -1) → short",
        why_it_failed="Annualized +111% by z-threshold ≠ +3.34% real walk-forward; LightGBM learns "
                       "non-linear decision boundaries, not threshold crossings.",
    ),
    FailurePattern(
        description="Pair-specific sign flipping",
        example_expression="Factor X: IC=+0.14 on BNB but -0.30 on LINK",
        why_it_failed="Factor works on specific regime that only some pairs experienced; "
                       "demands cross-sectional stability gate (same sign on >=8/10 pairs).",
    ),
]


# ============================================================
# Success patterns (what actually worked in our walk-forward)
# ============================================================
SUCCESS_PATTERNS: List[str] = [
    "Mean-reversion on z-normalized indicator spreads: z(A) - z(B) form "
    "(e.g. z(ema_spread) - z(mfi_28)) — the 13 g-factors beating all other sets",
    "Regime-conditional signals using ifelse(adx_14 > 25, trending_signal, chop_signal) "
    "— separates momentum vs mean-reversion regimes",
    "Multi-timeframe context: mtf4h_rsi_14 > 60 as regime gate for intra-hour factors "
    "— v4 mined 5+ factors using this pattern",
    "Simple composition: ema(f, 12) smoothing on raw spread factors improves Sharpe",
    "Cross-sectional stability requirement: >=8/10 pairs same-sign IC eliminates fragile signals",
]


def build_system_prompt(prompt_profile: str = "default") -> str:
    """System prompt establishing role + constraints."""
    profile = str(prompt_profile or "default").lower()
    if profile == "residual_alpha_v2":
        return textwrap.dedent("""
            You are a senior quantitative researcher at a cryptocurrency systematic trading firm.
            Your task is to propose residual alpha factors for 1-hour crypto panels.

            The objective is not raw IC. A candidate only matters if it keeps predictive
            power after cross-sectional winsorization, standardization, and neutralization
            against market, pair, volatility, liquidity, funding, multi-timeframe, and
            microstructure exposure groups.

            Optimize for neutralized_ic, residual_ic_ratio, low exposure_r2, low
            max_exposure_corr, and stable sign agreement across pairs. Reject expressions
            that are mainly market beta, pair beta, volatility, liquidity, funding, or
            simple cross-sectional return exposure in disguise.

            Every expression must include an orthogonality_claim explaining why the factor
            should survive exposure neutralization rather than merely restating a known
            exposure.
        """).strip()
    return textwrap.dedent("""
        You are a senior quantitative researcher at a cryptocurrency systematic trading firm.
        Your role is to propose predictive factor expressions that will be traded through a
        LightGBM ensemble on 1-hour candles across 10 major pairs (BTC, ETH, SOL, BNB, XRP,
        DOGE, ADA, AVAX, LINK, DOT).

        Your team has established through rigorous walk-forward backtesting what works and
        what doesn't. You will be shown:
          1. A library of validated factors with their real OOS IC and LightGBM gain
          2. Failure patterns from previous mining runs to AVOID
          3. Success patterns to BUILD ON
          4. Specific features and helper functions available

        You must generate expressions that are:
          - Cross-sectionally stable (same sign on >=8 of 10 pairs)
          - Distinct from existing library (correlation with top-20 < 0.7)
          - Simple enough for LightGBM to use (prefer depth <= 6)
          - Economically interpretable (every component has a reason)

        Reject any expression you would not defend to your portfolio manager.
    """).strip()


def build_generation_prompt_v2(
    *,
    feature_glossary: str,
    functions_doc: str,
    success_examples: Sequence[FactorExample],
    avoid_examples: Sequence[FactorExample],
    failure_patterns: Sequence[FailurePattern],
    round_idx: int,
    request_count: int,
    label_period: int,
    category_quota: Optional[Dict[str, int]] = None,
    prompt_profile: str = "default",
    rejection_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a round-N generation prompt with rich context.

    The prompt explicitly exposes:
      (a) best-performing reference factors (with their real metrics),
      (b) failed-factor patterns with the mechanistic reason they failed,
      (c) success patterns distilled from past rounds,
      (d) diversity/category quotas and correlation gate.
    """
    success_block = _format_examples(success_examples, tag="BEST-PERFORMING FACTORS (reference)")
    avoid_block = _format_examples(avoid_examples, tag="FACTORS TO AVOID (in library, avoid variants of these)")
    failure_block = _format_failures(failure_patterns)
    success_patterns_block = "\n".join(f"  • {p}" for p in SUCCESS_PATTERNS)
    quota_block = _format_quota(category_quota) if category_quota else ""
    profile = str(prompt_profile or "default").lower()
    residual_block = _format_residual_alpha_profile(rejection_summary) if profile == "residual_alpha_v2" else ""
    orthogonality_schema = (
        ',\n                "orthogonality_claim": "why this is not just market/pair/vol/liquidity/funding exposure"'
        if profile == "residual_alpha_v2" else ""
    )
    hard_constraints_extra = ""
    if profile == "residual_alpha_v2":
        hard_constraints_extra = textwrap.dedent("""
            7. Do not submit single-exposure template variants: plain `xs_ret_*`,
               `pair_beta_*`, raw volatility/liquidity/funding terms, or linear
               combinations of those terms without a surprise, lag, disagreement, or
               regime-conditioned residual mechanism.
            8. Each candidate must state an `orthogonality_claim`.
        """).rstrip()

    prompt = textwrap.dedent(f"""
        # ROUND {round_idx} — Generate {request_count} new factor expressions

        ## Trading context
        - Label horizon: {label_period} candles forward return
        - Timeframe: 1h bars, 10 pairs (spot data used for training; futures deployment)
        - Classes: down (<-0.5%) / flat / up (>+0.5%) at {label_period}h horizon

        ## Available features
        {feature_glossary}

        ## Allowed helper functions
        {functions_doc}

        ## Success patterns (build on these)
        {success_patterns_block}

        ## Failure patterns (AVOID these mistakes)
        {failure_block}

        {residual_block}

        ## Best validated factors in library (you already have these; aim to DIVERSIFY)
        {success_block}

        ## Avoid generating variants of these existing factors
        {avoid_block}

        {quota_block}

        ## Required output schema
        Respond with a single JSON object:
        {{"expressions": [
            {{
                "name": "short_snake_case_name",
                "expression": "Python-style expression using only listed features+functions",
                "category": "trend | momentum | volatility | volume | mean_reversion | regime",
                "description": "one-line Chinese or English summary",
                "rationale": "why this SHOULD predict forward returns (economic logic)",
                "diversification_claim": "why this is not redundant with the library above",
                "risk_note": "what market regime might break this factor"{orthogonality_schema}
            }}
        ]}}

        ## Hard constraints
        1. Use ONLY the listed feature columns; do not invent new symbol names.
        2. No forward-looking `shift(x, N)` with N < 0 (enforced by parser).
        3. Expression length < 450 chars.
        4. Include numerical stability guards (e.g. `... / (abs(x) + 1e-6)`).
        5. Each expression should be IMPLEMENTABLE and INTERPRETABLE.
        6. Prefer depth <= 6 over deeper nesting.
        {hard_constraints_extra}
    """).strip()

    return prompt


def build_review_prompt(
    *,
    candidate_expressions: Sequence[Dict[str, Any]],
    library_examples: Sequence[FactorExample],
) -> str:
    """Ask LLM to critique each candidate factor on 3 dimensions.

    Returns a prompt that requests a structured review per candidate:
      - semantic_valid: does the expression make economic sense?
      - overfit_risk: low / medium / high
      - redundancy_with_library: which library factor is closest (by idea), 0-1 overlap score
    """
    cand_block = "\n".join(
        f"  [{i}] name={c.get('name','?')}  expr={c.get('expression','')[:120]}"
        for i, c in enumerate(candidate_expressions)
    )
    lib_block = "\n".join(
        f"  L{i}: {ex.name} ({ex.category}, IC={ex.oos_ic:+.3f})  {ex.expression[:80]}"
        for i, ex in enumerate(library_examples[:10])
    )

    return textwrap.dedent(f"""
        You are a senior factor reviewer. Critique each candidate expression.

        ## Library reference (top 10 known-good factors)
        {lib_block}

        ## Candidates to review
        {cand_block}

        ## Output schema (one entry per candidate index)
        {{"reviews": [
            {{
                "index": 0,
                "semantic_valid": true,
                "overfit_risk": "low|medium|high",
                "redundancy_with_library": 0.0,         // 0 = novel, 1 = duplicate
                "closest_library_factor": "L3 or null",
                "recommended_action": "keep | reject | revise",
                "revision_suggestion": "specific edit if 'revise'",
                "one_line_verdict": "brief reason"
            }}
        ]}}
    """).strip()


def build_refinement_prompt(
    *,
    factor: FactorExample,
    critique: Dict[str, Any],
    feature_glossary: str,
) -> str:
    """Prompt for iterating on a factor that needs refinement."""
    return textwrap.dedent(f"""
        # Refine this factor based on reviewer feedback

        Original factor:
          name: {factor.name}
          expression: {factor.expression}
          current IC: {factor.oos_ic:+.3f}

        Reviewer feedback:
          overfit_risk: {critique.get('overfit_risk', '?')}
          redundancy: {critique.get('redundancy_with_library', '?')}
          verdict: {critique.get('one_line_verdict', '?')}
          suggestion: {critique.get('revision_suggestion', '?')}

        Available features:
        {feature_glossary}

        Propose ONE refined variant that addresses the feedback.
        Output: {{"name": "...", "expression": "...", "rationale": "..."}}
    """).strip()


# ============================================================
# Helpers
# ============================================================

def _format_examples(examples: Sequence[FactorExample], *, tag: str) -> str:
    if not examples:
        return f"## {tag}\n  (none)"
    lines = [f"## {tag}"]
    for ex in examples[:15]:
        gain_str = f"gain={ex.lgb_gain:.0f}" if ex.lgb_gain is not None else ""
        purity_parts = []
        if ex.raw_ic is not None:
            purity_parts.append(f"raw_ic={ex.raw_ic:+.4f}")
        if ex.clean_ic is not None:
            purity_parts.append(f"clean_ic={ex.clean_ic:+.4f}")
        if ex.neutralized_ic is not None:
            purity_parts.append(f"neutralized_ic={ex.neutralized_ic:+.4f}")
        if ex.residual_ic_ratio is not None:
            purity_parts.append(f"residual_ic_ratio={ex.residual_ic_ratio:.2f}")
        if ex.exposure_r2 is not None:
            purity_parts.append(f"exposure_r2={ex.exposure_r2:.2f}")
        if ex.max_exposure_corr is not None:
            purity_parts.append(f"max_exposure_corr={ex.max_exposure_corr:.2f}")
        purity_str = (" | " + ", ".join(purity_parts)) if purity_parts else ""
        lines.append(
            f"  • {ex.name:<8} [{ex.category:<14}] IC={ex.oos_ic:+.3f} {gain_str}"
            f"{purity_str}"
            f"\n      expr: {ex.expression[:120]}"
        )
    return "\n".join(lines)


def _format_failures(failures: Sequence[FailurePattern]) -> str:
    lines = []
    for i, f in enumerate(failures):
        lines.append(
            f"  {i+1}. {f.description}\n"
            f"     example:  {f.example_expression}\n"
            f"     why bad:  {f.why_it_failed}"
        )
    return "\n\n".join(lines)


def _format_quota(quota: Dict[str, int]) -> str:
    if not quota: return ""
    lines = ["## Category quota (aim for this distribution)"]
    for cat, n in quota.items():
        lines.append(f"  - {cat}: {n}")
    return "\n".join(lines)


def _format_residual_alpha_profile(rejection_summary: Optional[Dict[str, Any]]) -> str:
    counts = {}
    recent = []
    if isinstance(rejection_summary, dict):
        counts = rejection_summary.get("counts") or {}
        recent = rejection_summary.get("recent") or []
    count_line = ", ".join(f"{k}={int(v)}" for k, v in counts.items() if int(v or 0) > 0) or "none yet"
    recent_lines = []
    for row in recent[:8]:
        expr = str(row.get("expression") or "")[:90]
        reason = str(row.get("reason") or "")
        detail = str(row.get("detail") or "")[:90]
        suffix = f" ({detail})" if detail else ""
        recent_lines.append(f"  - {reason}: {expr}{suffix}")
    recent_block = "\n".join(recent_lines) if recent_lines else "  (none yet)"
    return textwrap.dedent(f"""
        ## Residual-alpha objective
        Rank candidates by neutralized_ic, not raw_ic. The target gates are:
        abs(neutralized_ic) >= 0.008, sign_agree >= 6,
        residual_ic_ratio >= 0.15, exposure_r2 <= 0.90,
        max_exposure_corr <= 0.50.

        Recent rejection summary:
        {count_line}

        Recent rejected examples:
        {recent_block}

        `low_feature_coverage` means the expression referenced a column that is too sparse
        or constant in the current panel; do not reuse omitted or low-coverage feature names.

        Prefer mechanisms that can survive neutralization:
        - microstructure surprise after controlling for market/pair/vol/liquidity/funding
        - funding dislocation after beta/vol control
        - liquidity shock reversal residual
        - cross-asset disagreement residual
        - regime-conditioned residual momentum
    """).strip()


def load_library_from_clean(path) -> List[FactorExample]:
    """Load the current clean factor library into FactorExample list."""
    import pathlib
    p = pathlib.Path(path)
    if not p.exists(): return []
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    out = []
    for e in data.get("expressions", []):
        out.append(FactorExample(
            name=e.get("name", "?"),
            expression=e.get("expression", ""),
            category=e.get("category", "other"),
            abs_ic=abs(float(e.get("oos_ic", e.get("train_ic", 0)))),
            oos_ic=float(e.get("oos_ic", e.get("train_ic", 0))),
            lgb_gain=float(e.get("gain", 0)) if e.get("gain") else None,
            description=e.get("description", ""),
        ))
    return out


def load_failures_from_dirty(path) -> List[FactorExample]:
    """Load factors from the dirty backup as 'avoid' examples."""
    import pathlib
    p = pathlib.Path(path)
    if not p.exists(): return []
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    out = []
    for e in data.get("expressions", []):
        out.append(FactorExample(
            name=e.get("name", "?"),
            expression=e.get("expression", ""),
            category=e.get("category", "other"),
            abs_ic=0.0, oos_ic=0.0, lgb_gain=None,
            description=e.get("description", ""),
        ))
    return out


def parse_llm_response(raw: str) -> List[Dict[str, Any]]:
    """Extract expressions list from LLM JSON output."""
    # Try to locate JSON block
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m: return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    return data.get("expressions", []) or []


def parse_review_response(raw: str) -> List[Dict[str, Any]]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m: return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    return data.get("reviews", []) or []


__all__ = [
    "FactorExample", "FailurePattern", "KNOWN_FAILURES", "SUCCESS_PATTERNS",
    "build_system_prompt", "build_generation_prompt_v2",
    "build_review_prompt", "build_refinement_prompt",
    "load_library_from_clean", "load_failures_from_dirty",
    "parse_llm_response", "parse_review_response",
]
