"""WorldQuant BRAIN FASTEXPR operator and field reference.

Used to validate LLM-generated expressions before submitting to the API
and to inject into prompts as a constraint list.
"""
from __future__ import annotations

import re
from typing import Optional

# ── Time-series operators ──────────────────────────────────────────────────
OPERATORS_TIME_SERIES: list[str] = [
    "ts_mean", "ts_std", "ts_rank", "ts_sum", "ts_min", "ts_max",
    "ts_product", "ts_skewness", "ts_kurtosis", "ts_zscore",
    "ts_delta", "ts_delay", "ts_decay_linear", "ts_decay_exp",
    "ts_regression_slope", "ts_regression_intercept",
    "ts_corr", "ts_covariance",
    "ts_arg_min", "ts_arg_max", "ts_count",
]

# ── Cross-sectional operators ──────────────────────────────────────────────
OPERATORS_CROSS_SECTIONAL: list[str] = [
    "rank", "group_rank", "group_mean", "group_sum", "group_std",
    "group_zscore", "group_neutralize", "group_count",
    "scale", "signed_power",
]

# ── Math / elementwise operators ───────────────────────────────────────────
OPERATORS_MATH: list[str] = [
    "abs", "log", "sqrt", "sign", "power", "exp",
    "min", "max", "sum", "mean",
    "if_else", "clamp",
    "correlation", "covariance",
    "returns",
]

ALL_OPERATORS: set[str] = (
    set(OPERATORS_TIME_SERIES)
    | set(OPERATORS_CROSS_SECTIONAL)
    | set(OPERATORS_MATH)
)

# ── Price/Volume fields ─────────────────────────────────────────────────────
FIELDS_PRICE_VOLUME: list[str] = [
    "open", "close", "high", "low", "volume", "vwap",
    "adv20", "adv60", "adv120", "adv180",
    "cap", "returns",
]

# ── Fundamental fields ──────────────────────────────────────────────────────
FIELDS_FUNDAMENTAL: list[str] = [
    "book_to_price", "earnings_yield", "sales_yield",
    "ebitda_yield", "fcf_yield", "dividend_yield",
    "debt_to_equity", "debt_to_assets",
    "roe", "roa", "gross_margin", "net_margin",
    "revenue_growth", "earnings_growth",
    "pe_ratio", "pb_ratio", "ps_ratio",
    "analyst_consensus", "analyst_revision",
]

# ── Sector/industry classification ─────────────────────────────────────────
FIELDS_CLASSIFICATION: list[str] = [
    "sector", "subindustry", "industry",
]

ALL_FIELDS: set[str] = (
    set(FIELDS_PRICE_VOLUME)
    | set(FIELDS_FUNDAMENTAL)
    | set(FIELDS_CLASSIFICATION)
)

# ── Tokenizer ───────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokens(expr: str) -> list[str]:
    return _TOKEN_RE.findall(expr)


# ── Validator ────────────────────────────────────────────────────────────────
_RESERVED_WORDS = {"if", "else", "and", "or", "not", "in", "True", "False", "None"}
_NUMERIC_LIKE = re.compile(r"^\d+$")


def validate_expression(expr: str) -> list[str]:
    """Light token-level pre-check. Returns list of error strings (empty = ok)."""
    errors: list[str] = []
    stripped = expr.strip()
    if not stripped:
        errors.append("Empty expression")
        return errors

    tokens = _tokens(stripped)
    if not tokens:
        errors.append("No recognisable tokens found")
        return errors

    # Check for obvious Python-only constructs that WQ doesn't support
    for bad in ("import", "def ", "class ", "lambda ", "print(", "exec(", "eval("):
        if bad in stripped:
            errors.append(f"Forbidden construct: {bad!r}")

    # Identify unknown identifiers (not operators, not fields, not numbers)
    unknown: list[str] = []
    for tok in tokens:
        if tok in _RESERVED_WORDS:
            continue
        if _NUMERIC_LIKE.match(tok):
            continue
        if tok in ALL_OPERATORS:
            continue
        if tok in ALL_FIELDS:
            continue
        # Single-letter variables (e.g. x in lambda-like expressions) are fine
        if len(tok) == 1:
            continue
        unknown.append(tok)

    if unknown:
        # Warn but don't hard-fail — WQ may have data fields we don't list
        errors.append(f"Unrecognised identifiers (may be unlisted fields): {unknown[:5]}")

    return errors


def operators_prompt_block() -> str:
    """Return a compact prompt snippet listing key operators and fields."""
    ts_ops = ", ".join(OPERATORS_TIME_SERIES[:10]) + ", ..."
    cs_ops = ", ".join(OPERATORS_CROSS_SECTIONAL)
    math_ops = ", ".join(OPERATORS_MATH[:8]) + ", ..."
    pv = ", ".join(FIELDS_PRICE_VOLUME)
    fund = ", ".join(FIELDS_FUNDAMENTAL[:10]) + ", ..."
    return f"""\
=== WorldQuant FASTEXPR Reference ===
Time-series  : {ts_ops}
Cross-section: {cs_ops}
Math         : {math_ops}
Price/Volume : {pv}
Fundamental  : {fund}

Rules:
- Output must be a cross-sectional SCORE per stock (positive=long, negative=short)
- Use rank() or group_rank() as the outermost wrapper to normalise output
- Delay=1 means signal computed at T is filled at T+1 open (no look-ahead)
- Do NOT use Python-only syntax (lambda, import, def, class)
- Do NOT reference data that would not exist at signal time
"""
