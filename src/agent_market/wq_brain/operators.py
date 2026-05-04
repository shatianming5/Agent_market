"""WorldQuant BRAIN FASTEXPR operator and field reference.

Used to validate LLM-generated expressions before submitting to the API
and to inject into prompts as a constraint list.
"""
from __future__ import annotations

import re
from typing import Optional

# ── Time-series operators ──────────────────────────────────────────────────
OPERATORS_TIME_SERIES: list[str] = [
    "ts_mean", "ts_rank", "ts_zscore", "ts_sum", "ts_max",
    "ts_delta", "ts_delay", "ts_decay_linear",
    "ts_corr",
]
OPERATORS_TIME_SERIES_UNAVAILABLE: list[str] = [
    "ts_std", "ts_min", "ts_product", "ts_skewness", "ts_kurtosis",
    "ts_decay_exp", "ts_regression_slope", "ts_regression_intercept",
    "ts_covariance", "ts_arg_min", "ts_arg_max", "ts_count",
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
]

ALL_OPERATORS: set[str] = (
    set(OPERATORS_TIME_SERIES)
    | set(OPERATORS_TIME_SERIES_UNAVAILABLE)
    | set(OPERATORS_CROSS_SECTIONAL)
    | set(OPERATORS_MATH)
)

# ── Price/Volume fields ─────────────────────────────────────────────────────
FIELDS_PRICE_VOLUME: list[str] = [
    "open", "close", "high", "low", "volume", "vwap",
    "adv20",
    "returns",  # "adv60","adv120","adv180","cap" NOT confirmed available
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

UNAVAILABLE_FUNDAMENTAL_FIELDS: frozenset[str] = frozenset(FIELDS_FUNDAMENTAL)
UNAVAILABLE_PRICE_FIELDS: frozenset[str] = frozenset(["adv60", "adv120", "adv180", "cap"])
UNAVAILABLE_TS_OPERATORS: frozenset[str] = frozenset(OPERATORS_TIME_SERIES_UNAVAILABLE)

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
        errors.append(f"Unrecognised identifiers (may be unlisted fields): {unknown[:5]}")

    # Block unavailable fundamental fields
    blocked_fund = [tok for tok in tokens if tok in UNAVAILABLE_FUNDAMENTAL_FIELDS]
    if blocked_fund:
        errors.append(f"Unavailable fundamental field(s): {blocked_fund[:5]} (not on this account)")

    # Block unavailable price fields
    blocked_pv = [tok for tok in tokens if tok in UNAVAILABLE_PRICE_FIELDS]
    if blocked_pv:
        errors.append(f"Unavailable price field(s): {blocked_pv[:5]} (not on this account)")

    # Block unavailable time-series operators
    blocked_ts = [tok for tok in tokens if tok in UNAVAILABLE_TS_OPERATORS]
    if blocked_ts:
        errors.append(f"Unavailable operator(s): {blocked_ts[:5]} (not on this account)")

    return errors


def operators_prompt_block() -> str:
    """Return a compact prompt snippet listing key operators and fields."""
    ts_ops = ", ".join(OPERATORS_TIME_SERIES) + "  (ONLY these! ts_std/ts_min NOT available)"
    cs_ops = ", ".join(OPERATORS_CROSS_SECTIONAL)
    math_ops = ", ".join(OPERATORS_MATH[:8]) + ", ..."
    pv = ", ".join(f for f in FIELDS_PRICE_VOLUME if f != "returns")
    return f"""\
=== WorldQuant FASTEXPR Reference ===

OPERATORS (called with arguments, e.g. rank(close)):
  Time-series  : {ts_ops}
  Cross-section: {cs_ops}
  Math/Scalar  : {math_ops}

AVAILABLE DATA FIELDS (plain tokens, NO parentheses ever):
  Price        : open, close, high, low, vwap
  Volume       : volume, adv20  (adv60/adv120/adv180 NOT available!)
  Return       : returns  (1-day return — use ts_mean(returns,N) for N-day avg)
  Grouping     : sector, industry, subindustry  (for group_* operators only)

*** FUNDAMENTAL FIELDS ARE NOT AVAILABLE (roe, book_to_price, etc.) ***

*** GROUP OPERATORS NOTE ***
  group_rank(x, group) and group_zscore(x, group) take 2 args — CONFIRMED WORKING.
  group_mean/group_sum/group_count require 3 args — AVOID, use group_rank/group_zscore instead.

*** CRITICAL SYNTAX RULES ***
1. Fields are plain tokens — NEVER add () after them.
   CORRECT: rank(returns)        WRONG: returns(20)
   CORRECT: ts_mean(returns, 60) WRONG: returns(60)
2. Use rank() or group_rank() as outermost wrapper to normalize output.
3. Target turnover < 0.60 (daily). HIGH_TURNOVER limit is 0.70 — crossing it kills fitness.
   Use LONG windows (20-120 days). BAD: rank(-ts_zscore(returns,5))  GOOD: rank(ts_mean(returns,60))
4. Delay=1: signal at T, fill at T+1 open (no look-ahead bias).
5. Do NOT use Python syntax: lambda, import, def, class, print.
"""
