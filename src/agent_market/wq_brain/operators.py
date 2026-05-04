"""FASTEXPR whitelist + simple validator.

Validator scope: token-level + balanced parens. The agent itself iterates
when WQ API rejects an expression — the validator is just a first-pass filter
to save WQ simulation budget on obviously broken expressions.
"""
from __future__ import annotations

import re

# AVAILABLE on free tier
OPERATORS_TS = [
    "ts_mean", "ts_rank", "ts_zscore", "ts_sum", "ts_max",
    "ts_delta", "ts_delay", "ts_decay_linear", "ts_corr",
]
OPERATORS_CS = [
    "rank", "group_rank", "group_mean", "group_sum", "group_std",
    "group_zscore", "group_neutralize", "group_count", "scale", "signed_power",
]
OPERATORS_MATH = [
    "abs", "log", "sqrt", "sign", "power", "exp",
    "min", "max", "sum", "mean", "if_else", "clamp",
    "correlation", "covariance",
]

# UNAVAILABLE on free tier — DO NOT USE
OPERATORS_TS_UNAVAILABLE = [
    "ts_std", "ts_min", "ts_product", "ts_skewness", "ts_kurtosis",
    "ts_decay_exp", "ts_regression_slope", "ts_regression_intercept",
    "ts_covariance", "ts_arg_min", "ts_arg_max", "ts_count",
]

FIELDS_PRICE_VOLUME = [
    "open", "close", "high", "low", "volume", "vwap", "adv20", "returns",
]
FIELDS_GROUP = ["sector", "industry", "subindustry"]
FIELDS_FUNDAMENTAL_UNAVAILABLE = [
    "book_to_price", "earnings_yield", "dividend_yield", "fcf_yield",
    "roe", "roa", "operating_margin", "gross_margin",
    "debt_to_equity", "current_ratio", "quick_ratio",
    "revenue_growth", "earnings_growth", "fcf_growth",
    "market_cap", "cap", "shares_out", "float_shares",
    "pe", "pb",
]

ALL_OPERATORS = OPERATORS_TS + OPERATORS_CS + OPERATORS_MATH
ALL_FIELDS = FIELDS_PRICE_VOLUME + FIELDS_GROUP

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PYTHON_FORBIDDEN = ("import ", "def ", "class ", "lambda", "print(", "exec(", "eval(", "__")


def validate_expression(expr: str) -> list[str]:
    """Returns list of error messages; empty list = OK."""
    errors: list[str] = []
    if not expr or not expr.strip():
        errors.append("empty expression")
        return errors

    for kw in _PYTHON_FORBIDDEN:
        if kw in expr:
            errors.append(f"forbidden Python syntax: {kw!r}")

    open_paren = expr.count("(")
    close_paren = expr.count(")")
    if open_paren != close_paren:
        errors.append(f"unbalanced parens: {open_paren} '(' vs {close_paren} ')'")

    tokens = set(_TOKEN_RE.findall(expr))
    for tok in tokens:
        if tok in OPERATORS_TS_UNAVAILABLE:
            errors.append(f"unavailable operator: {tok}")
        if tok in FIELDS_FUNDAMENTAL_UNAVAILABLE:
            errors.append(f"unavailable field: {tok}")

    return errors


def operators_prompt_block() -> str:
    """Markdown reference block — full disclosure, no truncation."""
    return f"""
=== FASTEXPR Operator Reference ===

AVAILABLE Time-Series ({len(OPERATORS_TS)}):
  {", ".join(OPERATORS_TS)}

AVAILABLE Cross-Sectional ({len(OPERATORS_CS)}):
  {", ".join(OPERATORS_CS)}

AVAILABLE Math ({len(OPERATORS_MATH)}):
  {", ".join(OPERATORS_MATH)}

UNAVAILABLE Time-Series — DO NOT USE ({len(OPERATORS_TS_UNAVAILABLE)}):
  {", ".join(OPERATORS_TS_UNAVAILABLE)}

AVAILABLE Fields:
  Price/Volume: {", ".join(FIELDS_PRICE_VOLUME)}
  Group: {", ".join(FIELDS_GROUP)}

UNAVAILABLE Fields (fundamental) — DO NOT USE ({len(FIELDS_FUNDAMENTAL_UNAVAILABLE)}):
  {", ".join(FIELDS_FUNDAMENTAL_UNAVAILABLE)}

=== Common Pitfalls ===
- Avoid nesting group_* INSIDE ts_* (causes timeouts):
    OK:  rank(group_zscore(ts_mean(returns,20), sector))
    BAD: rank(ts_mean(group_zscore(returns, sector), 20))
- All ts_* operators take (field, window). Window typically in [3, 240].
- rank() should usually be the OUTERMOST layer.
- Quality gates: sharpe >= 1.25 AND fitness >= 1.0.
""".strip()
