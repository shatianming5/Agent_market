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

# WQ-specific transforms (apply outside ts_/group_/math families).
# `hump(alpha, hump_value=0.01)` is the canonical turnover-reduction wrapper:
# clips per-period weight changes smaller than hump_value, slashing turnover
# at modest sharpe cost. Often the difference between fi=0.7 and fi=1.0+.
OPERATORS_TRANSFORM = ["hump"]

# UNAVAILABLE on free tier — DO NOT USE
OPERATORS_TS_UNAVAILABLE = [
    "ts_std", "ts_min", "ts_product", "ts_skewness", "ts_kurtosis",
    "ts_decay_exp", "ts_regression_slope", "ts_regression_intercept",
    "ts_covariance", "ts_arg_min", "ts_arg_max", "ts_count",
]

FIELDS_PRICE_VOLUME = [
    "open", "close", "high", "low", "volume", "vwap",
    "adv20", "adv60", "adv120", "adv180",
    "returns",
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

ALL_OPERATORS = OPERATORS_TS + OPERATORS_CS + OPERATORS_MATH + OPERATORS_TRANSFORM
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

AVAILABLE Turnover-Reduction Transform ({len(OPERATORS_TRANSFORM)}):
  {", ".join(OPERATORS_TRANSFORM)}

UNAVAILABLE Time-Series — DO NOT USE ({len(OPERATORS_TS_UNAVAILABLE)}):
  {", ".join(OPERATORS_TS_UNAVAILABLE)}

AVAILABLE Fields:
  Price/Volume:  {", ".join(FIELDS_PRICE_VOLUME)}
  ADV variants:  adv20 (20-day), adv60 (60-day), adv120, adv180 — pick longer window for smoother liquidity normalization (lower turnover)
  Group:         {", ".join(FIELDS_GROUP)}

UNAVAILABLE Fields (fundamental) — DO NOT USE ({len(FIELDS_FUNDAMENTAL_UNAVAILABLE)}):
  {", ".join(FIELDS_FUNDAMENTAL_UNAVAILABLE)}

=== Turnover Reduction (KEY for passing fitness gate) ===

Past best alpha hit sh=1.47 fi=0.77 to=0.46 — the turnover (0.46) caps fitness.
Two main levers to slash turnover (and thus boost fitness):

(a) `hump(<alpha>, hump_value)` — clips per-period weight deltas below
    hump_value. Typical reductions:
        hump(<alpha>, 0.005) → ~30% turnover cut
        hump(<alpha>, 0.01)  → ~50% turnover cut (most common)
        hump(<alpha>, 0.05)  → ~80% turnover cut, larger sharpe hit
    Example: rank(ts_rank(close,252) * (-ts_delta(close,3)/close))  # to=0.46
             hump(rank(ts_rank(close,252) * (-ts_delta(close,3)/close)), 0.01)  # to≈0.20

(b) Use longer ADV/window: replace `adv20` with `adv60`/`adv120`, replace
    `ts_*(field, 5)` with `ts_*(field, 20)` etc. — same shape, less churn.

(c) Wrap with `ts_decay_linear(<alpha>, N)` for N=10-20 — smooths weights.

=== Multi-Line Binding (FASTEXPR supports `;` separated statements) ===

When alphas get long, USE BINDINGS instead of deep nesting. The last
statement is the alpha output:

  x = ts_mean(close, 20);
  y = ts_zscore(close, 20);
  z = group_zscore(returns, sector);
  rank(x * y - 0.5 * z)

  # equivalent flat form (harder to debug):
  rank(ts_mean(close,20) * ts_zscore(close,20) - 0.5 * group_zscore(returns,sector))

=== Common Pitfalls ===
- Avoid nesting group_* INSIDE ts_* (causes timeouts):
    OK:  rank(group_zscore(ts_mean(returns,20), sector))
    BAD: rank(ts_mean(group_zscore(returns, sector), 20))
- All ts_* operators take (field, window). Window typically in [3, 240].
- rank() should usually be the OUTERMOST layer (or wrap with hump → rank).
- Quality gates: sharpe >= 1.25 AND fitness >= 1.0.
- ADV variants: adv60 / adv120 / adv180 give smoother liquidity than adv20
  but exist only on free tier — confirmed.
""".strip()
