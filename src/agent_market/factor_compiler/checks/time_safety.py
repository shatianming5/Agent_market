from __future__ import annotations

import re
from typing import Dict, List, Optional

from agent_market.factor_compiler.api_models import ExprNode
from agent_market.factor_compiler.checks.leakage import check_no_negative_shift
from agent_market.factor_compiler.checks.types import CheckResult, fail, ok
from agent_market.factor_compiler.dsl.types import FactorType, infer_expr_type


# ---------------------------------------------------------------------------
# Timeframe → bar-close delay mapping (plan.md §3.5.3 availability_delay_ms)
# ---------------------------------------------------------------------------

_TF_REGEX = re.compile(r"^(\d+)\s*(s|m|h|d|w)$", re.IGNORECASE)

_UNIT_MS = {
    "s": 1_000,
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
}


def estimate_bar_close_delay_ms(timeframe: Optional[str]) -> int:
    """Estimate the minimum delay (ms) for a bar to become available.

    For OHLCV data the bar is only complete after its close timestamp,
    so the *minimum* availability delay equals the bar duration itself.
    For example a 1h bar has at least 3_600_000 ms delay.

    Returns 0 if timeframe is None or cannot be parsed.
    """
    if not timeframe:
        return 0
    m = _TF_REGEX.match(str(timeframe).strip())
    if not m:
        return 0
    count = int(m.group(1))
    unit = m.group(2).lower()
    return count * _UNIT_MS.get(unit, 0)


def check_availability_delay(
    expr: ExprNode,
    *,
    min_delay_ms: int,
    var_types: Optional[Dict[str, FactorType]] = None,
    timeframe: Optional[str] = None,
) -> CheckResult:
    inferred = infer_expr_type(expr, var_types=var_types)
    required = int(getattr(inferred, "availability_delay_ms", 0) or 0)

    # If a timeframe is provided, the bar-close delay is the lower bound
    bar_delay = estimate_bar_close_delay_ms(timeframe)
    effective_delay = max(required, bar_delay)

    allowed = int(min_delay_ms or 0)
    if effective_delay > allowed:
        return fail(
            "availability_delay",
            code="AVAILABILITY_DELAY_EXCEEDED",
            message=(
                f"effective availability_delay_ms={effective_delay} "
                f"(expr={required}, bar_close={bar_delay}) "
                f"exceeds min_delay_ms={allowed}"
            ),
            details={
                "availability_delay_ms": required,
                "bar_close_delay_ms": bar_delay,
                "effective_delay_ms": effective_delay,
                "min_delay_ms": allowed,
            },
        )
    return ok(
        "availability_delay",
        message="ok",
        details={
            "availability_delay_ms": required,
            "bar_close_delay_ms": bar_delay,
            "effective_delay_ms": effective_delay,
            "min_delay_ms": allowed,
        },
    )


def check_time_safety(
    expr: ExprNode,
    *,
    min_delay_ms: int = 0,
    var_types: Optional[Dict[str, FactorType]] = None,
    timeframe: Optional[str] = None,
) -> List[CheckResult]:
    """
    Time-safety checks (plan.md §3.5.3).

    1. Structural: no negative shifts (lookahead prevention).
    2. Availability: bar-close delay + expr delay <= min_delay_ms.
    """

    results: List[CheckResult] = [
        check_no_negative_shift(expr),
        check_availability_delay(
            expr,
            min_delay_ms=min_delay_ms,
            var_types=var_types,
            timeframe=timeframe,
        ),
    ]
    if all(r.ok for r in results):
        results.append(ok("time_safety", message="ok"))
    return results


__all__ = [
    "check_availability_delay",
    "check_time_safety",
    "estimate_bar_close_delay_ms",
]
