from __future__ import annotations

from typing import Dict, List, Optional

from agent_market.factor_compiler.api_models import ExprNode
from agent_market.factor_compiler.checks.leakage import check_no_negative_shift
from agent_market.factor_compiler.checks.types import CheckResult, fail, ok
from agent_market.factor_compiler.dsl.types import FactorType, infer_expr_type


def check_availability_delay(
    expr: ExprNode,
    *,
    min_delay_ms: int,
    var_types: Optional[Dict[str, FactorType]] = None,
) -> CheckResult:
    inferred = infer_expr_type(expr, var_types=var_types)
    required = int(getattr(inferred, "availability_delay_ms", 0) or 0)
    allowed = int(min_delay_ms or 0)
    if required > allowed:
        return fail(
            "availability_delay",
            code="AVAILABILITY_DELAY_EXCEEDED",
            message=f"availability_delay_ms={required} exceeds min_delay_ms={allowed}",
            details={"availability_delay_ms": required, "min_delay_ms": allowed},
        )
    return ok(
        "availability_delay",
        message="ok",
        details={"availability_delay_ms": required, "min_delay_ms": allowed},
    )


def check_time_safety(
    expr: ExprNode,
    *,
    min_delay_ms: int = 0,
    var_types: Optional[Dict[str, FactorType]] = None,
) -> List[CheckResult]:
    """
    Minimal time-safety checks.

    Today we treat negative shifts as the primary structural lookahead risk.
    We also enforce a best-effort `availability_delay_ms` gate via FactorType inference.
    Full delay modeling remains out of scope without real execution/market-data pipelines.
    """

    results: List[CheckResult] = [
        check_no_negative_shift(expr),
        check_availability_delay(expr, min_delay_ms=min_delay_ms, var_types=var_types),
    ]
    if all(r.ok for r in results):
        results.append(ok("time_safety", message="ok"))
    return results


__all__ = ["check_availability_delay", "check_time_safety"]
