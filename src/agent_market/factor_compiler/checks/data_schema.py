from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from agent_market.factor_compiler.api_models import ExprArg, ExprNode
from agent_market.factor_compiler.checks.types import CheckResult, fail, ok
from agent_market.factor_compiler.dsl.grammar import CONST_OP, VAR_OP


def collect_var_names(expr: ExprNode) -> Set[str]:
    names: set[str] = set()
    stack: list[ExprArg] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, ExprNode):
            if str(node.op) == VAR_OP and node.args and isinstance(node.args[0], str):
                names.add(str(node.args[0]))
            for child in node.args or []:
                stack.append(child)
    return names


def _literal_int(arg: ExprArg) -> Optional[int]:
    try:
        if isinstance(arg, bool) or arg is None:
            return None
        if isinstance(arg, int):
            return int(arg)
        if isinstance(arg, float) and float(arg).is_integer():
            return int(arg)
    except Exception:
        return None
    return None


def check_variables_exist(expr: ExprNode, *, available_columns: Iterable[str]) -> CheckResult:
    available = {str(c) for c in available_columns if str(c)}
    used = collect_var_names(expr)
    missing = sorted(used - available)
    if missing:
        return fail(
            "data_schema_vars",
            code="DATA_SCHEMA_MISSING_VARS",
            message=f"Missing variables in data schema: {missing}",
            details={"missing": missing, "used": sorted(used)},
        )
    return ok("data_schema_vars", message="ok", details={"used": sorted(used)})


def check_literal_param_ranges(expr: ExprNode) -> List[CheckResult]:
    """
    Best-effort parameter range checks for common window/levels arguments.
    """

    results: list[CheckResult] = []

    stack: list[ExprArg] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, ExprNode):
            op = str(node.op)
            args = list(node.args or [])
            if op in {"roll_mean", "roll_std", "rolling_sum", "rolling_min", "rolling_max", "ema", "ts_z"}:
                if len(args) >= 2:
                    w = _literal_int(args[1])
                    if w is not None and w <= 0:
                        results.append(
                            fail(
                                "data_schema_params",
                                code="INVALID_WINDOW",
                                message=f"Invalid window for {op}: {w} (must be > 0)",
                                details={"op": op, "window": int(w)},
                            )
                        )
            if op in {"depth_bid", "depth_ask", "imbalance"}:
                # depth_bid(levels)
                if args:
                    levels = _literal_int(args[0])
                    if levels is not None and levels <= 0:
                        results.append(
                            fail(
                                "data_schema_params",
                                code="INVALID_LEVELS",
                                message=f"Invalid levels for {op}: {levels} (must be > 0)",
                                details={"op": op, "levels": int(levels)},
                            )
                        )
            for child in args:
                stack.append(child)

    if not results:
        results.append(ok("data_schema_params", message="ok"))
    return results


def check_data_schema(expr: ExprNode, *, available_columns: Iterable[str]) -> List[CheckResult]:
    results = [check_variables_exist(expr, available_columns=available_columns)]
    results.extend(check_literal_param_ranges(expr))
    return results


__all__ = [
    "check_data_schema",
    "check_literal_param_ranges",
    "check_variables_exist",
    "collect_var_names",
]

