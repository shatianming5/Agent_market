from __future__ import annotations

from typing import Iterable, List, Optional, Set

from agent_market.factor_compiler.api_models import ExprArg, ExprNode
from agent_market.factor_compiler.checks.types import CheckResult, fail, ok
from agent_market.factor_compiler.dsl.ast_utils import literal_int
from agent_market.factor_compiler.dsl.grammar import (
    CONST_OP,
    OP_TO_BINOP_SYMBOL,
    OP_TO_CMP_SYMBOL,
    OP_TO_UNARY_SYMBOL,
    VAR_OP,
)


_ALLOWED_FACTOR_OPS: Set[str] = {
    # structural nodes
    VAR_OP,
    CONST_OP,
    # binary/unary/compare ops
    *OP_TO_BINOP_SYMBOL.keys(),
    *OP_TO_UNARY_SYMBOL.keys(),
    *OP_TO_CMP_SYMBOL.keys(),
    # core expression-engine function calls
    "z",
    "ts_z",
    "zscore",
    "shift",
    "lag",
    "diff",
    "roll_mean",
    "roll_std",
    "rolling_mean",
    "rolling_std",
    "rolling_sum",
    "rolling_min",
    "rolling_max",
    "ema",
    "pct_change",
    "sign",
    "clip",
    "decay_linear",
    "winsorize",
    "robust_z",
    "log1p",
    "log",
    "exp",
    "sqrt",
    "tanh",
    "abs",
    "ifelse",
    "rank_xs",
    "zscore_xs",
    "corr_xs",
    "neutralize",
    "fill_prob",
    "impact_proxy",
    "queue_pos_proxy",
    # microstructure column/call shims (compiled to existing columns)
    "mid",
    "spread",
    "microprice",
    "trade_sign",
    "depth_bid",
    "depth_ask",
    "imbalance",
    "vwap",
    "ofi",
    "arrival_intensity",
    "rv",
}


def collect_ops(expr: ExprNode) -> Set[str]:
    ops: set[str] = set()
    stack: list[ExprArg] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, ExprNode):
            ops.add(str(node.op))
            for child in node.args or []:
                stack.append(child)
    return ops


def check_operator_whitelist(expr: ExprNode, *, allowed_ops: Optional[Set[str]] = None) -> CheckResult:
    allowed = set(allowed_ops or _ALLOWED_FACTOR_OPS)
    used = collect_ops(expr)
    unknown = sorted(used - allowed)
    if unknown:
        return fail(
            "data_schema_ops",
            code="UNKNOWN_OPERATOR",
            message=f"Unknown operator(s) in FactorSpec AST: {unknown}",
            details={"unknown": unknown, "used": sorted(used)},
        )
    return ok("data_schema_ops", message="ok", details={"used": sorted(used)})


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
            if op in {"roll_mean", "roll_std", "rolling_mean", "rolling_std", "rolling_sum", "rolling_min", "rolling_max", "ema", "ts_z", "robust_z", "decay_linear"}:
                if len(args) >= 2:
                    w = literal_int(args[1])
                    if w is not None and w <= 0:
                        results.append(
                            fail(
                                "data_schema_params",
                                code="INVALID_WINDOW",
                                message=f"Invalid window for {op}: {w} (must be > 0)",
                                details={"op": op, "window": int(w)},
                            )
                        )
            if op == "winsorize":
                if len(args) >= 2:
                    try:
                        p = float(args[1]) if not isinstance(args[1], ExprNode) else None
                    except Exception:
                        p = None
                    if p is not None and not (0.0 < float(p) < 0.5):
                        results.append(
                            fail(
                                "data_schema_params",
                                code="INVALID_WINSORIZE_P",
                                message=f"Invalid p for winsorize: {p} (must be in (0, 0.5))",
                                details={"op": op, "p": float(p)},
                            )
                        )
            if op == "zscore":
                if len(args) == 2:
                    w = literal_int(args[1])
                    if w is not None and w <= 0:
                        results.append(
                            fail(
                                "data_schema_params",
                                code="INVALID_WINDOW",
                                message=f"Invalid window for {op}: {w} (must be > 0)",
                                details={"op": op, "window": int(w)},
                            )
                        )
            if op in {"diff", "pct_change"}:
                if len(args) >= 2:
                    n = literal_int(args[1])
                    if n is not None and n <= 0:
                        results.append(
                            fail(
                                "data_schema_params",
                                code="INVALID_N",
                                message=f"Invalid n for {op}: {n} (must be > 0)",
                                details={"op": op, "n": int(n)},
                            )
                        )
            if op in {"depth_bid", "depth_ask", "imbalance"}:
                # depth_bid(levels)
                if args:
                    levels = literal_int(args[0])
                    if levels is not None and levels <= 0:
                        results.append(
                            fail(
                                "data_schema_params",
                                code="INVALID_LEVELS",
                                message=f"Invalid levels for {op}: {levels} (must be > 0)",
                                details={"op": op, "levels": int(levels)},
                            )
                        )
            if op in {"vwap", "ofi", "arrival_intensity", "rv"}:
                if len(args) >= 1:
                    w = literal_int(args[0])
                    if w is not None and w <= 0:
                        results.append(
                            fail(
                                "data_schema_params",
                                code="INVALID_WINDOW",
                                message=f"Invalid window for {op}: {w} (must be > 0)",
                                details={"op": op, "window": int(w)},
                            )
                        )
            if op == "impact_proxy":
                if args:
                    w = literal_int(args[0])
                    if w is not None and w <= 0:
                        results.append(
                            fail(
                                "data_schema_params",
                                code="INVALID_WINDOW",
                                message=f"Invalid window for {op}: {w} (must be > 0)",
                                details={"op": op, "window": int(w)},
                            )
                        )
            if op == "fill_prob":
                if len(args) >= 2:
                    horizon = literal_int(args[1])
                    if horizon is not None and horizon <= 0:
                        results.append(
                            fail(
                                "data_schema_params",
                                code="INVALID_HORIZON",
                                message=f"Invalid horizon for {op}: {horizon} (must be > 0)",
                                details={"op": op, "horizon": int(horizon)},
                            )
                        )
            for child in args:
                stack.append(child)

    if not results:
        results.append(ok("data_schema_params", message="ok"))
    return results


def check_data_schema(expr: ExprNode, *, available_columns: Iterable[str]) -> List[CheckResult]:
    results = [
        check_operator_whitelist(expr),
        check_variables_exist(expr, available_columns=available_columns),
    ]
    results.extend(check_literal_param_ranges(expr))
    return results


__all__ = [
    "check_data_schema",
    "check_operator_whitelist",
    "check_literal_param_ranges",
    "check_variables_exist",
    "collect_ops",
    "collect_var_names",
]
