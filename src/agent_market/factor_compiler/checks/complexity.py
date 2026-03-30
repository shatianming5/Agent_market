from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..api_models import ComplexityBudget, ExprArg, ExprNode
from ..dsl.ast_utils import literal_int
from .types import CheckResult, fail, ok


_EXPENSIVE_OPS: set[str] = {
    # windows
    "roll_mean",
    "roll_std",
    "rolling_mean",
    "rolling_std",
    "rolling_sum",
    "rolling_min",
    "rolling_max",
    "ema",
    "ts_z",
    "robust_z",
    "decay_linear",
    # other heavier transforms
    "winsorize",
}


def _stats(arg: ExprArg) -> Tuple[int, int, int]:
    if isinstance(arg, ExprNode):
        children = list(arg.args or [])
        if not children:
            expensive = 1 if str(arg.op) in _EXPENSIVE_OPS else 0
            return 1, 1, int(expensive)
        child_stats = [_stats(child) for child in children]
        nodes = 1 + sum(n for n, _, _ in child_stats)
        depth = 1 + max(d for _, d, _ in child_stats)
        expensive = (1 if str(arg.op) in _EXPENSIVE_OPS else 0) + sum(e for _, _, e in child_stats)
        return int(nodes), int(depth), int(expensive)
    # primitive leaf counts as 1 node
    return 1, 1, 0


def expr_complexity_stats(expr: ExprNode) -> Dict[str, Any]:
    nodes, depth, expensive_ops = _stats(expr)
    return {"nodes": int(nodes), "depth": int(depth), "expensive_ops": int(expensive_ops)}


def estimate_compute_budget(expr: ExprNode) -> Dict[str, Any]:
    """
    Best-effort compute budget proxy (plan.md 3.5.5).

    This is a static estimate based on:
      - node count / depth
      - expensive op count
      - literal windows / lookbacks found in the AST
    """

    stats = expr_complexity_stats(expr)
    windows: list[int] = []

    stack: list[ExprArg] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, ExprNode):
            op = str(node.op)
            args = list(node.args or [])

            # windowed calls (x, w)
            if op in {"roll_mean", "roll_std", "rolling_sum", "rolling_min", "rolling_max", "ema", "ts_z", "robust_z", "decay_linear"}:
                if len(args) >= 2:
                    w = literal_int(args[1])
                    if w is not None and w > 0:
                        windows.append(int(w))

            # differencing / percent change (x, n)
            if op in {"diff", "pct_change", "shift"}:
                if len(args) >= 2:
                    n = literal_int(args[1])
                    if n is not None and n > 0:
                        windows.append(int(n))

            # microstructure rolling proxies (w)
            if op in {"vwap", "ofi", "arrival_intensity", "impact_proxy"}:
                if args:
                    w = literal_int(args[0])
                    if w is not None and w > 0:
                        windows.append(int(w))

            for child in args:
                stack.append(child)

    max_window = int(max(windows)) if windows else 0
    sum_windows = int(sum(windows)) if windows else 0
    estimated_units = int(stats["nodes"]) + (10 * int(stats["expensive_ops"])) + sum_windows
    return {**stats, "max_window": max_window, "sum_windows": sum_windows, "estimated_units": estimated_units}


def estimate_turnover_budget(expr: ExprNode) -> Dict[str, Any]:
    """
    Best-effort turnover budget proxy (plan.md 3.5.5).

    This is a static heuristic based on operator presence.
    """

    counts: Dict[str, int] = {"diff": 0, "pct_change": 0, "sign": 0, "rank_xs": 0, "zscore_xs": 0}
    stack: list[ExprArg] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, ExprNode):
            op = str(node.op)
            if op in counts:
                counts[op] += 1
            for child in node.args or []:
                stack.append(child)

    proxy = 1.0
    proxy += 0.5 * float(counts["diff"])
    proxy += 0.5 * float(counts["pct_change"])
    proxy += 0.3 * float(counts["sign"])
    proxy += 0.2 * float(counts["rank_xs"] + counts["zscore_xs"])
    return {"turnover_proxy": float(proxy), "op_counts": counts}


def check_complexity(expr: ExprNode, budget: ComplexityBudget) -> CheckResult:
    stats = expr_complexity_stats(expr)
    nodes = int(stats["nodes"])
    depth = int(stats["depth"])
    expensive_ops = int(stats["expensive_ops"])

    if nodes > int(budget.max_nodes):
        return fail(
            "complexity_budget",
            code="COMPLEXITY_MAX_NODES",
            message=f"Expression too large: nodes={nodes} > max_nodes={budget.max_nodes}",
            details={**stats, "max_nodes": int(budget.max_nodes), "max_depth": int(budget.max_depth)},
        )
    if depth > int(budget.max_depth):
        return fail(
            "complexity_budget",
            code="COMPLEXITY_MAX_DEPTH",
            message=f"Expression too deep: depth={depth} > max_depth={budget.max_depth}",
            details={**stats, "max_nodes": int(budget.max_nodes), "max_depth": int(budget.max_depth)},
        )
    if expensive_ops > int(budget.max_expensive_ops):
        return fail(
            "complexity_budget",
            code="COMPLEXITY_MAX_EXPENSIVE_OPS",
            message=f"Too many expensive ops: expensive_ops={expensive_ops} > max_expensive_ops={budget.max_expensive_ops}",
            details={
                **stats,
                "max_nodes": int(budget.max_nodes),
                "max_depth": int(budget.max_depth),
                "max_expensive_ops": int(budget.max_expensive_ops),
            },
        )

    return ok(
        "complexity_budget",
        message="within_budget",
        details={
            **stats,
            "max_nodes": int(budget.max_nodes),
            "max_depth": int(budget.max_depth),
            "max_expensive_ops": int(budget.max_expensive_ops),
        },
    )
