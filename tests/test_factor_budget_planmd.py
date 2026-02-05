from __future__ import annotations


def test_complexity_budget_enforces_max_expensive_ops_and_budget_estimators() -> None:
    from agent_market.factor_compiler.api_models import ComplexityBudget, ExprNode
    from agent_market.factor_compiler.checks import (
        check_complexity,
        estimate_compute_budget,
        estimate_turnover_budget,
        expr_complexity_stats,
    )

    x = ExprNode(op="var", args=["close"])
    expr = ExprNode(
        op="add",
        args=[
            ExprNode(op="roll_mean", args=[x, 5]),
            ExprNode(op="roll_std", args=[x, 10]),
        ],
    )

    stats = expr_complexity_stats(expr)
    assert "expensive_ops" in stats
    assert int(stats["expensive_ops"]) >= 2

    budget = ComplexityBudget(max_nodes=100, max_depth=20, max_expensive_ops=1)
    res = check_complexity(expr, budget)
    assert res.ok is False
    assert res.code == "COMPLEXITY_MAX_EXPENSIVE_OPS"

    compute = estimate_compute_budget(expr)
    assert compute["max_window"] == 10
    assert compute["sum_windows"] >= 15
    assert compute["estimated_units"] >= compute["nodes"]

    turn = estimate_turnover_budget(ExprNode(op="diff", args=[x, 1]))
    assert float(turn["turnover_proxy"]) > 1.0

