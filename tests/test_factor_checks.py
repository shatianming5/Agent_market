from __future__ import annotations

import pandas as pd


def test_check_complexity_budget_fails_on_depth():
    from agent_market.factor_compiler.api_models import ComplexityBudget, ExprNode
    from agent_market.factor_compiler.checks import check_complexity

    expr = ExprNode(op="add", args=[ExprNode(op="add", args=[ExprNode(op="var", args=["x"]), 1]), 2])
    res = check_complexity(expr, ComplexityBudget(max_nodes=100, max_depth=2))
    assert res.ok is False
    assert res.code == "COMPLEXITY_MAX_DEPTH"


def test_check_no_negative_shift_flags():
    from agent_market.factor_compiler.api_models import ExprNode
    from agent_market.factor_compiler.checks import check_no_negative_shift

    expr = ExprNode(op="shift", args=[ExprNode(op="var", args=["close"]), -1])
    res = check_no_negative_shift(expr)
    assert res.ok is False
    assert res.code == "LEAKAGE_NEGATIVE_SHIFT"


def test_check_permutation_leakage_flags_perfect_corr():
    from agent_market.factor_compiler.checks import check_permutation_leakage

    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    res = check_permutation_leakage(x, y, n_perm=30, seed=7)
    assert res.ok is False
    assert res.code == "LEAKAGE_PERMUTATION_SUSPECT"

