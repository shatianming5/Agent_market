from __future__ import annotations


def test_check_operator_whitelist_flags_unknown_ops() -> None:
    from agent_market.factor_compiler.api_models import ExprNode
    from agent_market.factor_compiler.checks.data_schema import check_operator_whitelist

    expr = ExprNode(op="unknown_op", args=[ExprNode(op="var", args=["close"])])
    res = check_operator_whitelist(expr)
    assert res.ok is False
    assert res.code == "UNKNOWN_OPERATOR"


def test_check_literal_param_ranges_flags_invalid_microstructure_windows() -> None:
    from agent_market.factor_compiler.api_models import ExprNode
    from agent_market.factor_compiler.checks.data_schema import check_literal_param_ranges

    expr = ExprNode(op="vwap", args=[0])
    results = check_literal_param_ranges(expr)
    assert any(r.ok is False for r in results)
