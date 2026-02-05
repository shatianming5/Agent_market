from __future__ import annotations

import json

import pytest


def test_factor_dsl_formula_ast_json_roundtrip():
    from agent_market.factor_compiler.api_models import ExprNode
    from agent_market.factor_compiler.dsl import parse_formula, to_formula

    formula = "roll_mean(pct_change(close,1),24)+ema(close,12)"
    node_a = parse_formula(formula)
    dumped = json.loads(node_a.canonical_json())

    node_b = ExprNode.model_validate(dumped)
    formula_b = to_formula(node_b)
    node_c = parse_formula(formula_b)

    assert node_a.canonical_sha256() == node_c.canonical_sha256()


@pytest.mark.parametrize(
    "bad",
    [
        "close.__class__",
        "foo(bar=1)",
        "__import__('os')",
        "(lambda x: x)(1)",
        "a[0]",
    ],
)
def test_factor_dsl_rejects_unsafe_syntax(bad: str):
    from agent_market.factor_compiler.dsl import FormulaParseError, parse_formula

    with pytest.raises(FormulaParseError):
        parse_formula(bad)

