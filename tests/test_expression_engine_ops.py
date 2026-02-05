from __future__ import annotations

import pandas as pd
import pytest


def test_expression_engine_rejects_negative_shift():
    from agent_market.freqai.expression_engine import ExpressionValidationError, safe_eval_expression

    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    with pytest.raises(ExpressionValidationError):
        safe_eval_expression("shift(close,-1)", df)


def test_expression_engine_new_ops_work():
    from agent_market.freqai.expression_engine import safe_eval_expression

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    s = safe_eval_expression("rolling_sum(x,2)", df)
    assert s.isna().sum() >= 1
    assert float(s.iloc[-1]) == 5.0

    s2 = safe_eval_expression("ifelse(x>2,1,0)", df)
    assert s2.tolist() == [0.0, 0.0, 1.0]


def test_factor_dsl_compile_to_expression_engine_executes():
    from agent_market.factor_compiler.dsl import parse_formula
    from agent_market.factor_compiler.dsl.operators import compile_to_expression_engine
    from agent_market.freqai.expression_engine import safe_eval_expression

    df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]})
    node = parse_formula("zscore(rolling_mean(pct_change(close,1),2),2)")
    expr = compile_to_expression_engine(node)
    assert "ts_z" in expr
    assert "roll_mean" in expr

    out = safe_eval_expression(expr, df)
    assert len(out) == len(df)
    assert out.dropna().size > 0

