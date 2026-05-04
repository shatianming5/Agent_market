"""Operator whitelist + validate_expression tests."""
from __future__ import annotations

from agent_market.wq_brain.operators import (
    ALL_FIELDS,
    ALL_OPERATORS,
    FIELDS_FUNDAMENTAL_UNAVAILABLE,
    FIELDS_PRICE_VOLUME,
    OPERATORS_CS,
    OPERATORS_MATH,
    OPERATORS_TRANSFORM,
    OPERATORS_TS,
    OPERATORS_TS_UNAVAILABLE,
    operators_prompt_block,
    validate_expression,
)


def test_basic_valid_expressions_pass():
    for expr in [
        "rank(close)",
        "rank(close / ts_mean(close, 20) - 1)",
        "rank(ts_corr(close, volume, 20))",
        "rank(group_zscore(returns, sector))",
        "rank(ts_decay_linear(returns, 10))",
        "rank((high - low) / close)",
        # New: hump for turnover reduction (1-arg ONLY on WQ free tier)
        "hump(rank(close))",
        "hump(rank(ts_rank(close,252) * (-ts_delta(close,3)/close)))",
        # New: longer ADV variants
        "rank(volume / adv60)",
        "rank(returns * volume / adv120)",
        "rank(ts_mean(volume / adv180, 20))",
        # New: multi-line binding (semicolon-separated)
        "x = ts_mean(close, 20); rank(close - x)",
    ]:
        assert validate_expression(expr) == [], f"unexpected errors for {expr}"


def test_empty_expression_rejected():
    assert "empty expression" in str(validate_expression(""))
    assert "empty expression" in str(validate_expression("   "))


def test_python_syntax_rejected():
    for kw, expr in [
        ("import", "import os"),
        ("def ", "def foo(): pass"),
        ("class ", "class A: pass"),
        ("lambda", "lambda x: x"),
        ("print(", "print(close)"),
        ("eval(", "eval('x')"),
        ("__", "__import__('os')"),
    ]:
        errors = validate_expression(expr)
        assert any(kw in e for e in errors), f"{kw} not caught in {expr}: {errors}"


def test_unbalanced_parens_rejected():
    assert any("unbalanced" in e for e in validate_expression("rank(close"))
    assert any("unbalanced" in e for e in validate_expression("rank(close))"))


def test_unavailable_ts_operators_rejected():
    for op in OPERATORS_TS_UNAVAILABLE[:5]:
        errors = validate_expression(f"rank({op}(close, 20))")
        assert any(op in e for e in errors), f"{op} not caught"


def test_unavailable_fundamental_fields_rejected():
    for fld in FIELDS_FUNDAMENTAL_UNAVAILABLE[:5]:
        errors = validate_expression(f"rank({fld})")
        assert any(fld in e for e in errors), f"{fld} not caught"


def test_whitelist_disjoint_from_unavailable():
    available_ts = set(OPERATORS_TS)
    unavailable_ts = set(OPERATORS_TS_UNAVAILABLE)
    assert not (available_ts & unavailable_ts)
    available_fields = set(FIELDS_PRICE_VOLUME)
    unavailable_fields = set(FIELDS_FUNDAMENTAL_UNAVAILABLE)
    assert not (available_fields & unavailable_fields)


def test_operators_prompt_block_contains_all_categories():
    block = operators_prompt_block()
    for op in OPERATORS_TS:
        assert op in block, f"missing {op} in prompt block"
    for op in OPERATORS_CS:
        assert op in block
    for op in OPERATORS_MATH:
        assert op in block, f"missing {op} in prompt block"
    for op in OPERATORS_TRANSFORM:
        assert op in block, f"missing transform {op}"
    for op in OPERATORS_TS_UNAVAILABLE:
        assert op in block
    for fld in FIELDS_FUNDAMENTAL_UNAVAILABLE:
        assert fld in block


def test_operators_prompt_block_advertises_hump_with_examples():
    block = operators_prompt_block()
    assert "hump(" in block
    assert "Turnover Reduction" in block
    # Should show concrete hump_value examples
    assert "0.01" in block


def test_operators_prompt_block_advertises_adv_variants():
    block = operators_prompt_block()
    for adv in ("adv20", "adv60", "adv120", "adv180"):
        assert adv in block, f"missing {adv} in prompt block"


def test_operators_prompt_block_advertises_multiline_binding():
    block = operators_prompt_block()
    assert "Multi-Line Binding" in block
    assert ";" in block
    # The classic x = ...; y = ...; rank(...) shape
    assert "x =" in block or "x=" in block


def test_operators_prompt_block_no_truncation_marker():
    block = operators_prompt_block()
    # Regression for v1 bug: OPERATORS_MATH[:8] + ", ..." was truncating
    assert ", ..." not in block, "prompt block should not contain truncation marker"


def test_all_operators_count():
    assert len(ALL_OPERATORS) == (
        len(OPERATORS_TS) + len(OPERATORS_CS) + len(OPERATORS_MATH) + len(OPERATORS_TRANSFORM)
    )
    assert len(ALL_OPERATORS) >= 33


def test_adv_variants_in_price_volume_fields():
    for adv in ("adv20", "adv60", "adv120", "adv180"):
        assert adv in FIELDS_PRICE_VOLUME, f"{adv} missing"


def test_hump_in_transform():
    assert "hump" in OPERATORS_TRANSFORM
    assert "hump" in ALL_OPERATORS
