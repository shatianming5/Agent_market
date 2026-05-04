"""Tests for the recursive-descent FASTEXPR validator."""
from __future__ import annotations

import pytest

from agent_market.wq_brain.expr_parser import (
    MAX_EXPRESSION_LENGTH,
    MAX_NESTING_DEPTH,
    ParseError,
    count_nesting,
    parse,
    tokenize,
    validate_expression_strict,
)


# ── Tokenizer ────────────────────────────────────────────────────────────

def test_tokenize_basic():
    toks = tokenize("rank(close)")
    kinds = [t.kind for t in toks]
    assert kinds == ["IDENT", "OP", "IDENT", "OP", "EOF"]


def test_tokenize_numbers():
    toks = tokenize("ts_mean(close, 20)")
    nums = [t.value for t in toks if t.kind == "NUMBER"]
    assert nums == ["20"]


def test_tokenize_handles_floats():
    toks = tokenize("hump(rank(close), 0.01)")
    nums = [t.value for t in toks if t.kind == "NUMBER"]
    assert "0.01" in nums


def test_tokenize_rejects_unknown_char():
    with pytest.raises(ParseError):
        tokenize("rank@close")


# ── Parser ───────────────────────────────────────────────────────────────

def test_parse_simple_call():
    ast = parse("rank(close)")
    assert ast.kind == "program"
    assert len(ast.args) == 1
    stmt = ast.args[0]
    assert stmt.kind == "call"
    assert stmt.value == "rank"


def test_parse_nested_calls():
    ast = parse("rank(ts_mean(close, 20))")
    inner = ast.args[0].args[0]
    assert inner.kind == "call"
    assert inner.value == "ts_mean"
    assert len(inner.args) == 2


def test_parse_arithmetic():
    ast = parse("rank(close - vwap)")
    rank_call = ast.args[0]
    binop = rank_call.args[0]
    assert binop.kind == "binop"
    assert binop.value == "-"


def test_parse_multi_statement_binding():
    ast = parse("x = ts_mean(close, 20); rank(close - x)")
    assert len(ast.args) == 2
    assert ast.args[0].kind == "assign"
    assert ast.args[0].value == "x"
    assert ast.args[1].kind == "call"
    assert ast.args[1].value == "rank"


def test_parse_unary_minus():
    ast = parse("rank(-ts_corr(close, volume, 20))")
    rank_call = ast.args[0]
    unary = rank_call.args[0]
    assert unary.kind == "unaryop"
    assert unary.value == "-"


def test_parse_comparison_and_if_else():
    ast = parse("rank(if_else(volume > adv20, close, open))")
    rank_call = ast.args[0]
    if_else = rank_call.args[0]
    assert if_else.kind == "call"
    assert if_else.value == "if_else"
    cond = if_else.args[0]
    assert cond.kind == "binop"
    assert cond.value == ">"


# ── Validator: passing cases ─────────────────────────────────────────────

@pytest.mark.parametrize("expr", [
    "rank(close)",
    "rank(close / ts_mean(close, 20) - 1)",
    "rank(ts_corr(close, volume, 20))",
    "rank(group_zscore(returns, sector))",
    "rank(ts_decay_linear(returns, 10))",
    "rank((high - low) / close)",
    "hump(rank(close), 0.01)",
    "hump(rank(ts_rank(close, 252) * (-ts_delta(close, 3) / close)), 0.01)",
    "rank(volume / adv60)",
    "rank(returns * volume / adv120)",
    "rank(ts_mean(volume / adv180, 20))",
    "x = ts_mean(close, 20); rank(close - x)",
    "rank(if_else(volume > adv20, close, open))",
    "rank(-ts_corr(close, volume, 20))",
])
def test_strict_validator_accepts_known_good(expr):
    errors = validate_expression_strict(expr)
    assert errors == [], f"unexpected errors for {expr!r}: {errors}"


# ── Validator: rejecting cases ───────────────────────────────────────────

def test_rejects_empty():
    assert "empty" in str(validate_expression_strict(""))
    assert "empty" in str(validate_expression_strict("   "))


def test_rejects_overlong_expression():
    long_expr = "rank(" + " + ".join(["close"] * 100) + ")"
    errors = validate_expression_strict(long_expr)
    assert any("too long" in e for e in errors)


def test_rejects_python_keywords():
    for kw_expr in [
        "lambda x: x",
        "import os",
        "def foo(): pass",
        "class A: pass",
        "print(close)",
        "exec('x')",
        "eval('y')",
        "__import__('os')",
    ]:
        errors = validate_expression_strict(kw_expr)
        assert any("forbidden Python syntax" in e for e in errors), kw_expr


def test_rejects_unbalanced_parens():
    assert any("unbalanced" in e for e in validate_expression_strict("rank(close"))
    assert any("unbalanced" in e for e in validate_expression_strict("rank(close))"))


def test_rejects_unavailable_ts_operator():
    errors = validate_expression_strict("rank(ts_std(close, 20))")
    assert any("ts_std" in e and "unavailable" in e for e in errors)


def test_rejects_unavailable_fundamental_field():
    errors = validate_expression_strict("rank(book_to_price)")
    assert any("book_to_price" in e and "unavailable" in e for e in errors)


def test_rejects_unknown_operator():
    errors = validate_expression_strict("rank(bogus_op(close, 20))")
    assert any("unknown operator: bogus_op" in e for e in errors)


def test_rejects_arity_too_few():
    errors = validate_expression_strict("ts_corr(close, volume)")
    assert any("ts_corr requires at least 3" in e for e in errors)


def test_rejects_arity_too_many():
    errors = validate_expression_strict("ts_mean(close, 20, 0.95)")
    assert any("ts_mean accepts at most 2" in e for e in errors)


def test_rejects_unknown_identifier():
    # `momentum_x` is neither a field nor a defined binding
    errors = validate_expression_strict("rank(momentum_x)")
    assert any("unknown identifier: momentum_x" in e for e in errors)


def test_accepts_locally_defined_binding():
    # x is defined in the first statement, used in the second
    errors = validate_expression_strict("x = rank(close); x")
    assert errors == []


def test_rejects_deep_nesting():
    # Build expression with > MAX_NESTING_DEPTH parens
    depth = MAX_NESTING_DEPTH + 3
    expr = "rank(" * depth + "close" + ")" * depth
    errors = validate_expression_strict(expr)
    assert any("nesting depth" in e for e in errors)


# ── Helpers ──────────────────────────────────────────────────────────────

def test_count_nesting():
    assert count_nesting("close") == 0
    assert count_nesting("rank(close)") == 1
    assert count_nesting("rank(ts_mean(close, 20))") == 2
    assert count_nesting("rank(ts_corr(close, ts_mean(volume, 5), 20))") == 3
