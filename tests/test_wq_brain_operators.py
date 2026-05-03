"""Tests for wq_brain.operators — expression validation."""
from __future__ import annotations

import pytest

from agent_market.wq_brain.operators import validate_expression, operators_prompt_block


class TestValidateExpression:
    def test_empty_expression(self):
        errors = validate_expression("")
        assert errors

    def test_valid_simple(self):
        errors = validate_expression("rank(close / ts_mean(close, 20))")
        assert not any("Forbidden" in e for e in errors)

    def test_forbidden_import(self):
        errors = validate_expression("import os; rank(close)")
        assert any("import" in e.lower() for e in errors)

    def test_forbidden_def(self):
        errors = validate_expression("def f(): return rank(close)")
        assert any("def" in e.lower() for e in errors)

    def test_unknown_identifiers_warn(self):
        errors = validate_expression("rank(nonexistent_field_xyz)")
        assert any("Unrecognised" in e for e in errors)

    def test_known_fields_pass(self):
        errors = validate_expression("rank(close / book_to_price)")
        unknown_errors = [e for e in errors if "Unrecognised" in e]
        assert not unknown_errors

    def test_nested_ts_operators(self):
        expr = "rank(ts_mean(returns, 20) / ts_std(returns, 20))"
        errors = validate_expression(expr)
        assert not any("Forbidden" in e for e in errors)

    def test_group_rank(self):
        errors = validate_expression("group_rank(close, sector)")
        assert not any("Forbidden" in e for e in errors)


class TestOperatorsPromptBlock:
    def test_contains_key_sections(self):
        block = operators_prompt_block()
        assert "Time-series" in block
        assert "Cross-section" in block
        assert "rank" in block
        assert "ts_mean" in block
        assert "FASTEXPR" in block
