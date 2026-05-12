"""Tests for the SymPy-backed WQ BRAIN math CLI helper."""
from __future__ import annotations

import pytest

from agent_market.wq_brain.symbolic_math import symbolic_math
from agent_market.wq_brain.llm_validator import validate_cli_invocation


def test_symbolic_math_simplifies_rational_expression():
    out = symbolic_math("simplify", "(x^2 - 1) / (x - 1)")
    assert out["ok"] is True
    assert out["result"] == "x + 1"
    assert out["free_symbols"] == ["x"]


def test_symbolic_math_derivative_uses_named_variable():
    out = symbolic_math("diff", "log(S)", var="S")
    assert out["result"] == "1/S"
    assert out["variable"] == "S"


def test_symbolic_math_solves_linear_equation():
    out = symbolic_math("solve", "a*x + b = 0", solve_for="x")
    assert out["result"] == ["-b/a"]


def test_symbolic_math_rejects_unsafe_tokens():
    with pytest.raises(ValueError, match="unsafe token"):
        symbolic_math("simplify", "__import__('os').system('echo bad')")


def test_llm_validator_accepts_new_research_cli_commands():
    assert validate_cli_invocation("python scripts/wq_brain.py search-papers 'alpha factor'").ok
    assert validate_cli_invocation("python scripts/wq_brain.py math diff 'log(S)' --var S").ok
