from __future__ import annotations

"""
Canonical AST shim for plan.md alignment.

The project uses `agent_market.factor_compiler.api_models.ExprNode` as the canonical
AST node (Pydantic, stable JSON, sha256). `plan.md` proposes `dsl/ast.py` as the
home for AST definitions, so this module provides a stable import path while
keeping the canonical implementation in `api_models`.
"""

from agent_market.factor_compiler.api_models import ExprArg, ExprNode

__all__ = ["ExprArg", "ExprNode"]

