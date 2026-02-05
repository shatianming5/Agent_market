from __future__ import annotations

import ast
from typing import Optional

from ..api_models import ExprArg, ExprNode
from .errors import FormulaParseError
from .grammar import (
    BINOP_TO_OP,
    CMPOP_TO_OP,
    MAX_AST_NODES,
    MAX_FORMULA_LENGTH,
    UNARYOP_TO_OP,
    CONST_OP,
    VAR_OP,
    is_safe_identifier,
)


class _Converter(ast.NodeVisitor):
    def __init__(self, *, allowed_calls: Optional[set[str]]) -> None:
        self.allowed_calls = allowed_calls

    def generic_visit(self, node: ast.AST) -> ExprArg:  # noqa: D401
        """Reject any syntax we don't explicitly support."""
        raise FormulaParseError(f"Unsupported syntax: {type(node).__name__}")

    def visit_Name(self, node: ast.Name) -> ExprArg:  # noqa: N802
        if not is_safe_identifier(node.id):
            raise FormulaParseError(f"Invalid identifier: {node.id!r}")
        return ExprNode(op=VAR_OP, args=[node.id])

    def visit_Constant(self, node: ast.Constant) -> ExprArg:  # noqa: N802
        value = node.value
        if isinstance(value, bool) or value is None:
            raise FormulaParseError("Only numeric constants are allowed")
        if isinstance(value, (int, float)):
            return value
        raise FormulaParseError("Only numeric constants are allowed")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ExprArg:  # noqa: N802
        op = UNARYOP_TO_OP.get(type(node.op))
        if not op:
            raise FormulaParseError(f"Unary operator not supported: {type(node.op).__name__}")
        operand = self.visit(node.operand)
        return ExprNode(op=op, args=[operand])

    def visit_BinOp(self, node: ast.BinOp) -> ExprArg:  # noqa: N802
        op = BINOP_TO_OP.get(type(node.op))
        if not op:
            raise FormulaParseError(f"Binary operator not supported: {type(node.op).__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return ExprNode(op=op, args=[left, right])

    def visit_Call(self, node: ast.Call) -> ExprArg:  # noqa: N802
        if not isinstance(node.func, ast.Name):
            raise FormulaParseError("Only simple function calls are allowed")
        name = node.func.id
        if not is_safe_identifier(name):
            raise FormulaParseError(f"Invalid call name: {name!r}")
        if self.allowed_calls is not None and name not in self.allowed_calls:
            raise FormulaParseError(f"Function call not allowed: {name}")
        if node.keywords:
            raise FormulaParseError("Keyword arguments are not allowed")
        args = [self.visit(arg) for arg in node.args]
        return ExprNode(op=name, args=args)

    def visit_Compare(self, node: ast.Compare) -> ExprArg:  # noqa: N802
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise FormulaParseError("Chained comparisons are not supported")
        op = CMPOP_TO_OP.get(type(node.ops[0]))
        if not op:
            raise FormulaParseError(f"Comparison not supported: {type(node.ops[0]).__name__}")
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        return ExprNode(op=op, args=[left, right])


def parse_formula(formula: str, *, allowed_calls: Optional[set[str]] = None) -> ExprNode:
    if not isinstance(formula, str) or not formula.strip():
        raise FormulaParseError("Formula must be a non-empty string")
    if len(formula) > MAX_FORMULA_LENGTH:
        raise FormulaParseError(f"Formula too long (>{MAX_FORMULA_LENGTH} chars)")
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise FormulaParseError(f"Invalid formula syntax: {exc}") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise FormulaParseError(f"Formula too complex (>{MAX_AST_NODES} AST nodes)")

    value: ExprArg = _Converter(allowed_calls=allowed_calls).visit(tree.body)
    if isinstance(value, ExprNode):
        return value
    return ExprNode(op=CONST_OP, args=[value])

