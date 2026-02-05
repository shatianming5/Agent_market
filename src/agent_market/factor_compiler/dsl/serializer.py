from __future__ import annotations

from typing import Any

from ..api_models import ExprArg, ExprNode
from .errors import FormulaSerializeError
from .grammar import (
    CONST_OP,
    OP_TO_BINOP_SYMBOL,
    OP_TO_CMP_SYMBOL,
    OP_TO_UNARY_SYMBOL,
    VAR_OP,
    is_safe_identifier,
)


def _format_number(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        raise FormulaSerializeError("Only numeric constants are supported in Formula")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    raise FormulaSerializeError(f"Unsupported constant type: {type(value).__name__}")


def _render(arg: ExprArg) -> str:
    if isinstance(arg, ExprNode):
        op = str(arg.op)
        args = list(arg.args or [])

        if op == VAR_OP:
            if len(args) != 1 or not isinstance(args[0], str) or not is_safe_identifier(args[0]):
                raise FormulaSerializeError(f"Invalid var node: {arg!r}")
            return args[0]

        if op == CONST_OP:
            if len(args) != 1:
                raise FormulaSerializeError(f"Invalid const node: {arg!r}")
            return _format_number(args[0])

        if op in OP_TO_BINOP_SYMBOL:
            if len(args) != 2:
                raise FormulaSerializeError(f"Invalid binary node: {arg!r}")
            left = _render(args[0])
            right = _render(args[1])
            return f"({left}{OP_TO_BINOP_SYMBOL[op]}{right})"

        if op in OP_TO_UNARY_SYMBOL:
            if len(args) != 1:
                raise FormulaSerializeError(f"Invalid unary node: {arg!r}")
            inner = _render(args[0])
            return f"({OP_TO_UNARY_SYMBOL[op]}{inner})"

        if op in OP_TO_CMP_SYMBOL:
            if len(args) != 2:
                raise FormulaSerializeError(f"Invalid compare node: {arg!r}")
            left = _render(args[0])
            right = _render(args[1])
            return f"({left}{OP_TO_CMP_SYMBOL[op]}{right})"

        if not is_safe_identifier(op):
            raise FormulaSerializeError(f"Invalid call name: {op!r}")
        inner = ",".join(_render(a) for a in args)
        return f"{op}({inner})"

    return _format_number(arg)


def to_formula(expr: ExprNode) -> str:
    return _render(expr)

