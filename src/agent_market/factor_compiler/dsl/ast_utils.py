"""Shared AST helpers for the factor compiler DSL."""
from __future__ import annotations

from typing import Any, Optional

from ..api_models import ExprArg, ExprNode
from .errors import FormulaSerializeError


def literal_int(arg: ExprArg) -> Optional[int]:
    """Extract a literal integer from *arg*, recursing into ``pos``/``neg``/``const`` nodes."""
    if isinstance(arg, bool) or arg is None:
        return None
    if isinstance(arg, int):
        return int(arg)
    if isinstance(arg, float) and float(arg).is_integer():
        return int(arg)
    if isinstance(arg, ExprNode) and str(arg.op) in {"pos", "neg", "const"}:
        if not arg.args:
            return None
        inner = literal_int(arg.args[0])
        if inner is None:
            return None
        if str(arg.op) == "neg":
            return -inner
        return inner
    return None


def format_number(value: Any) -> str:
    """Format a numeric constant for formula serialisation."""
    if isinstance(value, bool) or value is None:
        raise FormulaSerializeError("Only numeric constants are supported in Formula")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    raise FormulaSerializeError(f"Unsupported constant type: {type(value).__name__}")
