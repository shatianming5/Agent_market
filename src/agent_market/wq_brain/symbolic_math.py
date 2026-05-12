"""Small SymPy wrapper for agent-facing symbolic derivations."""
from __future__ import annotations

import string
from typing import Any


DENIED_TOKENS = ("__", "import", "exec", "eval", "open(", "read(", "write(")


def _load_sympy():
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except Exception as exc:  # pragma: no cover - exercised only when dep missing
        raise RuntimeError("sympy is required for `wq_brain.py math`; install `sympy`.") from exc
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    return sp, parse_expr, transformations


def _reject_unsafe(expr: str) -> None:
    lowered = expr.lower()
    for token in DENIED_TOKENS:
        if token in lowered:
            raise ValueError(f"unsafe token rejected: {token}")


def _parse(expr: str):
    if not expr or not expr.strip():
        raise ValueError("empty expression")
    _reject_unsafe(expr)
    sp, parse_expr, transformations = _load_sympy()
    local_dict = {
        "Abs": sp.Abs,
        "Eq": sp.Eq,
        "Max": sp.Max,
        "Min": sp.Min,
        "acos": sp.acos,
        "asin": sp.asin,
        "atan": sp.atan,
        "cos": sp.cos,
        "diff": sp.diff,
        "exp": sp.exp,
        "integrate": sp.integrate,
        "log": sp.log,
        "pi": sp.pi,
        "sin": sp.sin,
        "sqrt": sp.sqrt,
        "tan": sp.tan,
    }
    # Finance notation commonly uses uppercase single-letter symbols such as
    # S, T, P, Q.  SymPy's default namespace gives some of these special
    # meanings, so pin them to Symbols for this CLI.
    local_dict.update({letter: sp.Symbol(letter) for letter in string.ascii_letters})
    return parse_expr(expr, local_dict=local_dict, transformations=transformations, evaluate=True)


def _symbol(name: str | None, expr: Any | None = None):
    sp, _parse_expr, _transformations = _load_sympy()
    if name:
        _reject_unsafe(name)
        return sp.Symbol(name)
    if expr is not None and getattr(expr, "free_symbols", None):
        return sorted(expr.free_symbols, key=lambda s: str(s))[0]
    return sp.Symbol("x")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def symbolic_math(
    operation: str,
    expr: str,
    *,
    var: str | None = None,
    solve_for: str | None = None,
    point: str = "0",
    order: int = 6,
) -> dict[str, Any]:
    """Run a bounded symbolic operation and return JSON-serializable output."""
    sp, _parse_expr, _transformations = _load_sympy()
    operation = operation.strip().lower().replace("-", "_")
    if operation == "solve" and "=" in expr and "==" not in expr:
        left, right = expr.split("=", 1)
        parsed = sp.Eq(_parse(left), _parse(right))
    else:
        parsed = _parse(expr)

    if operation == "simplify":
        result = sp.simplify(parsed)
    elif operation == "expand":
        result = sp.expand(parsed)
    elif operation == "factor":
        result = sp.factor(parsed)
    elif operation == "diff":
        result = sp.diff(parsed, _symbol(var, parsed))
    elif operation == "integrate":
        result = sp.integrate(parsed, _symbol(var, parsed))
    elif operation == "solve":
        target = _symbol(solve_for or var, parsed)
        result = sp.solve(parsed, target)
    elif operation == "series":
        result = sp.series(parsed, _symbol(var, parsed), _parse(point), max(1, int(order)))
    elif operation == "latex":
        result = parsed
    else:
        raise ValueError(
            "operation must be one of: simplify, expand, factor, diff, integrate, solve, series, latex"
        )

    return {
        "ok": True,
        "operation": operation,
        "input": expr,
        "parsed": str(parsed),
        "variable": str(_symbol(var, parsed)) if operation in {"diff", "integrate", "series"} else (solve_for or var or ""),
        "result": _jsonable(result),
        "latex": sp.latex(result),
        "free_symbols": [str(s) for s in sorted(parsed.free_symbols, key=lambda s: str(s))],
    }
