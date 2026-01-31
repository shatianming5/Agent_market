from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

__all__ = [
    "ExpressionSpec",
    "load_expression_file",
    "safe_eval",
    "safe_eval_expression",
    "apply_expressions",
]


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

_MAX_EXPRESSION_LENGTH = 512
_MAX_AST_NODES = 256


class ExpressionValidationError(ValueError):
    pass


_ALLOWED_BINOPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.FloorDiv,
)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)
_ALLOWED_CMPOPS = (ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq)


class _ExpressionValidator(ast.NodeVisitor):
    def __init__(self, *, allowed_names: set[str], allowed_calls: set[str]) -> None:
        self.allowed_names = allowed_names
        self.allowed_calls = allowed_calls

    def generic_visit(self, node: ast.AST) -> None:  # noqa: D401
        """Reject any syntax we don't explicitly allow."""
        raise ExpressionValidationError(f"Unsupported expression syntax: {type(node).__name__}")

    def visit_Expression(self, node: ast.Expression) -> None:  # noqa: N802
        self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if node.id not in self.allowed_names:
            raise ExpressionValidationError(f"Unknown name '{node.id}'")

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if not isinstance(node.value, (int, float)):
            raise ExpressionValidationError("Only numeric constants are allowed")

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise ExpressionValidationError(f"Operator not allowed: {type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:  # noqa: N802
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise ExpressionValidationError(f"Unary operator not allowed: {type(node.op).__name__}")
        self.visit(node.operand)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if not isinstance(node.func, ast.Name):
            raise ExpressionValidationError("Only simple function calls are allowed")
        func_name = node.func.id
        if func_name not in self.allowed_calls:
            raise ExpressionValidationError(f"Function call not allowed: {func_name}")
        if node.keywords:
            raise ExpressionValidationError("Keyword arguments are not allowed")
        for arg in node.args:
            self.visit(arg)

    def visit_Compare(self, node: ast.Compare) -> None:  # noqa: N802
        self.visit(node.left)
        for op in node.ops:
            if not isinstance(op, _ALLOWED_CMPOPS):
                raise ExpressionValidationError(f"Comparison operator not allowed: {type(op).__name__}")
        for comp in node.comparators:
            self.visit(comp)


def _validate_expression_ast(expr: str, *, allowed_names: set[str], allowed_calls: set[str]) -> None:
    if not isinstance(expr, str) or not expr.strip():
        raise ExpressionValidationError("Expression must be a non-empty string")
    if len(expr) > _MAX_EXPRESSION_LENGTH:
        raise ExpressionValidationError(f"Expression too long (>{_MAX_EXPRESSION_LENGTH} chars)")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExpressionValidationError(f"Invalid expression syntax: {exc}") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise ExpressionValidationError(f"Expression too complex (>{_MAX_AST_NODES} AST nodes)")
    _ExpressionValidator(allowed_names=allowed_names, allowed_calls=allowed_calls).visit(tree)


def safe_eval(expr: str, env: Dict[str, Any], *, allowed_calls: Optional[set[str]] = None) -> Any:
    if allowed_calls is None:
        allowed_calls = {k for k, v in env.items() if callable(v)}
    _validate_expression_ast(expr, allowed_names=set(env.keys()), allowed_calls=allowed_calls)
    return eval(expr, {"__builtins__": {}}, env)  # noqa: S307


@dataclass(frozen=True, slots=True)
class ExpressionSpec:
    name: str
    expression: str
    meta: Dict[str, Any]


def _z(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    return (series - mean) / (std + 1e-9)


def _ts_z(series: pd.Series, window: int) -> pd.Series:
    w = int(window)
    mean = series.rolling(w).mean()
    std = series.rolling(w).std(ddof=0)
    return (series - mean) / (std + 1e-9)


def _build_eval_env(df: DataFrame) -> Dict[str, Any]:
    env: Dict[str, Any] = {col: df[col] for col in df.columns}
    env.update(
        {
            "z": _z,
            "ts_z": _ts_z,
            "shift": lambda s, n: s.shift(int(n)),
            "roll_mean": lambda s, w: s.rolling(int(w)).mean(),
            "roll_std": lambda s, w: s.rolling(int(w)).std(ddof=0),
            "pct_change": lambda s, n: s.pct_change(periods=int(n)),
            "sign": lambda s: np.sign(s),
            "clip": lambda s, lo, hi: s.clip(float(lo), float(hi)),
            "ema": lambda s, span: s.ewm(span=int(span), adjust=False).mean(),
            "rolling_max": lambda s, w: s.rolling(int(w)).max(),
            "rolling_min": lambda s, w: s.rolling(int(w)).min(),
            "log1p": lambda s: np.log1p(s),
            "tanh": lambda s: np.tanh(s),
            "abs": abs,
        }
    )
    return env


def safe_eval_expression(expr: str, df: DataFrame) -> pd.Series:
    env = _build_eval_env(df)
    value = safe_eval(expr, env)
    if isinstance(value, pd.Series):
        series = value
    else:
        series = pd.Series(value, index=df.index)
    series = series.astype(float)
    return series.replace([np.inf, -np.inf], np.nan)


def load_expression_file(path: Path) -> List[ExpressionSpec]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    raw = payload.get("expressions")
    if not isinstance(raw, list):
        raise ValueError(f"Expression file missing 'expressions' list: {path}")

    out: List[ExpressionSpec] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        expr = str(item.get("expression") or item.get("formula") or "").strip()
        if not name or not expr:
            continue
        if not _NAME_RE.match(name):
            raise ValueError(f"Invalid expression name '{name}' in {path}")
        if name in seen:
            continue
        seen.add(name)
        meta = {k: v for k, v in item.items() if k not in {"name", "expression", "formula"}}
        out.append(ExpressionSpec(name=name, expression=expr, meta=meta))
    return out


def apply_expressions(
    df: DataFrame,
    expressions: Sequence[ExpressionSpec],
    *,
    allow_overwrite: bool = False,
    on_error: str = "raise",
) -> Tuple[DataFrame, List[str]]:
    """
    Apply expressions to dataframe and return (dataframe, added_columns).

    on_error:
      - "raise": raise on first invalid expression
      - "skip": skip invalid expressions
    """

    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be 'raise' or 'skip'")

    added: List[str] = []
    for spec in expressions:
        if not allow_overwrite and spec.name in df.columns:
            continue
        try:
            series = safe_eval_expression(spec.expression, df)
        except Exception:
            if on_error == "raise":
                raise
            continue
        df[spec.name] = series
        added.append(spec.name)
    return df, added
