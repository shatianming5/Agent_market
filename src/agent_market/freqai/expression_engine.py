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
    "allowed_expression_functions",
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

    def _literal_int(self, node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or value is None:
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            inner = self._literal_int(node.operand)
            if inner is None:
                return None
            return +inner if isinstance(node.op, ast.UAdd) else -inner
        return None

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
        if func_name == "shift":
            if len(node.args) != 2:
                raise ExpressionValidationError("shift expects 2 positional arguments")
            n = self._literal_int(node.args[1])
            if n is None:
                raise ExpressionValidationError("shift second argument must be an integer constant")
            if n < 0:
                raise ExpressionValidationError("shift second argument must be >= 0 (no lookahead)")
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

    index = df.index
    eps = 1e-12

    def _shift(series: pd.Series, n: int) -> pd.Series:
        n_int = int(n)
        if n_int < 0:
            raise ExpressionValidationError("shift second argument must be >= 0 (no lookahead)")
        return series.shift(n_int)

    def _diff(series: pd.Series, n: int = 1) -> pd.Series:
        n_int = int(n)
        if n_int <= 0:
            raise ExpressionValidationError("diff second argument must be > 0")
        return series.diff(periods=n_int)

    def _decay_linear(series: pd.Series, window: int) -> pd.Series:
        w = int(window)
        if w <= 0:
            raise ExpressionValidationError("decay_linear window must be > 0")
        weights = np.arange(1.0, float(w) + 1.0, dtype=float)
        denom = float(weights.sum())

        def _apply(x: np.ndarray) -> float:
            if x.size != weights.size:
                return float("nan")
            if np.all(np.isnan(x)):
                return float("nan")
            return float(np.nansum(x * weights) / denom)

        return series.astype("float64").rolling(w).apply(_apply, raw=True)

    def _winsorize(series: pd.Series, p: float) -> pd.Series:
        prob = float(p)
        if not (0.0 < prob < 0.5):
            raise ExpressionValidationError("winsorize p must be in (0, 0.5)")
        s = series.astype("float64")
        lo = float(s.quantile(prob))
        hi = float(s.quantile(1.0 - prob))
        return s.clip(lo, hi)

    def _robust_z(series: pd.Series, window: int, *, eps: float = 1e-9) -> pd.Series:
        w = int(window)
        if w <= 0:
            raise ExpressionValidationError("robust_z window must be > 0")
        s = series.astype("float64")
        med = s.rolling(w).median()

        def _mad(x: np.ndarray) -> float:
            if x.size == 0:
                return float("nan")
            m = np.nanmedian(x)
            return float(np.nanmedian(np.abs(x - m)))

        mad = s.rolling(w).apply(_mad, raw=True)
        scale = (1.4826 * mad) + float(eps)
        return (s - med) / scale

    def _ifelse(cond: Any, a: Any, b: Any) -> pd.Series:
        return pd.Series(np.where(cond, a, b), index=index)

    def _xs_group_key(group: Any | None) -> pd.Series:
        if group is None:
            if "date" in df.columns:
                return df["date"]
            if "ts" in df.columns:
                return df["ts"]
            raise ExpressionValidationError("Cross-sectional op requires a group key (provide df['date'] or df['ts'])")
        if isinstance(group, pd.Series):
            return group
        return pd.Series(group, index=index)

    def _rank_xs(series: pd.Series, group: Any | None = None) -> pd.Series:
        g = _xs_group_key(group)
        x = pd.Series(series, index=index).astype("float64")
        tmp = pd.DataFrame({"g": g, "x": x})
        return tmp.groupby("g", sort=False)["x"].rank(method="average", pct=True)

    def _zscore_xs(series: pd.Series, group: Any | None = None) -> pd.Series:
        g = _xs_group_key(group)
        x = pd.Series(series, index=index).astype("float64")
        tmp = pd.DataFrame({"g": g, "x": x})
        mean = tmp.groupby("g", sort=False)["x"].transform("mean")
        var = ((x - mean) ** 2).groupby(g, sort=False).transform("mean")
        std = np.sqrt(var)
        return (x - mean) / (std + 1e-9)

    def _corr_xs(x: pd.Series, y: pd.Series, group: Any | None = None) -> pd.Series:
        g = _xs_group_key(group)
        xs = pd.Series(x, index=index).astype("float64")
        ys = pd.Series(y, index=index).astype("float64")
        tmp = pd.DataFrame({"g": g, "x": xs, "y": ys})
        mean_x = tmp.groupby("g", sort=False)["x"].transform("mean")
        mean_y = tmp.groupby("g", sort=False)["y"].transform("mean")
        x0 = xs - mean_x
        y0 = ys - mean_y
        cov = (x0 * y0).groupby(g, sort=False).transform("mean")
        var_x = (x0**2).groupby(g, sort=False).transform("mean")
        var_y = (y0**2).groupby(g, sort=False).transform("mean")
        return cov / (np.sqrt(var_x) * np.sqrt(var_y) + float(eps))

    def _neutralize(series: pd.Series, *against: Any) -> pd.Series:
        g = _xs_group_key(None)
        y = pd.Series(series, index=index).astype("float64")
        cols: Dict[str, Any] = {"g": g, "y": y}
        for i, a in enumerate(against):
            cols[f"x{i}"] = pd.Series(a, index=index).astype("float64")
        tmp = pd.DataFrame(cols)

        if not against:
            return y - y.groupby(g, sort=False).transform("mean")

        feature_cols = [c for c in tmp.columns if c.startswith("x")]

        def _resid(gdf: pd.DataFrame) -> pd.Series:
            # Drop rows with any non-finite value.
            mat = gdf[feature_cols].to_numpy(dtype=float)
            vec = gdf["y"].to_numpy(dtype=float)
            mask = np.isfinite(vec) & np.all(np.isfinite(mat), axis=1)
            if int(mask.sum()) < max(2, 1 + len(feature_cols)):
                return pd.Series([float("nan")] * int(gdf.shape[0]), index=gdf.index)
            X = mat[mask]
            yv = vec[mask]
            # Add intercept.
            Xd = np.concatenate([np.ones((X.shape[0], 1), dtype=float), X], axis=1)
            beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
            y_hat = np.concatenate([np.ones((mat.shape[0], 1), dtype=float), mat], axis=1) @ beta
            return pd.Series(vec - y_hat, index=gdf.index)

        grouped = tmp.groupby("g", sort=False, group_keys=False)
        return grouped[feature_cols + ["y"]].apply(_resid)

    def _sigmoid(x: pd.Series) -> pd.Series:
        return 1.0 / (1.0 + np.exp(-x.astype("float64")))

    def _fill_prob(limit_px_offset: Any, horizon: Any) -> pd.Series:
        """
        Best-effort fill probability proxy for limit orders.

        Uses available microstructure columns when present; otherwise falls back to a stable constant.
        """

        try:
            off = float(limit_px_offset)
        except Exception:
            off = 0.0
        try:
            h = float(horizon)
        except Exception:
            h = 0.0

        if "mid" in df.columns and "spread" in df.columns:
            mid = pd.Series(df["mid"], index=index).astype("float64")
            spread = pd.Series(df["spread"], index=index).astype("float64")
            spread_bps = 10000.0 * (spread / (mid.abs() + 1e-9))
        else:
            spread_bps = pd.Series([0.0] * int(df.shape[0]), index=index)

        depth_bid = None
        depth_ask = None
        for name in ("depth_bid_20", "depth_bid_10", "depth_bid"):
            if name in df.columns:
                depth_bid = pd.Series(df[name], index=index).astype("float64")
                break
        for name in ("depth_ask_20", "depth_ask_10", "depth_ask"):
            if name in df.columns:
                depth_ask = pd.Series(df[name], index=index).astype("float64")
                break
        depth_total = None
        if depth_bid is not None and depth_ask is not None:
            depth_total = depth_bid.fillna(0.0) + depth_ask.fillna(0.0)

        imbalance = None
        for name in ("imbalance_20", "imbalance_10", "imbalance"):
            if name in df.columns:
                imbalance = pd.Series(df[name], index=index).astype("float64")
                break

        intensity = None
        for name in ("arrival_intensity_10", "arrival_intensity_5", "arrival_intensity"):
            if name in df.columns:
                intensity = pd.Series(df[name], index=index).astype("float64")
                break

        # Score: lower spread/offset, higher depth/intensity -> higher fill probability.
        score = pd.Series([0.0] * int(df.shape[0]), index=index, dtype="float64")
        score = score - 0.25 * spread_bps.fillna(0.0)
        score = score - 5.0 * abs(float(off))
        if depth_total is not None:
            score = score + 0.02 * np.log1p(depth_total.clip(lower=0.0))
        if imbalance is not None:
            score = score + 0.5 * imbalance.abs().fillna(0.0)
        if intensity is not None and h > 0:
            score = score + 0.2 * np.log1p(intensity.clip(lower=0.0) * float(h))

        prob = _sigmoid(score)
        return prob.clip(lower=0.0, upper=1.0)

    def _impact_proxy(window: Any) -> pd.Series:
        """
        Best-effort market impact proxy.

        Heuristic:
          impact ~ |OFI| * rolling_mean(|return|)
        """

        try:
            w = int(window)
        except Exception:
            w = 1
        w = max(1, min(w, 5000))

        if "mid" in df.columns:
            ret = pd.Series(df["mid"], index=index).astype("float64").pct_change()
        elif "close" in df.columns:
            ret = pd.Series(df["close"], index=index).astype("float64").pct_change()
        else:
            ret = pd.Series([0.0] * int(df.shape[0]), index=index, dtype="float64")

        ofi = None
        name = f"ofi_{w}"
        if name in df.columns:
            ofi = pd.Series(df[name], index=index).astype("float64")
        elif "ofi_10" in df.columns:
            ofi = pd.Series(df["ofi_10"], index=index).astype("float64")
        elif "ofi" in df.columns:
            ofi = pd.Series(df["ofi"], index=index).astype("float64")
        else:
            ofi = pd.Series([0.0] * int(df.shape[0]), index=index, dtype="float64")

        ret_abs = ret.abs().fillna(0.0)
        base = ret_abs.rolling(w).mean().fillna(0.0)
        return (ofi.abs().fillna(0.0) * base).astype("float64")

    def _queue_pos_proxy() -> pd.Series:
        """Best-effort queue position proxy in [0, 1] (0=front, 1=back)."""
        for name in ("imbalance_20", "imbalance_10", "imbalance"):
            if name in df.columns:
                imb = pd.Series(df[name], index=index).astype("float64").clip(lower=-1.0, upper=1.0).fillna(0.0)
                return ((1.0 - imb) / 2.0).clip(lower=0.0, upper=1.0)
        return pd.Series([0.5] * int(df.shape[0]), index=index, dtype="float64")

    def _zscore_dispatch(*args: Any) -> pd.Series:
        if len(args) == 1:
            return _z(args[0])
        if len(args) == 2:
            return _ts_z(args[0], args[1])
        raise ExpressionValidationError("zscore expects 1 or 2 positional arguments")

    env.update(
        {
            "z": _z,
            "ts_z": _ts_z,
            "zscore": _zscore_dispatch,
            "shift": _shift,
            "diff": _diff,
            "roll_mean": lambda s, w: s.rolling(int(w)).mean(),
            "roll_std": lambda s, w: s.rolling(int(w)).std(ddof=0),
            "rolling_mean": lambda s, w: s.rolling(int(w)).mean(),
            "rolling_std": lambda s, w: s.rolling(int(w)).std(ddof=0),
            "rolling_sum": lambda s, w: s.rolling(int(w)).sum(),
            "pct_change": lambda s, n: s.pct_change(periods=int(n)),
            "sign": lambda s: np.sign(s),
            "clip": lambda s, lo, hi: s.clip(float(lo), float(hi)),
            "ema": lambda s, span: s.ewm(span=int(span), adjust=False).mean(),
            "rolling_max": lambda s, w: s.rolling(int(w)).max(),
            "rolling_min": lambda s, w: s.rolling(int(w)).min(),
            "decay_linear": _decay_linear,
            "winsorize": _winsorize,
            "robust_z": _robust_z,
            "log1p": lambda s: np.log1p(s),
            "log": lambda s: np.log(s),
            "exp": lambda s: np.exp(s),
            "sqrt": lambda s: np.sqrt(s),
            "tanh": lambda s: np.tanh(s),
            "abs": abs,
            "ifelse": _ifelse,
            "rank_xs": _rank_xs,
            "zscore_xs": _zscore_xs,
            "corr_xs": _corr_xs,
            "neutralize": _neutralize,
            "fill_prob": _fill_prob,
            "impact_proxy": _impact_proxy,
            "queue_pos_proxy": _queue_pos_proxy,
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


def allowed_expression_functions() -> set[str]:
    """Return the function names accepted by the expression DSL validator."""
    probe = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [2.0, 3.0, 4.0, 5.0],
            "low": [0.5, 1.5, 2.5, 3.5],
            "close": [1.5, 2.5, 3.5, 4.5],
            "volume": [10.0, 11.0, 12.0, 13.0],
            "date": [0, 0, 1, 1],
        }
    )
    env = _build_eval_env(probe)
    return {name for name, value in env.items() if callable(value)}


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
