from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .objectives import information_coefficient, rank_information_coefficient
from .novelty import ast_similarity_max, corr_to_library_max
from .stability import rolling_ic_std


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nan_ratio(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.isna().mean())


def _turnover(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 2:
        return float("nan")
    return float(s.diff().abs().mean())

def _safe_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def _rolling_ic_ir(factor: pd.Series, target: pd.Series, *, window: int = 50) -> tuple[Optional[float], Optional[float]]:
    if factor is None or target is None:
        return None, None
    df = pd.concat([factor.rename("x"), target.rename("y")], axis=1).dropna()
    w = int(window)
    if df.shape[0] < max(3, w):
        return None, None
    ics = df["x"].rolling(w).corr(df["y"]).dropna()
    if ics.empty:
        return None, None
    mean = float(ics.mean())
    std = float(ics.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std):
        return None, None
    ic_ir = mean / (std + 1e-9)
    return (float(ic_ir) if np.isfinite(ic_ir) else None), (float(std) if np.isfinite(std) else None)


def _trading_proxies(factor: pd.Series, target: pd.Series) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Best-effort trading proxies.

    We do not have a full execution simulator here; use a simple directional
    strategy: position = sign(factor), pnl = position * target.
    """

    if factor is None or target is None:
        return None, None, None
    df = pd.concat([factor.rename("f"), target.rename("y")], axis=1).dropna()
    if df.shape[0] < 3:
        return None, None, None

    pos = np.sign(df["f"].astype("float64")).fillna(0.0)
    pnl = pos.astype("float64") * df["y"].astype("float64")
    if pnl.empty:
        return None, None, None

    mean = float(pnl.mean())
    std = float(pnl.std(ddof=0))
    sharpe = mean / (std + 1e-9) if np.isfinite(mean) and np.isfinite(std) else float("nan")

    downside = pnl.where(pnl < 0.0).dropna()
    dstd = float(downside.std(ddof=0)) if not downside.empty else float("nan")
    sortino = mean / (dstd + 1e-9) if np.isfinite(mean) and np.isfinite(dstd) else float("nan")

    equity = (1.0 + pnl.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    mdd = float(abs(dd.min())) if not dd.empty else float("nan")

    return _safe_float(sharpe), _safe_float(sortino), _safe_float(mdd)


_EXPENSIVE_CALLS: set[str] = {
    "roll_mean",
    "roll_std",
    "rolling_mean",
    "rolling_std",
    "rolling_sum",
    "rolling_min",
    "rolling_max",
    "ema",
    "ts_z",
    "robust_z",
    "decay_linear",
    "winsorize",
}


def _expr_stats(expr: str) -> tuple[Optional[int], Optional[int], str, Optional[int]]:
    text = str(expr or "").strip()
    if not text:
        return None, None, "", None
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        return None, None, hashlib.sha256(text.encode("utf-8")).hexdigest(), None

    node_count = sum(1 for _ in ast.walk(tree))

    def _depth(node: ast.AST) -> int:
        kids = list(ast.iter_child_nodes(node))
        if not kids:
            return 1
        return 1 + max(_depth(k) for k in kids)

    depth = _depth(tree)
    expensive_ops = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _EXPENSIVE_CALLS:
            expensive_ops += 1
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(node_count), int(depth), sha, int(expensive_ops)


def _regime_consistency(factor: pd.Series, target: pd.Series, *, segments: int = 4) -> Optional[float]:
    if factor is None or target is None:
        return None
    df = pd.concat([factor.rename("x"), target.rename("y")], axis=1).dropna()
    if df.shape[0] < 12:
        return None
    overall = information_coefficient(df["x"], df["y"])
    if not np.isfinite(overall) or abs(float(overall)) < 1e-12:
        return None
    sign = 1.0 if float(overall) > 0 else -1.0

    k = max(2, int(segments))
    n = int(df.shape[0])
    seg = max(3, n // k)
    wins = 0
    total = 0
    for i in range(k):
        start = i * seg
        end = (i + 1) * seg if i < k - 1 else n
        if end - start < 3:
            continue
        ic = information_coefficient(df["x"].iloc[start:end], df["y"].iloc[start:end])
        if not np.isfinite(ic):
            continue
        total += 1
        wins += 1 if (float(ic) * sign) > 0 else 0
    if total == 0:
        return None
    return float(wins) / float(total)


def _train_test_gap(factor: pd.Series, target: pd.Series, *, train_ratio: float = 0.7) -> Optional[float]:
    if factor is None or target is None:
        return None
    df = pd.concat([factor.rename("x"), target.rename("y")], axis=1).dropna()
    if df.shape[0] < 10:
        return None
    r = float(max(0.1, min(0.9, float(train_ratio))))
    split = int(df.shape[0] * r)
    split = max(3, min(split, int(df.shape[0]) - 3))
    ic_tr = information_coefficient(df["x"].iloc[:split], df["y"].iloc[:split])
    ic_te = information_coefficient(df["x"].iloc[split:], df["y"].iloc[split:])
    if not (np.isfinite(ic_tr) and np.isfinite(ic_te)):
        return None
    return float(abs(float(ic_tr) - float(ic_te)))


def _weighted_score(
    *,
    ic_ir: Optional[float],
    sharpe_net: Optional[float],
    ic_rolling_std: Optional[float],
    turnover: Optional[float],
    max_turnover: float,
    node_count: Optional[int],
    depth: Optional[int],
) -> Optional[float]:
    if ic_ir is None and sharpe_net is None and ic_rolling_std is None and turnover is None:
        return None

    ic_ir_v = float(ic_ir or 0.0)
    sharpe_v = float(sharpe_net or 0.0)
    stability = 0.0
    if ic_rolling_std is not None and np.isfinite(ic_rolling_std):
        stability = float(1.0 / (float(ic_rolling_std) + 1e-9))
        stability = float(min(stability, 10.0))

    to_pen = 0.0
    if turnover is not None and np.isfinite(turnover):
        denom = float(max(max_turnover, 1e-9))
        to_pen = float(min(1.0, float(turnover) / denom))

    comp_pen = 0.0
    if node_count is not None and depth is not None:
        comp_pen = 0.5 * (min(1.0, float(node_count) / 100.0) + min(1.0, float(depth) / 20.0))

    score = (
        0.35 * ic_ir_v
        + 0.25 * sharpe_v
        + 0.15 * stability
        - 0.15 * to_pen
        - 0.10 * comp_pen
    )
    return float(score) if np.isfinite(score) else None


def _pareto_front(rows: list[dict[str, Any]]) -> set[str]:
    """
    Return names on a simple Pareto front.

    Objectives:
      - maximize ic_abs, rank_ic_abs
      - minimize turnover, nan_ratio
    """

    items = [r for r in rows if r.get("gate_pass") is True]
    names = [str(r["name"]) for r in items]
    if not items:
        return set()

    def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        a_ic = float(a.get("ic_abs") or float("-inf"))
        b_ic = float(b.get("ic_abs") or float("-inf"))
        a_ric = float(a.get("rank_ic_abs") or float("-inf"))
        b_ric = float(b.get("rank_ic_abs") or float("-inf"))
        a_to = float(a.get("turnover") or float("inf"))
        b_to = float(b.get("turnover") or float("inf"))
        a_nan = float(a.get("nan_ratio") or float("inf"))
        b_nan = float(b.get("nan_ratio") or float("inf"))

        better_or_equal = (
            a_ic >= b_ic
            and a_ric >= b_ric
            and a_to <= b_to
            and a_nan <= b_nan
        )
        strictly_better = (
            a_ic > b_ic
            or a_ric > b_ric
            or a_to < b_to
            or a_nan < b_nan
        )
        return bool(better_or_equal and strictly_better)

    pareto: set[str] = set()
    for i, a in enumerate(items):
        dominated = False
        for j, b in enumerate(items):
            if i == j:
                continue
            if dominates(b, a):
                dominated = True
                break
        if not dominated:
            pareto.add(str(a["name"]))
    return pareto


def score_factors_to_artifacts(
    df: pd.DataFrame,
    *,
    factor_cols: Sequence[str],
    target_col: str,
    out_dir: Path,
    max_nan_ratio: float = 0.2,
    max_turnover: float = 8.0,
    max_corr_to_library: float = 0.95,
    factor_exprs: Optional[Mapping[str, str]] = None,
    factor_library: Optional[Mapping[str, pd.Series]] = None,
    library_expr_sha256: Optional[Sequence[str]] = None,
    ic_rolling_window: int = 50,
) -> dict[str, Any]:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if target_col not in df.columns:
        raise ValueError(f"target_col not found: {target_col}")

    items: list[dict[str, Any]] = []
    for name in factor_cols:
        if name not in df.columns:
            continue
        s = df[name]
        y = df[target_col]
        ic = information_coefficient(s, y)
        rank_ic = rank_information_coefficient(s, y)
        ic_ir, ic_roll_std = _rolling_ic_ir(s, y, window=int(ic_rolling_window))
        ic_roll_std2 = rolling_ic_std(s, y, window=int(ic_rolling_window))
        if ic_roll_std is None:
            ic_roll_std = ic_roll_std2
        nan_ratio = _nan_ratio(s)
        turnover = _turnover(s)
        sharpe_net, sortino_net, mdd = _trading_proxies(s, y)
        ic_abs = float(abs(ic)) if np.isfinite(ic) else float("nan")
        rank_ic_abs = float(abs(rank_ic)) if np.isfinite(rank_ic) else float("nan")

        expr_text = (factor_exprs or {}).get(str(name))
        node_count, depth, expr_sha256, expensive_ops = _expr_stats(str(expr_text or ""))

        corr_max = corr_to_library_max(s, dict(factor_library or {})) if factor_library else None
        ast_sim = ast_similarity_max(expr_sha256, library_expr_sha256 or []) if expr_sha256 else 0.0

        regime_cons = _regime_consistency(s, y, segments=4)
        tt_gap = _train_test_gap(s, y, train_ratio=0.7)
        cap_proxy = None
        if np.isfinite(turnover):
            cap_proxy = float(1.0 / (float(turnover) + 1e-9))

        weighted = _weighted_score(
            ic_ir=ic_ir,
            sharpe_net=sharpe_net,
            ic_rolling_std=ic_roll_std,
            turnover=_safe_float(turnover),
            max_turnover=float(max_turnover),
            node_count=node_count,
            depth=depth,
        )

        corr_gate_ok = bool(corr_max is None or float(corr_max) <= float(max_corr_to_library))
        gate_pass = bool(
            np.isfinite(ic)
            and np.isfinite(rank_ic)
            and nan_ratio <= float(max_nan_ratio)
            and turnover <= float(max_turnover)
            and corr_gate_ok
        )
        items.append(
            {
                "name": str(name),
                "ic": float(ic) if np.isfinite(ic) else None,
                "rank_ic": float(rank_ic) if np.isfinite(rank_ic) else None,
                "ic_ir": _safe_float(ic_ir),
                "ic_rolling_std": _safe_float(ic_roll_std),
                "ic_abs": float(ic_abs) if np.isfinite(ic_abs) else None,
                "rank_ic_abs": float(rank_ic_abs) if np.isfinite(rank_ic_abs) else None,
                "sharpe_net": _safe_float(sharpe_net),
                "sortino_net": _safe_float(sortino_net),
                "mdd": _safe_float(mdd),
                "nan_ratio": float(nan_ratio) if np.isfinite(nan_ratio) else None,
                "turnover": float(turnover) if np.isfinite(turnover) else None,
                "corr_to_library_max": _safe_float(corr_max),
                "ast_similarity_max": _safe_float(ast_sim),
                "node_count": int(node_count) if node_count is not None else None,
                "depth": int(depth) if depth is not None else None,
                "expensive_ops": int(expensive_ops) if expensive_ops is not None else None,
                "regime_consistency": _safe_float(regime_cons),
                "train_test_gap": _safe_float(tt_gap),
                "capacity_proxy": _safe_float(cap_proxy),
                "slippage_reduction_bps": None,
                "fill_rate": None,
                "adverse_selection_proxy": None,
                "weighted_score": _safe_float(weighted),
                "gate_pass": gate_pass,
                "gates": {
                    "max_nan_ratio": float(max_nan_ratio),
                    "max_turnover": float(max_turnover),
                    "max_corr_to_library": float(max_corr_to_library),
                },
            }
        )

    pareto = _pareto_front(items)
    for row in items:
        row["pareto"] = bool(row.get("name") in pareto)

    scores_path = out_dir / "factor_scores.json"
    pareto_path = out_dir / "pareto.csv"

    payload = {
        "version": 1,
        "generated_at": _utc_now(),
        "target_col": str(target_col),
        "items": items,
        "pareto_front": sorted(pareto),
        "artifacts": {
            "factor_scores_json": str(scores_path),
            "pareto_csv": str(pareto_path),
        },
    }
    scores_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.DataFrame(items).to_csv(pareto_path, index=False)

    return payload
