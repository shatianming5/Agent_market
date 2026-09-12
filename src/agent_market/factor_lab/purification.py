"""Cross-sectional factor purification and exposure diagnostics.

The functions here are intentionally lightweight and dependency-free.  They
operate per timestamp and never inspect labels, so they can run inside mining
without creating a forward-return leakage path.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
import re
import warnings
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .cache import CACHE_VERSION, get_cache, panel_fingerprint, stable_hash


DEFAULT_EXPOSURE_GROUPS: Tuple[str, ...] = (
    "market",
    "pair",
    "volatility",
    "liquidity",
    "funding",
    "mtf",
    "micro",
)

EXPOSURE_PATTERNS: Mapping[str, Tuple[str, ...]] = {
    "market": (
        r"(^|_)ret(_|$)",
        r"return",
        r"momentum",
        r"relative_strength",
        r"xs_ret",
        r"btc",
        r"eth",
        r"sol",
        r"beta",
    ),
    "pair": (
        r"^pair_",
        r"pair_",
        r"xs_ret_vs_btc",
        r"relative_strength",
        r"beta",
    ),
    "volatility": (
        r"atr_norm",
        r"realized_vol",
        r"(^|_)rv_",
        r"volatility",
        r"roll_std",
        r"rolling_std",
        r"range_pct",
        r"donchian",
        r"bb_",
    ),
    "liquidity": (
        r"volume",
        r"dollar_vol",
        r"amihud",
        r"liquidity",
        r"turnover",
    ),
    "funding": (
        r"^funding",
        r"funding_",
        r"open_interest",
        r"premium",
        r"basis",
    ),
    "mtf": (
        r"^mtf",
        r"mtf4h",
    ),
    "micro": (
        r"^micro",
        r"spread",
        r"\bofi\b",
        r"imbalance",
        r"depth",
        r"vwap",
        r"queue",
        r"bid",
        r"ask",
        r"lob",
    ),
}

RESERVED_COLUMNS = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "__pair__",
    "__fwd_ret__",
    "__fwd_raw__",
    "__fwd_ref__",
    "__ref_close__",
    "__ret_ref_1__",
}

_EXPOSURE_FRAME_CACHE: Dict[Tuple[int, int, Tuple[str, ...], Tuple[str, ...]], Tuple[pd.DataFrame, List[str]]] = {}


@dataclass(frozen=True)
class PurifyConfig:
    mode: str = "off"
    winsor: str = "mad"
    standardize: str = "zscore"
    neutralize: str = "ridge"
    exposures: Tuple[str, ...] = DEFAULT_EXPOSURE_GROUPS
    ridge_alpha: float = 1e-3
    cache_dir: str | None = None
    no_cache: bool = False
    panel_fingerprint: str = ""


@dataclass
class PurifyResult:
    raw: pd.Series
    clean: pd.Series
    neutralized: pd.Series
    selected: pd.Series
    exposure_frame: pd.DataFrame
    exposure_columns: List[str]
    diagnostics: Dict[str, float | int | str | List[str]]


def parse_exposure_groups(value: str | Sequence[str] | None) -> Tuple[str, ...]:
    if value is None:
        return DEFAULT_EXPOSURE_GROUPS
    if isinstance(value, str):
        parts = [p.strip().lower() for p in value.split(",") if p.strip()]
    else:
        parts = [str(p).strip().lower() for p in value if str(p).strip()]
    if not parts:
        return DEFAULT_EXPOSURE_GROUPS
    return tuple(dict.fromkeys(parts))


def _safe_series(series: pd.Series, index: pd.Index) -> pd.Series:
    return pd.Series(series, index=index, dtype="float64").replace([np.inf, -np.inf], np.nan)


def _date_groups(panel: pd.DataFrame) -> pd.core.groupby.SeriesGroupBy:
    if "date" not in panel.columns:
        raise ValueError("purification requires a date column")
    return panel["date"]


def winsorize_cross_section(
    panel: pd.DataFrame,
    factor: pd.Series,
    method: str = "mad",
    *,
    mad_k: float = 5.0,
    quantile: float = 0.01,
) -> pd.Series:
    """Clip factor extremes within each timestamp cross-section."""
    mode = str(method or "none").lower()
    s = _safe_series(factor, panel.index)
    if mode == "none":
        return s
    group_sizes = panel.groupby(_date_groups(panel), sort=False).size()
    if group_sizes.empty or int(group_sizes.max()) < 3:
        return s

    dates = _date_groups(panel)
    work = pd.DataFrame({"date": dates, "x": s}, index=panel.index)
    grouped = work.groupby("date", sort=False)["x"]
    if mode == "mad":
        med = grouped.transform("median")
        mad = (work["x"] - med).abs().groupby(dates, sort=False).transform("median")
        width = float(mad_k) * 1.4826 * mad
        out = work["x"].clip(med - width, med + width).where(mad > 1e-12, work["x"])
    elif mode == "quantile":
        q = min(max(float(quantile), 0.0), 0.49)
        lo = grouped.transform(lambda x: x.quantile(q))
        hi = grouped.transform(lambda x: x.quantile(1.0 - q))
        out = work["x"].clip(lo, hi)
    elif mode == "iqr":
        q1 = grouped.transform(lambda x: x.quantile(0.25))
        q3 = grouped.transform(lambda x: x.quantile(0.75))
        iqr = q3 - q1
        out = work["x"].clip(q1 - 3.0 * iqr, q3 + 3.0 * iqr).where(iqr > 1e-12, work["x"])
    else:
        raise ValueError(f"unknown winsorization method: {method}")
    return _safe_series(out, panel.index)


def standardize_cross_section(
    panel: pd.DataFrame,
    factor: pd.Series,
    method: str = "zscore",
) -> pd.Series:
    """Normalize factor values within each timestamp cross-section."""
    mode = str(method or "none").lower()
    s = _safe_series(factor, panel.index)
    if mode == "none":
        return s
    work = pd.DataFrame({"date": _date_groups(panel), "x": s}, index=panel.index)

    if mode == "zscore":
        dates = work["date"]
        mean = work.groupby("date", sort=False)["x"].transform("mean")
        centered = work["x"] - mean
        std = np.sqrt((centered ** 2).groupby(dates, sort=False).transform("mean"))
        out = (work["x"] - mean) / (std + 1e-9)
        out = out.where(std > 1e-12, 0.0)
        return _safe_series(out, panel.index)

    if mode == "rank":
        rank = work.groupby("date", sort=False)["x"].rank(method="average", pct=True)
        return _safe_series(2.0 * rank - 1.0, panel.index)

    if mode == "rank_gaussianize":
        pct = work.groupby("date", sort=False)["x"].rank(method="average", pct=True)
        clipped = pct.clip(1e-4, 1.0 - 1e-4)
        nd = NormalDist()
        out = clipped.map(nd.inv_cdf)
        return standardize_cross_section(panel, out, "zscore")

    raise ValueError(f"unknown standardization method: {method}")


def exposure_category(column: str) -> str:
    text = str(column).lower()
    priority = ("funding", "mtf", "micro", "pair", "volatility", "liquidity", "market")
    for group in priority:
        patterns = EXPOSURE_PATTERNS[group]
        if any(re.search(pattern, text) for pattern in patterns):
            return group
    return "other"


def build_exposure_frame(
    panel: pd.DataFrame,
    groups: Sequence[str] | str | None = None,
    *,
    cache_dir: str | None = None,
    no_cache: bool = False,
    panel_fingerprint_hint: str | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build numeric exposure candidates aligned to ``panel.index``.

    The builder uses existing feature columns plus a few causal return proxies.
    Forward labels are deliberately excluded.
    """
    selected_groups = set(parse_exposure_groups(groups))
    cache_key = (id(panel), len(panel), tuple(map(str, panel.columns)), tuple(sorted(selected_groups)))
    cached = _EXPOSURE_FRAME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    persistent_key = ""
    cache_enabled = bool(cache_dir) and not bool(no_cache)
    cache = get_cache(cache_dir, no_cache=not cache_enabled)
    if cache_enabled:
        pfp = panel_fingerprint_hint or str(panel.attrs.get("factor_lab_panel_key") or "")
        if not pfp:
            pfp = panel_fingerprint(panel)
        persistent_key = stable_hash({
            "kind": "exposure_frame",
            "cache_version": CACHE_VERSION,
            "panel": pfp,
            "groups": sorted(selected_groups),
            "patterns": EXPOSURE_PATTERNS,
            "reserved": sorted(RESERVED_COLUMNS),
        })
        loaded = cache.load_exposure(persistent_key)
        if loaded is not None:
            out_cached, cols_cached, _ = loaded
            out_cached.index = panel.index
            if len(_EXPOSURE_FRAME_CACHE) > 8:
                _EXPOSURE_FRAME_CACHE.clear()
            _EXPOSURE_FRAME_CACHE[cache_key] = (out_cached, cols_cached)
            return out_cached, cols_cached

    out = pd.DataFrame(index=panel.index)
    if "date" in panel.columns:
        out["date"] = panel["date"]
    if "__pair__" in panel.columns:
        out["__pair__"] = panel["__pair__"]

    numeric = panel.select_dtypes(include=[np.number]).copy()
    candidate_cols: List[str] = []
    for col in numeric.columns:
        if col in RESERVED_COLUMNS or col.startswith("__fwd"):
            continue
        cat = exposure_category(col)
        if cat in selected_groups:
            candidate_cols.append(col)

    if candidate_cols:
        out = pd.concat([out, numeric[candidate_cols]], axis=1)

    if {"close", "__pair__"}.issubset(panel.columns):
        close = pd.Series(panel["close"], index=panel.index, dtype="float64")
        by_pair = close.groupby(panel["__pair__"], sort=False)
        if "market" in selected_groups or "pair" in selected_groups:
            out["ret_1"] = by_pair.pct_change(1)
            out["ret_24"] = by_pair.pct_change(24)
            out["ret_72"] = by_pair.pct_change(72)
            out["relative_strength_24"] = by_pair.pct_change(24)
        if "market" in selected_groups:
            for sym in ("BTC", "ETH", "SOL"):
                pair = f"{sym}/USDT"
                ref = panel.loc[panel["__pair__"] == pair, ["date", "close"]]
                if ref.empty or "date" not in panel.columns:
                    continue
                ref = ref.drop_duplicates("date", keep="last").sort_values("date")
                ref[f"{sym.lower()}_ret_1"] = ref["close"].pct_change(1)
                ref[f"{sym.lower()}_ret_24"] = ref["close"].pct_change(24)
                out = out.merge(
                    ref[["date", f"{sym.lower()}_ret_1", f"{sym.lower()}_ret_24"]],
                    on="date",
                    how="left",
                    sort=False,
                )
                out.index = panel.index

    exposure_cols = [
        c for c in out.columns
        if c not in {"date", "__pair__"} and pd.api.types.is_numeric_dtype(out[c])
    ]
    exposure_cols = list(dict.fromkeys(exposure_cols))
    out[exposure_cols] = out[exposure_cols].replace([np.inf, -np.inf], np.nan)
    if len(_EXPOSURE_FRAME_CACHE) > 8:
        _EXPOSURE_FRAME_CACHE.clear()
    _EXPOSURE_FRAME_CACHE[cache_key] = (out, exposure_cols)
    if cache_enabled and persistent_key:
        cache.save_exposure(
            persistent_key,
            out,
            exposure_cols,
            {
                "panel_fingerprint": panel_fingerprint_hint or str(panel.attrs.get("factor_lab_panel_key") or ""),
                "groups": sorted(selected_groups),
                "rows": int(len(out)),
            },
        )
    return out, exposure_cols


def _standardize_matrix(x: pd.DataFrame) -> pd.DataFrame:
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    good = std > 1e-12
    if not good.any():
        return pd.DataFrame(index=x.index)
    return (x.loc[:, good.index[good]] - mean.loc[good]) / (std.loc[good] + 1e-9)


def _select_exposure_subset(x: pd.DataFrame, y: pd.Series, max_cols: int) -> pd.DataFrame:
    if x.empty or max_cols <= 0:
        return pd.DataFrame(index=x.index)
    if x.shape[1] <= max_cols:
        return x
    scores: Dict[str, float] = {}
    for col in x.columns:
        xv = x[col]
        mask = xv.notna() & y.notna()
        if mask.sum() < 3 or xv.loc[mask].std(ddof=0) <= 1e-12:
            scores[col] = 0.0
            continue
        scores[col] = abs(float(xv.loc[mask].corr(y.loc[mask])))
    keep = [col for col, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:max_cols]]
    return x[keep]


def _global_exposure_subset(
    factor: pd.Series,
    exposures: pd.DataFrame,
    exposure_cols: Sequence[str],
    max_cols: int,
    *,
    max_rows: int = 50_000,
) -> List[str]:
    cols = list(exposure_cols)
    if not cols or max_cols <= 0:
        return []
    if len(cols) <= max_cols:
        return cols
    y = pd.Series(factor, index=exposures.index, dtype="float64").replace([np.inf, -np.inf], np.nan)
    valid_y = y.notna() & np.isfinite(y)
    if valid_y.sum() < 100:
        return []
    idx = np.flatnonzero(valid_y.to_numpy())
    if len(idx) > max_rows:
        take = np.linspace(0, len(idx) - 1, int(max_rows), dtype=int)
        idx = idx[take]
    yv = y.iloc[idx].to_numpy(dtype=float)
    y_std = float(np.nanstd(yv))
    if y_std <= 1e-12:
        return []
    scores: Dict[str, float] = {}
    for col in cols:
        xv = pd.to_numeric(exposures[col].iloc[idx], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(xv) & np.isfinite(yv)
        if int(mask.sum()) < 100:
            scores[col] = 0.0
            continue
        xs = float(np.nanstd(xv[mask]))
        if xs <= 1e-12:
            scores[col] = 0.0
            continue
        scores[col] = abs(float(np.corrcoef(xv[mask], yv[mask])[0, 1]))
    return [col for col, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:max_cols]]


def _neutralize_balanced_panel(
    panel: pd.DataFrame,
    y_all: pd.Series,
    exposures: pd.DataFrame,
    selected_cols: Sequence[str],
    *,
    ridge_alpha: float,
) -> pd.Series | None:
    if "__pair__" not in panel.columns or not selected_cols:
        return None
    n = len(panel)
    if n == 0:
        return y_all
    date_codes, date_uniques = pd.factorize(panel["date"], sort=False)
    pair_codes, pair_uniques = pd.factorize(panel["__pair__"], sort=False)
    t_count = len(date_uniques)
    p_count = len(pair_uniques)
    if t_count <= 0 or p_count < 3:
        return None
    # Keep memory bounded and require near-rectangular panels for the vectorized path.
    if t_count * p_count > max(n * 2, n + p_count):
        return None

    k_count = len(selected_cols)
    y_values = y_all.to_numpy(dtype=float)
    x_values = exposures[list(selected_cols)].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

    y_mat = np.full((t_count, p_count), np.nan, dtype=float)
    x_mat = np.full((t_count, p_count, k_count), np.nan, dtype=float)
    y_mat[date_codes, pair_codes] = y_values
    for k in range(k_count):
        x_mat[date_codes, pair_codes, k] = x_values[:, k]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            x_mean = np.nanmean(x_mat, axis=1, keepdims=True)
            x_std = np.nanstd(x_mat, axis=1, keepdims=True)
    x_mean = np.where(np.isfinite(x_mean), x_mean, 0.0)
    x_std = np.where((np.isfinite(x_std)) & (x_std > 1e-12), x_std, 1.0)
    x_std_mat = np.where(np.isfinite(x_mat), (x_mat - x_mean) / (x_std + 1e-9), 0.0)

    y_valid = np.isfinite(y_mat)
    valid_counts = y_valid.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            y_mean = np.nanmean(y_mat, axis=1, keepdims=True)
    y_mean = np.where(np.isfinite(y_mean), y_mean, 0.0)
    y0 = np.where(y_valid, y_mat - y_mean, 0.0)
    x_eff = np.where(y_valid[:, :, None], x_std_mat, 0.0)

    xtx = np.einsum("tnk,tnl->tkl", x_eff, x_eff)
    xty = np.einsum("tnk,tn->tk", x_eff, y0)
    eye = np.eye(k_count, dtype=float)[None, :, :]
    alpha = max(float(ridge_alpha), 1e-9)
    try:
        beta = np.linalg.solve(xtx + alpha * eye, xty[..., None])[..., 0]
    except np.linalg.LinAlgError:
        beta = np.einsum("tkl,tl->tk", np.linalg.pinv(xtx + alpha * eye), xty)
    pred = np.einsum("tnk,tk->tn", x_eff, beta)
    resid = y0 - pred
    resid = np.where((valid_counts[:, None] >= 3) & y_valid, resid, y_mat)

    out = y_values.copy()
    out[:] = resid[date_codes, pair_codes]
    return _safe_series(pd.Series(out, index=panel.index), panel.index)


def _sample_group_items(panel: pd.DataFrame, max_groups: int = 128) -> List[Tuple[object, np.ndarray]]:
    items = [(k, np.asarray(v, dtype=int)) for k, v in panel.groupby(_date_groups(panel), sort=False).groups.items()]
    if len(items) <= max_groups:
        return items
    take = np.linspace(0, len(items) - 1, int(max_groups), dtype=int)
    return [items[int(i)] for i in take]


def neutralize_cross_section(
    panel: pd.DataFrame,
    factor: pd.Series,
    exposures: pd.DataFrame,
    exposure_cols: Sequence[str],
    method: str = "ridge",
    *,
    ridge_alpha: float = 1e-3,
) -> pd.Series:
    """Residualize factor against exposures independently per timestamp."""
    mode = str(method or "none").lower()
    y_all = _safe_series(factor, panel.index)
    if mode == "none" or not exposure_cols:
        return y_all
    if mode not in {"ols", "ridge"}:
        raise ValueError(f"unknown neutralization method: {method}")
    group_sizes = panel.groupby(_date_groups(panel), sort=False).size()
    if group_sizes.empty or int(group_sizes.max()) < 3:
        return y_all

    max_group = int(group_sizes.max())
    selected_cols = _global_exposure_subset(
        y_all,
        exposures,
        exposure_cols,
        max_cols=max(1, min(max_group - 2, 8)),
    )
    if not selected_cols:
        return y_all

    balanced = _neutralize_balanced_panel(
        panel,
        y_all,
        exposures,
        selected_cols,
        ridge_alpha=ridge_alpha,
    )
    if balanced is not None:
        return balanced

    y_values = y_all.to_numpy(dtype=float)
    x_values = exposures[selected_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    out = y_values.copy()
    alpha = max(float(ridge_alpha), 1e-9)
    for _, idx in panel.groupby(_date_groups(panel), sort=False).groups.items():
        pos = panel.index.get_indexer(pd.Index(idx))
        pos = pos[pos >= 0]
        if pos.size == 0:
            continue
        y = y_values[pos]
        x = x_values[pos, :]
        valid = np.isfinite(y) & (np.isfinite(x).sum(axis=1) > 0)
        if int(valid.sum()) < 3 or float(np.nanstd(y[valid])) <= 1e-12:
            continue

        yv = y[valid]
        xv = x[valid, :]
        finite_counts = np.isfinite(xv).sum(axis=0)
        good_cols = finite_counts >= 3
        if not np.any(good_cols):
            continue
        xv = xv[:, good_cols]
        med = np.nanmedian(xv, axis=0)
        inds = np.where(~np.isfinite(xv))
        if inds[0].size:
            xv[inds] = np.take(med, inds[1])
        x_mean = xv.mean(axis=0)
        x_std = xv.std(axis=0)
        good_std = x_std > 1e-12
        if not np.any(good_std):
            continue
        xmat = (xv[:, good_std] - x_mean[good_std]) / (x_std[good_std] + 1e-9)
        yvec = yv - yv.mean()
        try:
            if mode == "ridge":
                xtx = xmat.T @ xmat
                beta = np.linalg.solve(xtx + alpha * np.eye(xtx.shape[0]), xmat.T @ yvec)
            else:
                beta = np.linalg.lstsq(xmat, yvec, rcond=None)[0]
            resid = yvec - xmat @ beta
        except np.linalg.LinAlgError:
            continue
        valid_pos = pos[valid]
        out[valid_pos] = resid
    return _safe_series(pd.Series(out, index=panel.index), panel.index)


def exposure_r2_by_date(
    panel: pd.DataFrame,
    factor: pd.Series,
    exposures: pd.DataFrame,
    exposure_cols: Sequence[str],
    *,
    ridge_alpha: float = 1e-3,
) -> float:
    if not exposure_cols:
        return 0.0
    group_sizes = panel.groupby(_date_groups(panel), sort=False).size()
    if group_sizes.empty or int(group_sizes.max()) < 4:
        return 0.0
    y_all = _safe_series(factor, panel.index)
    values: List[float] = []
    for _, idx in _sample_group_items(panel, max_groups=128):
        idx = pd.Index(idx)
        y = y_all.loc[idx]
        x = exposures.loc[idx, list(exposure_cols)]
        valid = y.notna() & np.isfinite(y)
        valid &= x.notna().sum(axis=1) > 0
        if valid.sum() < 4 or y.loc[valid].std(ddof=0) <= 1e-12:
            continue
        yv = y.loc[valid].astype("float64")
        xv = x.loc[valid].astype("float64").loc[:, lambda d: d.notna().sum(axis=0) >= 3]
        if xv.empty:
            continue
        xv = _standardize_matrix(xv.fillna(xv.median(axis=0)))
        xv = _select_exposure_subset(xv, yv, max_cols=max(1, int(valid.sum()) - 2))
        if xv.empty:
            continue
        y0 = yv - yv.mean()
        xmat = xv.to_numpy(dtype=float)
        yvec = y0.to_numpy(dtype=float)
        try:
            xtx = xmat.T @ xmat
            beta = np.linalg.solve(xtx + max(float(ridge_alpha), 1e-9) * np.eye(xtx.shape[0]), xmat.T @ yvec)
            pred = xmat @ beta
        except np.linalg.LinAlgError:
            continue
        denom = float(np.sum((yvec - yvec.mean()) ** 2))
        if denom <= 1e-12:
            continue
        r2 = 1.0 - float(np.sum((yvec - pred) ** 2)) / denom
        values.append(max(0.0, min(1.0, r2)))
    return float(np.mean(values)) if values else 0.0


def max_abs_exposure_corr(
    panel: pd.DataFrame,
    factor: pd.Series,
    exposures: pd.DataFrame,
    exposure_cols: Sequence[str],
    *,
    max_groups: int = 128,
) -> float:
    if not exposure_cols:
        return 0.0
    group_sizes = panel.groupby(_date_groups(panel), sort=False).size()
    if group_sizes.empty or int(group_sizes.max()) < 3:
        return 0.0
    y_all = _safe_series(factor, panel.index)
    by_col: Dict[str, List[float]] = {c: [] for c in exposure_cols}
    for _, idx in _sample_group_items(panel, max_groups=max_groups):
        idx = pd.Index(idx)
        y = y_all.loc[idx]
        if y.notna().sum() < 3 or y.std(ddof=0) <= 1e-12:
            continue
        for col in exposure_cols:
            x = exposures.loc[idx, col]
            mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3 or x.loc[mask].std(ddof=0) <= 1e-12:
                continue
            corr = y.loc[mask].corr(x.loc[mask])
            if np.isfinite(corr):
                by_col[col].append(abs(float(corr)))
    means = [float(np.mean(vals)) for vals in by_col.values() if vals]
    return float(max(means)) if means else 0.0


def residual_exposure_r2(panel: pd.DataFrame, clean: pd.Series, residual: pd.Series) -> float:
    """Fast global R2 implied by neutralization residuals."""
    y = _safe_series(clean, panel.index)
    r = _safe_series(residual, panel.index)
    mean = y.groupby(_date_groups(panel), sort=False).transform("mean")
    y0 = y - mean
    mask = y0.notna() & r.notna() & np.isfinite(y0) & np.isfinite(r)
    if int(mask.sum()) < 10:
        return 0.0
    denom = float(np.sum(np.square(y0.loc[mask].to_numpy(dtype=float))))
    if denom <= 1e-12:
        return 0.0
    resid_ss = float(np.sum(np.square(r.loc[mask].to_numpy(dtype=float))))
    return float(max(0.0, min(1.0, 1.0 - resid_ss / denom)))


def apply_purification(
    panel: pd.DataFrame,
    raw_factor: pd.Series,
    cfg: PurifyConfig,
) -> PurifyResult:
    raw = _safe_series(raw_factor, panel.index)
    exposure_frame, exposure_cols = build_exposure_frame(
        panel,
        cfg.exposures,
        cache_dir=cfg.cache_dir,
        no_cache=cfg.no_cache,
        panel_fingerprint_hint=cfg.panel_fingerprint or None,
    )

    clean = winsorize_cross_section(panel, raw, cfg.winsor)
    clean = standardize_cross_section(panel, clean, cfg.standardize)
    neutralized = neutralize_cross_section(
        panel,
        clean,
        exposure_frame,
        exposure_cols,
        cfg.neutralize,
        ridge_alpha=cfg.ridge_alpha,
    )

    mode = str(cfg.mode or "off").lower()
    if mode == "clean":
        selected = clean
    elif mode in {"neutralized", "blend"}:
        selected = neutralized
    else:
        selected = raw

    target_for_corr = neutralized if mode in {"neutralized", "blend"} else clean
    diag_cols = _global_exposure_subset(
        target_for_corr,
        exposure_frame,
        exposure_cols,
        max_cols=min(8, max(1, int(panel.groupby(_date_groups(panel), sort=False).size().max()) - 2))
        if exposure_cols else 0,
        max_rows=25_000,
    )
    diagnostics: Dict[str, float | int | str | List[str]] = {
        "purify_mode": mode,
        "purify_winsor": cfg.winsor,
        "purify_standardize": cfg.standardize,
        "purify_neutralize": cfg.neutralize,
        "purify_exposures": list(cfg.exposures),
        "exposure_count": int(len(exposure_cols)),
        "exposure_r2": residual_exposure_r2(panel, clean, neutralized),
        "max_exposure_corr": max_abs_exposure_corr(panel, target_for_corr, exposure_frame, diag_cols),
    }
    return PurifyResult(
        raw=raw,
        clean=clean,
        neutralized=neutralized,
        selected=selected,
        exposure_frame=exposure_frame,
        exposure_columns=exposure_cols,
        diagnostics=diagnostics,
    )
