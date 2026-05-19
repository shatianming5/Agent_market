"""FactorLab diagnostic reports for mined factors."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import mining
from .cache import DEFAULT_CACHE_DIR, panel_fingerprint
from .paths import LAB_STATE, USER_DATA
from .purification import (
    DEFAULT_EXPOSURE_GROUPS,
    PurifyConfig,
    build_exposure_frame,
    exposure_category,
    parse_exposure_groups,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _report_dir(tag: str) -> Path:
    out = LAB_STATE / "reports" / tag
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_candidates(tag: str, n: int, score_mode: str = "portfolio") -> List[mining.CandidateRecord]:
    state = mining.load_state(tag)
    if state:
        _, survivors, _ = state
        survivors = [mining.annotate_diversity(c) for c in survivors]
        return sorted(survivors, key=lambda c: mining._portfolio_key(c, score_mode), reverse=True)[:n]  # noqa: SLF001

    for path in (
        USER_DATA / f"freqai_expressions_{tag}_diverse.json",
        USER_DATA / f"freqai_expressions_{tag}.json",
    ):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = payload.get("expressions", []) or []
        out: List[mining.CandidateRecord] = []
        for row in rows[:n]:
            expr = str(row.get("expression") or "").strip()
            if not expr:
                continue
            out.append(
                mining.CandidateRecord(
                    expression=expr,
                    origin=str(row.get("origin") or path.name),
                    train_ic=_safe_float(row.get("train_ic")),
                    oos_ic=_safe_float(row.get("oos_ic")),
                    sign_agree=int(row.get("sign_agree") or 0),
                    combined=_safe_float(row.get("combined")),
                    fitness=_safe_float(row.get("fitness")),
                    stability_ic=_safe_float(row.get("stability_ic")),
                    raw_ic=_safe_float(row.get("raw_ic")),
                    clean_ic=_safe_float(row.get("clean_ic")),
                    neutralized_ic=_safe_float(row.get("neutralized_ic")),
                    residual_ic_ratio=_safe_float(row.get("residual_ic_ratio")),
                    exposure_r2=_safe_float(row.get("exposure_r2")),
                    max_exposure_corr=_safe_float(row.get("max_exposure_corr")),
                    exposure_count=int(row.get("exposure_count") or 0),
                    purify_mode=str(row.get("purify_mode") or "off"),
                    eval_cache_key=str(row.get("eval_cache_key") or ""),
                )
            )
        return [mining.annotate_diversity(c) for c in out]
    raise FileNotFoundError(f"no mining state or exported library found for tag={tag}")


def _load_run_config(tag: str) -> Dict[str, Any]:
    path = LAB_STATE / "mining" / tag / "latest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    cfg = payload.get("config")
    return cfg if isinstance(cfg, dict) else {}


def _inherit_run_defaults(
    tag: str,
    *,
    timeframe: str,
    label_bars: Optional[int],
    label_mode: str,
    pair_reference: str,
    data_dir: Optional[str],
    data_venue: str,
    pairs: str,
) -> Tuple[str, Optional[int], str, str, Optional[str], str, str]:
    cfg = _load_run_config(tag)
    if not cfg:
        return timeframe, label_bars, label_mode, pair_reference, data_dir, data_venue, pairs
    if timeframe == "1h":
        timeframe = str(cfg.get("timeframe") or timeframe)
    if label_bars is None:
        raw_label = cfg.get("label_period")
        if raw_label in (None, ""):
            horizons = cfg.get("label_horizons")
            if isinstance(horizons, (list, tuple)) and horizons:
                raw_label = list(horizons)[0]
        try:
            label_bars = int(raw_label) if raw_label not in (None, "") else None
        except Exception:
            label_bars = None
    if label_mode == "forward_return":
        label_mode = str(cfg.get("label_mode") or label_mode)
    if pair_reference == "BTC/USDT":
        pair_reference = str(cfg.get("pair_reference") or pair_reference)
    if data_dir is None and cfg.get("data_dir") is not None:
        data_dir = str(cfg.get("data_dir"))
    if str(data_venue or "auto").lower() == "auto":
        data_venue = str(cfg.get("data_venue") or data_venue)
    if pairs == "auto":
        pairs = str(cfg.get("pairs") or pairs)
    return timeframe, label_bars, label_mode, pair_reference, data_dir, data_venue, pairs


def _inherit_cache_defaults(tag: str, cache_dir: Optional[str | Path], no_cache: bool) -> Tuple[Optional[str | Path], bool]:
    cfg = _load_run_config(tag)
    if not cfg:
        return cache_dir, no_cache
    if cache_dir is not None and str(cache_dir) == str(DEFAULT_CACHE_DIR) and cfg.get("cache_dir"):
        cache_dir = str(cfg.get("cache_dir"))
    if not no_cache:
        no_cache = bool(cfg.get("no_cache", no_cache))
    return cache_dir, no_cache


def _purify_config(
    *,
    mode: str,
    winsor: str,
    standardize: str,
    neutralize: str,
    exposures: str | Sequence[str],
    cache_dir: Optional[str | Path] = None,
    no_cache: bool = False,
    panel_fingerprint_hint: str = "",
) -> PurifyConfig:
    return PurifyConfig(
        mode=str(mode or "off").lower(),
        winsor=str(winsor or "mad").lower(),
        standardize=str(standardize or "zscore").lower(),
        neutralize=str(neutralize or "ridge").lower(),
        exposures=parse_exposure_groups(exposures),
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        no_cache=bool(no_cache),
        panel_fingerprint=panel_fingerprint_hint,
    )


def _factor_versions(
    big: pd.DataFrame,
    expr: str,
    cfg: mining.MiningConfig,
) -> Dict[str, Any]:
    return mining.factor_versions(big, expr, cfg)


def rank_ic_series(panel: pd.DataFrame, factor: pd.Series, *, min_pairs: int = 2) -> pd.Series:
    work = pd.DataFrame(
        {
            "date": panel["date"],
            "pair": panel["__pair__"],
            "factor": pd.Series(factor, index=panel.index, dtype="float64"),
            "fwd": panel["__fwd_ret__"],
        },
        index=panel.index,
    ).replace([np.inf, -np.inf], np.nan)
    factor_wide = work.pivot_table(index="date", columns="pair", values="factor", aggfunc="last")
    fwd_wide = work.pivot_table(index="date", columns="pair", values="fwd", aggfunc="last")
    if factor_wide.empty or fwd_wide.empty:
        return pd.Series(dtype="float64", name="rank_ic")
    factor_wide, fwd_wide = factor_wide.align(fwd_wide, join="inner", axis=0)
    factor_wide, fwd_wide = factor_wide.align(fwd_wide, join="inner", axis=1)
    valid = factor_wide.notna() & fwd_wide.notna()
    n = valid.sum(axis=1).astype(float)
    factor_rank = factor_wide.rank(axis=1, method="average").where(valid)
    fwd_rank = fwd_wide.rank(axis=1, method="average").where(valid)
    fx = factor_rank.sub(factor_rank.mean(axis=1), axis=0).where(valid)
    fy = fwd_rank.sub(fwd_rank.mean(axis=1), axis=0).where(valid)
    cov = (fx * fy).sum(axis=1)
    denom = np.sqrt((fx * fx).sum(axis=1) * (fy * fy).sum(axis=1))
    ic = cov / denom.replace(0.0, np.nan)
    ic = ic[(n >= int(min_pairs)) & np.isfinite(ic)]
    ic.name = "rank_ic"
    return ic.astype("float64")


def _long_short_weights(panel: pd.DataFrame, factor: pd.Series, *, quantile: float = 0.2) -> pd.Series:
    work = pd.DataFrame(
        {
            "date": panel["date"],
            "pair": panel["__pair__"],
            "factor": pd.Series(factor, index=panel.index, dtype="float64"),
            "fwd": panel["__fwd_ret__"],
        },
        index=panel.index,
    ).replace([np.inf, -np.inf], np.nan)
    weights = pd.Series(0.0, index=panel.index, dtype="float64")
    for _, idx in work.groupby("date", sort=False).groups.items():
        g = work.loc[idx].dropna(subset=["factor", "fwd"])
        if len(g) < 2:
            continue
        n_tail = max(1, int(np.floor(len(g) * float(quantile))))
        ordered = g.sort_values("factor")
        bottom = ordered.head(n_tail).index
        top = ordered.tail(n_tail).index
        if set(top) & set(bottom):
            continue
        weights.loc[top] = 1.0 / len(top)
        weights.loc[bottom] = -1.0 / len(bottom)
    return weights


def quantile_long_short(
    panel: pd.DataFrame,
    factor: pd.Series,
    *,
    quantile: float = 0.2,
) -> Tuple[pd.Series, float]:
    work = pd.DataFrame(
        {
            "date": panel["date"],
            "pair": panel["__pair__"],
            "factor": pd.Series(factor, index=panel.index, dtype="float64"),
            "fwd": panel["__fwd_ret__"],
        },
        index=panel.index,
    ).replace([np.inf, -np.inf], np.nan)
    factor_wide = work.pivot_table(index="date", columns="pair", values="factor", aggfunc="last")
    fwd_wide = work.pivot_table(index="date", columns="pair", values="fwd", aggfunc="last")
    if factor_wide.empty or fwd_wide.empty:
        return pd.Series(dtype="float64", name="long_short_return"), 0.0
    factor_wide, fwd_wide = factor_wide.align(fwd_wide, join="inner", axis=0)
    factor_wide, fwd_wide = factor_wide.align(fwd_wide, join="inner", axis=1)
    valid = factor_wide.notna() & fwd_wide.notna()
    n = valid.sum(axis=1)
    ranks = factor_wide.where(valid).rank(axis=1, method="first")
    tail = np.floor(n.astype(float) * float(quantile)).astype(int).clip(lower=1)
    bottom = ranks.le(tail, axis=0) & n.ge(2).to_numpy()[:, None]
    top = ranks.gt(n - tail, axis=0) & n.ge(2).to_numpy()[:, None]
    long_w = top.astype(float).div(top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    short_w = bottom.astype(float).div(bottom.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    wide_w = long_w - short_w
    ret = (wide_w * fwd_wide).sum(axis=1, min_count=1).rename("long_short_return")
    if len(wide_w) <= 1:
        turnover = 0.0
    else:
        turnover = float((wide_w.diff().abs().sum(axis=1) * 0.5).iloc[1:].mean())
        turnover = max(0.0, min(1.0, turnover))
    return ret.rename("long_short_return"), turnover


def factor_autocorr_decay(panel: pd.DataFrame, factor: pd.Series, lags: Sequence[int] = (1, 3, 6, 12, 24)) -> Dict[str, float]:
    work = pd.DataFrame(
        {"pair": panel["__pair__"], "factor": pd.Series(factor, index=panel.index, dtype="float64")},
        index=panel.index,
    ).replace([np.inf, -np.inf], np.nan)
    out: Dict[str, float] = {}
    for lag in lags:
        vals = []
        for _, g in work.groupby("pair", sort=False):
            s = g["factor"].dropna()
            if len(s) <= lag + 2 or s.std(ddof=0) <= 1e-12:
                continue
            c = s.autocorr(lag=lag)
            if np.isfinite(c):
                vals.append(float(c))
        out[f"autocorr_lag_{lag}"] = float(np.mean(vals)) if vals else 0.0
    return out


def factor_report(
    *,
    tag: str,
    n: int = 200,
    purify_mode: str = "off",
    purify_winsor: str = "mad",
    purify_standardize: str = "zscore",
    purify_neutralize: str = "ridge",
    purify_exposures: str = ",".join(DEFAULT_EXPOSURE_GROUPS),
    timeframe: str = "1h",
    label_bars: Optional[int] = None,
    label_mode: str = "forward_return",
    pair_reference: str = "BTC/USDT",
    data_dir: Optional[str] = None,
    data_venue: str = "auto",
    pairs: str = "auto",
    score_mode: str = "portfolio",
    cache_dir: Optional[str | Path] = DEFAULT_CACHE_DIR,
    no_cache: bool = False,
) -> Dict[str, Any]:
    candidates = _load_candidates(tag, n, score_mode=score_mode)
    timeframe, label_bars, label_mode, pair_reference, data_dir, data_venue, pairs = _inherit_run_defaults(
        tag,
        timeframe=timeframe,
        label_bars=label_bars,
        label_mode=label_mode,
        pair_reference=pair_reference,
        data_dir=data_dir,
        data_venue=data_venue,
        pairs=pairs,
    )
    cache_dir, no_cache = _inherit_cache_defaults(tag, cache_dir, no_cache)
    cfg = mining.MiningConfig(
        timeframe=timeframe,
        label_period=int(label_bars) if label_bars is not None else mining.DEFAULT_LABEL_PERIOD,
        label_mode=label_mode,
        pair_reference=pair_reference,
        data_dir=data_dir,
        data_venue=data_venue,
        pairs=pairs,
        purify_mode=purify_mode,
        purify_winsor=purify_winsor,
        purify_standardize=purify_standardize,
        purify_neutralize=purify_neutralize,
        purify_exposures=purify_exposures,
        cache_dir=str(cache_dir) if cache_dir is not None else "",
        no_cache=bool(no_cache),
    )
    big, _ = mining.build_big(
        timeframe=timeframe,
        label_bars=label_bars,
        label_mode=label_mode,
        pair_reference=pair_reference,
        data_dir=data_dir,
        data_venue=data_venue,
        pairs=pairs,
        cache_dir=cache_dir,
        no_cache=no_cache,
    )
    summary_rows: List[Dict[str, Any]] = []
    ts_rows: List[Dict[str, Any]] = []
    for i, cand in enumerate(candidates, start=1):
        name = f"f{i:03d}"
        versions = _factor_versions(big, cand.expression, cfg)
        selected = versions["selected"]
        ic = rank_ic_series(big, selected)
        raw_ic_s = rank_ic_series(big, versions["raw"])
        clean_ic_s = rank_ic_series(big, versions["clean"])
        neutral_ic_s = rank_ic_series(big, versions["neutralized"])
        rolling_ic = ic.rolling(24, min_periods=4).mean() if not ic.empty else ic
        ls_ret, to = quantile_long_short(big, selected)
        decay = factor_autocorr_decay(big, selected)
        diagnostics = versions.get("diagnostics", {}) or {}
        raw_ic = float(raw_ic_s.mean()) if not raw_ic_s.empty else 0.0
        clean_ic = float(clean_ic_s.mean()) if not clean_ic_s.empty else 0.0
        neutralized_ic = float(neutral_ic_s.mean()) if not neutral_ic_s.empty else 0.0

        icir = 0.0
        if not ic.empty and ic.std(ddof=0) > 1e-12:
            icir = float(ic.mean() / (ic.std(ddof=0) + 1e-12))
        spread_mean = float(ls_ret.dropna().mean()) if ls_ret.notna().any() else 0.0
        spread_ir = 0.0
        if ls_ret.dropna().std(ddof=0) > 1e-12:
            spread_ir = float(ls_ret.dropna().mean() / (ls_ret.dropna().std(ddof=0) + 1e-12))

        row = {
            "name": name,
            "expression": cand.expression,
            "origin": cand.origin,
            "rank_ic_mean": float(ic.mean()) if not ic.empty else 0.0,
            "rank_ic_ir": icir,
            "rolling_ic_last": float(rolling_ic.dropna().iloc[-1]) if rolling_ic.notna().any() else 0.0,
            "long_short_mean": spread_mean,
            "long_short_ir": spread_ir,
            "turnover": to,
            "raw_ic": raw_ic,
            "clean_ic": clean_ic,
            "neutralized_ic": neutralized_ic,
            "residual_ic_ratio": float(abs(neutralized_ic) / (abs(clean_ic) + 1e-9)),
            "exposure_r2": _safe_float(diagnostics.get("exposure_r2")),
            "max_exposure_corr": _safe_float(diagnostics.get("max_exposure_corr")),
            "purify_mode": purify_mode,
            **decay,
        }
        summary_rows.append(row)

        dates = sorted(set(ic.index).union(set(ls_ret.index)))
        for date in dates:
            ts_rows.append(
                {
                    "name": name,
                    "date": str(pd.Timestamp(date)),
                    "rank_ic": _safe_float(ic.get(date, np.nan), np.nan),
                    "rolling_ic_24": _safe_float(rolling_ic.get(date, np.nan), np.nan),
                    "long_short_return": _safe_float(ls_ret.get(date, np.nan), np.nan),
                }
            )

    out_dir = _report_dir(tag)
    summary = {
        "tag": tag,
        "generated_at": time.time(),
        "n_requested": int(n),
        "n_reported": len(summary_rows),
        "purify_mode": purify_mode,
        "timeframe": timeframe,
        "label_bars": label_bars,
        "label_mode": label_mode,
        "pair_reference": pair_reference,
        "data_venue": data_venue,
        "data_dir": data_dir,
        "pairs": pairs,
        "rows": summary_rows,
    }
    json_path = out_dir / "factor_report.json"
    csv_path = out_dir / "factor_report.csv"
    ts_path = out_dir / "factor_report_timeseries.csv"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    pd.DataFrame(ts_rows).to_csv(ts_path, index=False)
    html_path = _write_factor_html(out_dir, summary_rows, ts_rows)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "timeseries_csv": str(ts_path),
        "html": str(html_path) if html_path else None,
        "n_reported": len(summary_rows),
    }


def _write_factor_html(out_dir: Path, summary_rows: List[Dict[str, Any]], ts_rows: List[Dict[str, Any]]) -> Optional[Path]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return None
    if not summary_rows:
        return None
    df = pd.DataFrame(summary_rows)
    ts = pd.DataFrame(ts_rows)
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Rank IC Mean", "Long-Short Return"))
    fig.add_trace(go.Bar(x=df["name"], y=df["rank_ic_mean"], name="Rank IC"), row=1, col=1)
    if not ts.empty:
        for name, g in ts.groupby("name"):
            fig.add_trace(go.Scatter(x=g["date"], y=g["long_short_return"], mode="lines", name=name), row=2, col=1)
    fig.update_layout(height=820, title="FactorLab Factor Report")
    path = out_dir / "factor_report.html"
    fig.write_html(path)
    return path


def _empty_attribution_summary() -> Dict[str, float]:
    summary = {
        "mean_portfolio_return": 0.0,
        "mean_exposure_contribution": 0.0,
        "mean_residual_alpha": 0.0,
        "mean_abs_reconstruction_error": 0.0,
    }
    for group in DEFAULT_EXPOSURE_GROUPS:
        summary[f"mean_contrib_{group}"] = 0.0
    return summary


def _attribution_for_factor(
    panel: pd.DataFrame,
    factor: pd.Series,
    exposure_frame: pd.DataFrame,
    exposure_cols: Sequence[str],
    *,
    quantile: float = 0.2,
    ridge_alpha: float = 1e-3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    factor = pd.Series(factor, index=panel.index, dtype="float64").replace([np.inf, -np.inf], np.nan)
    group_sizes = panel.groupby("date", sort=False).size()
    if group_sizes.empty or int(group_sizes.max()) < 3:
        ls_ret, _ = quantile_long_short(panel, factor, quantile=quantile)
        rows = []
        for date, value in ls_ret.dropna().items():
            row: Dict[str, Any] = {
                "date": str(pd.Timestamp(date)),
                "portfolio_return": float(value),
                "exposure_contribution": 0.0,
                "residual_alpha": float(value),
                "reconstruction_error": 0.0,
            }
            for group in DEFAULT_EXPOSURE_GROUPS:
                row[f"contrib_{group}"] = 0.0
            rows.append(row)
        if not rows:
            return rows, _empty_attribution_summary()
        df = pd.DataFrame(rows)
        summary = {
            "mean_portfolio_return": float(df["portfolio_return"].mean()),
            "mean_exposure_contribution": 0.0,
            "mean_residual_alpha": float(df["residual_alpha"].mean()),
            "mean_abs_reconstruction_error": 0.0,
        }
        for group in DEFAULT_EXPOSURE_GROUPS:
            summary[f"mean_contrib_{group}"] = 0.0
        return rows, summary

    rows: List[Dict[str, Any]] = []
    for date, idx in panel.groupby("date", sort=False).groups.items():
        idx = pd.Index(idx)
        y = pd.Series(panel.loc[idx, "__fwd_ret__"], index=idx, dtype="float64")
        f = factor.loc[idx]
        x = exposure_frame.loc[idx, list(exposure_cols)] if exposure_cols else pd.DataFrame(index=idx)
        valid = f.notna() & y.notna() & np.isfinite(f) & np.isfinite(y)
        if x.shape[1]:
            valid &= x.notna().sum(axis=1) > 0
        g_idx = idx[valid.to_numpy()]
        if len(g_idx) < 3:
            continue
        f_valid = f.loc[g_idx]
        n_tail = max(1, int(np.floor(len(f_valid) * float(quantile))))
        ordered = f_valid.sort_values()
        bottom = ordered.head(n_tail).index
        top = ordered.tail(n_tail).index
        if set(top) & set(bottom):
            continue
        weights = pd.Series(0.0, index=g_idx, dtype="float64")
        weights.loc[top] = 1.0 / len(top)
        weights.loc[bottom] = -1.0 / len(bottom)
        yv = y.loc[g_idx]
        portfolio_return = float((weights * yv).sum())

        contributions: Dict[str, float] = {}
        if exposure_cols:
            xv = x.loc[g_idx].astype("float64")
            xv = xv.loc[:, xv.notna().sum(axis=0) >= 3]
            if not xv.empty:
                xv = xv.fillna(xv.median(axis=0))
                std = xv.std(axis=0, ddof=0)
                xv = xv.loc[:, std > 1e-12]
            if not xv.empty:
                xv = (xv - xv.mean(axis=0)) / (xv.std(axis=0, ddof=0) + 1e-9)
                max_cols = max(1, min(12, len(g_idx) - 2))
                if xv.shape[1] > max_cols:
                    scores = {
                        col: abs(float(xv[col].corr(yv))) if np.isfinite(xv[col].corr(yv)) else 0.0
                        for col in xv.columns
                    }
                    keep = [col for col, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:max_cols]]
                    xv = xv[keep]
                xmat = xv.to_numpy(dtype=float)
                yvec = (yv - yv.mean()).to_numpy(dtype=float)
                try:
                    xtx = xmat.T @ xmat
                    beta = np.linalg.solve(xtx + max(float(ridge_alpha), 1e-9) * np.eye(xtx.shape[0]), xmat.T @ yvec)
                    port_exposure = weights.loc[xv.index].to_numpy(dtype=float) @ xmat
                    for col, contrib in zip(xv.columns, port_exposure * beta):
                        cat = exposure_category(str(col))
                        contributions[cat] = contributions.get(cat, 0.0) + float(contrib)
                except np.linalg.LinAlgError:
                    contributions = {}

        total_contrib = float(sum(contributions.values()))
        residual = portfolio_return - total_contrib
        row: Dict[str, Any] = {
            "date": str(pd.Timestamp(date)),
            "portfolio_return": portfolio_return,
            "exposure_contribution": total_contrib,
            "residual_alpha": residual,
            "reconstruction_error": portfolio_return - total_contrib - residual,
        }
        for group in DEFAULT_EXPOSURE_GROUPS:
            row[f"contrib_{group}"] = float(contributions.get(group, 0.0))
        rows.append(row)

    if not rows:
        return rows, _empty_attribution_summary()
    df = pd.DataFrame(rows)
    summary = {
        "mean_portfolio_return": float(df["portfolio_return"].mean()),
        "mean_exposure_contribution": float(df["exposure_contribution"].mean()),
        "mean_residual_alpha": float(df["residual_alpha"].mean()),
        "mean_abs_reconstruction_error": float(df["reconstruction_error"].abs().mean()),
    }
    for group in DEFAULT_EXPOSURE_GROUPS:
        col = f"contrib_{group}"
        summary[f"mean_{col}"] = float(df[col].mean()) if col in df else 0.0
    return rows, summary


def _sample_attribution_dates(panel: pd.DataFrame, max_dates: int) -> List[Any]:
    dates = list(panel.groupby("date", sort=True).size().index)
    if max_dates <= 0 or len(dates) <= max_dates:
        return dates
    take = np.linspace(0, len(dates) - 1, int(max_dates), dtype=int)
    return [dates[int(i)] for i in take]


def _fast_exposure_columns(
    panel: pd.DataFrame,
    factor: pd.Series,
    exposure_frame: pd.DataFrame,
    exposure_cols: Sequence[str],
    sampled_dates: Sequence[Any],
    max_exposures: int,
) -> List[str]:
    cols = list(exposure_cols)
    if not cols or max_exposures <= 0:
        return []
    if len(cols) <= max_exposures:
        return cols
    mask = panel["date"].isin(sampled_dates).to_numpy()
    if int(mask.sum()) < 20:
        return cols[:max_exposures]
    y = pd.Series(panel.loc[mask, "__fwd_ret__"], index=panel.index[mask], dtype="float64")
    f = pd.Series(factor.loc[panel.index[mask]], index=panel.index[mask], dtype="float64")
    scores: Dict[str, float] = {}
    for col in cols:
        x = pd.Series(exposure_frame.loc[panel.index[mask], col], index=panel.index[mask], dtype="float64")
        valid = x.notna() & y.notna() & f.notna() & np.isfinite(x) & np.isfinite(y) & np.isfinite(f)
        if int(valid.sum()) < 20 or x.loc[valid].std(ddof=0) <= 1e-12:
            scores[col] = 0.0
            continue
        cy = x.loc[valid].corr(y.loc[valid])
        cf = x.loc[valid].corr(f.loc[valid])
        scores[col] = abs(float(cy)) + 0.5 * abs(float(cf)) if np.isfinite(cy) and np.isfinite(cf) else 0.0
    return [col for col, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:max_exposures]]


def _attribution_for_factor_fast(
    panel: pd.DataFrame,
    factor: pd.Series,
    exposure_frame: pd.DataFrame,
    exposure_cols: Sequence[str],
    *,
    quantile: float = 0.2,
    ridge_alpha: float = 1e-3,
    max_dates: int = 128,
    max_exposures: int = 12,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    factor = pd.Series(factor, index=panel.index, dtype="float64").replace([np.inf, -np.inf], np.nan)
    sampled_dates = _sample_attribution_dates(panel, max_dates=max_dates)
    selected_cols = _fast_exposure_columns(
        panel,
        factor,
        exposure_frame,
        exposure_cols,
        sampled_dates,
        max_exposures=max_exposures,
    )
    rows: List[Dict[str, Any]] = []
    groups = panel.groupby("date", sort=False).groups
    for date in sampled_dates:
        raw_idx = groups.get(date)
        if raw_idx is None:
            continue
        idx = pd.Index(raw_idx)
        y = pd.Series(panel.loc[idx, "__fwd_ret__"], index=idx, dtype="float64")
        f = factor.loc[idx]
        valid = f.notna() & y.notna() & np.isfinite(f) & np.isfinite(y)
        if selected_cols:
            x_all = exposure_frame.loc[idx, selected_cols]
            valid &= x_all.notna().sum(axis=1) > 0
        else:
            x_all = pd.DataFrame(index=idx)
        g_idx = idx[valid.to_numpy()]
        if len(g_idx) < 3:
            continue
        f_valid = f.loc[g_idx]
        n_tail = max(1, int(np.floor(len(f_valid) * float(quantile))))
        ordered = f_valid.sort_values()
        bottom = ordered.head(n_tail).index
        top = ordered.tail(n_tail).index
        if set(top) & set(bottom):
            continue
        weights = pd.Series(0.0, index=g_idx, dtype="float64")
        weights.loc[top] = 1.0 / len(top)
        weights.loc[bottom] = -1.0 / len(bottom)
        yv = y.loc[g_idx]
        portfolio_return = float((weights * yv).sum())

        contributions: Dict[str, float] = {}
        if selected_cols:
            xv = x_all.loc[g_idx, selected_cols].astype("float64")
            xv = xv.loc[:, xv.notna().sum(axis=0) >= 3]
            if not xv.empty:
                xv = xv.fillna(xv.median(axis=0))
                std = xv.std(axis=0, ddof=0)
                xv = xv.loc[:, std > 1e-12]
            if not xv.empty:
                xv = (xv - xv.mean(axis=0)) / (xv.std(axis=0, ddof=0) + 1e-9)
                xmat = xv.to_numpy(dtype=float)
                yvec = (yv.loc[xv.index] - yv.loc[xv.index].mean()).to_numpy(dtype=float)
                try:
                    xtx = xmat.T @ xmat
                    beta = np.linalg.solve(
                        xtx + max(float(ridge_alpha), 1e-9) * np.eye(xtx.shape[0]),
                        xmat.T @ yvec,
                    )
                    port_exposure = weights.loc[xv.index].to_numpy(dtype=float) @ xmat
                    for col, contrib in zip(xv.columns, port_exposure * beta):
                        cat = exposure_category(str(col))
                        contributions[cat] = contributions.get(cat, 0.0) + float(contrib)
                except np.linalg.LinAlgError:
                    contributions = {}

        total_contrib = float(sum(contributions.values()))
        residual = portfolio_return - total_contrib
        row: Dict[str, Any] = {
            "date": str(pd.Timestamp(date)),
            "portfolio_return": portfolio_return,
            "exposure_contribution": total_contrib,
            "residual_alpha": residual,
            "reconstruction_error": portfolio_return - total_contrib - residual,
        }
        for group in DEFAULT_EXPOSURE_GROUPS:
            row[f"contrib_{group}"] = float(contributions.get(group, 0.0))
        rows.append(row)

    if not rows:
        summary = _empty_attribution_summary()
    else:
        df = pd.DataFrame(rows)
        summary = {
            "mean_portfolio_return": float(df["portfolio_return"].mean()),
            "mean_exposure_contribution": float(df["exposure_contribution"].mean()),
            "mean_residual_alpha": float(df["residual_alpha"].mean()),
            "mean_abs_reconstruction_error": float(df["reconstruction_error"].abs().mean()),
        }
        for group in DEFAULT_EXPOSURE_GROUPS:
            col = f"contrib_{group}"
            summary[f"mean_{col}"] = float(df[col].mean()) if col in df else 0.0
    summary["sampled_dates"] = int(len(sampled_dates))
    summary["selected_exposures"] = int(len(selected_cols))
    return rows, summary


def exposure_report(
    *,
    tag: str,
    n: int = 200,
    purify_mode: str = "blend",
    purify_winsor: str = "mad",
    purify_standardize: str = "zscore",
    purify_neutralize: str = "ridge",
    purify_exposures: str = ",".join(DEFAULT_EXPOSURE_GROUPS),
    timeframe: str = "1h",
    label_bars: Optional[int] = None,
    label_mode: str = "forward_return",
    pair_reference: str = "BTC/USDT",
    data_dir: Optional[str] = None,
    data_venue: str = "auto",
    pairs: str = "auto",
    score_mode: str = "portfolio",
    cache_dir: Optional[str | Path] = DEFAULT_CACHE_DIR,
    no_cache: bool = False,
    attribution_mode: str = "fast",
    attribution_max_dates: int = 128,
    attribution_max_exposures: int = 12,
) -> Dict[str, Any]:
    candidates = _load_candidates(tag, n, score_mode=score_mode)
    timeframe, label_bars, label_mode, pair_reference, data_dir, data_venue, pairs = _inherit_run_defaults(
        tag,
        timeframe=timeframe,
        label_bars=label_bars,
        label_mode=label_mode,
        pair_reference=pair_reference,
        data_dir=data_dir,
        data_venue=data_venue,
        pairs=pairs,
    )
    cache_dir, no_cache = _inherit_cache_defaults(tag, cache_dir, no_cache)
    big, _ = mining.build_big(
        timeframe=timeframe,
        label_bars=label_bars,
        label_mode=label_mode,
        pair_reference=pair_reference,
        data_dir=data_dir,
        data_venue=data_venue,
        pairs=pairs,
        cache_dir=cache_dir,
        no_cache=no_cache,
    )
    cfg = mining.MiningConfig(
        timeframe=timeframe,
        label_period=int(label_bars) if label_bars is not None else mining.DEFAULT_LABEL_PERIOD,
        label_mode=label_mode,
        pair_reference=pair_reference,
        data_dir=data_dir,
        data_venue=data_venue,
        pairs=pairs,
        purify_mode=purify_mode,
        purify_winsor=purify_winsor,
        purify_standardize=purify_standardize,
        purify_neutralize=purify_neutralize,
        purify_exposures=purify_exposures,
        cache_dir=str(cache_dir) if cache_dir is not None else "",
        no_cache=bool(no_cache),
    )
    pcfg = _purify_config(
        mode=purify_mode,
        winsor=purify_winsor,
        standardize=purify_standardize,
        neutralize=purify_neutralize,
        exposures=purify_exposures,
        cache_dir=cache_dir,
        no_cache=no_cache,
        panel_fingerprint_hint=panel_fingerprint(big),
    )
    exposure_frame, exposure_cols = build_exposure_frame(
        big,
        pcfg.exposures,
        cache_dir=cache_dir,
        no_cache=no_cache,
        panel_fingerprint_hint=panel_fingerprint(big),
    )

    summary_rows: List[Dict[str, Any]] = []
    ts_rows: List[Dict[str, Any]] = []
    mode = str(attribution_mode or "fast").lower()
    if mode not in {"fast", "exact"}:
        raise ValueError(f"unknown attribution_mode: {attribution_mode}")
    for i, cand in enumerate(candidates, start=1):
        name = f"f{i:03d}"
        versions = _factor_versions(big, cand.expression, cfg)
        if mode == "exact":
            rows, summary = _attribution_for_factor(big, versions["selected"], exposure_frame, exposure_cols)
        else:
            rows, summary = _attribution_for_factor_fast(
                big,
                versions["selected"],
                exposure_frame,
                exposure_cols,
                max_dates=attribution_max_dates,
                max_exposures=attribution_max_exposures,
            )
        summary_rows.append(
            {
                "name": name,
                "expression": cand.expression,
                "origin": cand.origin,
                "exposure_count": len(exposure_cols),
                "attribution_mode": mode,
                "purify_mode": purify_mode,
                **summary,
            }
        )
        for row in rows:
            ts_rows.append({"name": name, **row})

    out_dir = _report_dir(tag)
    summary_payload = {
        "tag": tag,
        "generated_at": time.time(),
        "n_requested": int(n),
        "n_reported": len(summary_rows),
        "purify_mode": purify_mode,
        "timeframe": timeframe,
        "label_bars": label_bars,
        "label_mode": label_mode,
        "pair_reference": pair_reference,
        "data_venue": data_venue,
        "data_dir": data_dir,
        "pairs": pairs,
        "attribution_mode": mode,
        "attribution_max_dates": int(attribution_max_dates),
        "attribution_max_exposures": int(attribution_max_exposures),
        "exposure_columns": list(exposure_cols),
        "rows": summary_rows,
    }
    json_path = out_dir / "exposure_report.json"
    csv_path = out_dir / "exposure_report.csv"
    ts_path = out_dir / "exposure_report_timeseries.csv"
    json_path.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    pd.DataFrame(ts_rows).to_csv(ts_path, index=False)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "timeseries_csv": str(ts_path),
        "n_reported": len(summary_rows),
    }
