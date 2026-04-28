"""Composite fitness utilities — what replaces plain |IC| × sign_agree.

The goals:
    1. Penalize high turnover (naïve IC ignores trading cost).
    2. Penalize instability across rolling sub-periods (min-over-window).
    3. Optionally use cross-sectional rank IC instead of per-pair time-series IC.
    4. Keep everything numpy-only for speed (called inside tight loops).

Call-site API:
    timeseries_ic_mean(factor, fwd, pair_mask_by_pair, window_mask)
    cross_sectional_ic(factor_wide, fwd_wide, window_mask)
    turnover(series)
    composite_fitness(ics_by_window, turnover_, cfg)

Everything that does a "score" is a dict of floats — callers pick fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .paths import (DEFAULT_FUNDING_DAILY, DEFAULT_SLIPPAGE, DEFAULT_TAKER_FEE,
                    DEFAULT_VAL_WINDOWS)


# ============================================================
# Low-level IC primitives
# ============================================================

def _spearman(a: np.ndarray, b: np.ndarray, min_n: int = 200) -> float:
    """Spearman rank corr, NaN-safe. Returns 0.0 when too few obs or constant."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < min_n:
        return 0.0
    av, bv = a[mask], b[mask]
    if av.std() == 0 or bv.std() == 0:
        return 0.0
    ra = pd.Series(av).rank(method="average").values
    rb = pd.Series(bv).rank(method="average").values
    return float(np.corrcoef(ra, rb)[0, 1])


def timeseries_ic(factor: np.ndarray, fwd: np.ndarray,
                  mask: Optional[np.ndarray] = None,
                  min_n: int = 200) -> float:
    """Per-column Spearman IC on an optional window mask."""
    if mask is None:
        return _spearman(factor, fwd, min_n=min_n)
    return _spearman(factor[mask], fwd[mask], min_n=min_n)


def cross_sectional_ic(factor_wide: pd.DataFrame, fwd_wide: pd.DataFrame,
                        min_pairs: int = 3) -> float:
    """Mean XS rank IC: at each timestamp, rank factor values and forward returns
    across the asset axis, then take corr and average across time.

    Expected shape: rows = time, cols = pair names. NaN rows are dropped.
    """
    if factor_wide.empty or fwd_wide.empty:
        return 0.0
    df = factor_wide.align(fwd_wide, axis=0)
    f, y = df
    ics: List[float] = []
    for t in f.index:
        fv = f.loc[t].dropna()
        yv = y.loc[t].dropna()
        common = fv.index.intersection(yv.index)
        if len(common) < min_pairs:
            continue
        fr = fv.loc[common].rank()
        yr = yv.loc[common].rank()
        if fr.std() == 0 or yr.std() == 0:
            continue
        ics.append(float(np.corrcoef(fr, yr)[0, 1]))
    if not ics:
        return 0.0
    return float(np.mean(ics))


# ============================================================
# Turnover / trading-cost penalty
# ============================================================

def turnover(series: np.ndarray, skip_first: int = 100) -> float:
    """Normalized |Δseries| per step — proxy for how often a factor rotates.

    For a rank-transformed or z-scored factor, turnover ∈ [0, 1]; factors that
    flip sign every bar approach 1.0, slow-moving factors approach 0.
    """
    if len(series) <= skip_first:
        return 0.0
    s = series[skip_first:]
    finite = np.isfinite(s)
    if finite.sum() < 10:
        return 0.0
    s = s[finite]
    # Normalize to z-score first so the turnover is scale-invariant
    mu = s.mean(); sd = s.std()
    if sd == 0:
        return 0.0
    z = (s - mu) / sd
    diff = np.abs(np.diff(z))
    return float(diff.mean() / 2.0)  # diff mean / 2 ∈ roughly [0, 1]


def cost_multiplier(turnover_val: float, *,
                    fee: float = DEFAULT_TAKER_FEE,
                    slippage: float = DEFAULT_SLIPPAGE,
                    turnover_leverage: float = 50.0) -> float:
    """Return (1 - turnover × cost) clipped to [0, 1].

    turnover_leverage scales the raw turnover value into a per-bar trade rate
    estimate. 50x means turnover=0.02 → 1 trade per bar.
    """
    trade_rate = turnover_val * turnover_leverage
    cost = trade_rate * (fee + slippage)
    return float(max(0.0, min(1.0, 1.0 - cost)))


# ============================================================
# Multi-period aggregation (rolling windows inside VAL)
# ============================================================

def ics_by_windows(factor: np.ndarray, fwd: np.ndarray,
                   date_series: pd.Series,
                   windows: Sequence[Tuple[str, str]] = DEFAULT_VAL_WINDOWS,
                   min_n: int = 200) -> List[float]:
    """IC inside each sub-window. Returns list aligned with `windows`."""
    out: List[float] = []
    for start, end in windows:
        ts = pd.Timestamp(start, tz="UTC")
        te = pd.Timestamp(end, tz="UTC")
        m = (date_series >= ts) & (date_series < te)
        out.append(timeseries_ic(factor, fwd, m.values, min_n=min_n))
    return out


def aggregate_multi_period(ics: Sequence[float], mode: str = "min_abs") -> float:
    """Collapse per-window IC list into a single robustness score.

    mode:
      - "mean"    : plain mean
      - "min_abs" : min over |IC| (worst window matters most → anti-overfit)
      - "median"  : middle window
    """
    arr = np.asarray(list(ics), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if mode == "min_abs":
        # worst window in abs but preserve sign from mean direction
        sign = np.sign(arr.mean()) or 1.0
        return float(sign * np.min(np.abs(arr)))
    if mode == "median":
        return float(np.median(arr))
    return float(arr.mean())


# ============================================================
# Composite fitness
# ============================================================

@dataclass
class FitnessConfig:
    fee: float = DEFAULT_TAKER_FEE
    slippage: float = DEFAULT_SLIPPAGE
    turnover_weight: float = 1.0          # 0 = disable cost penalty
    stability_mode: str = "min_abs"       # how to collapse multi-period IC
    xs_weight: float = 0.0                # 0 = pure TS, 1.0 = pure XS
    decay_weight: float = 0.0             # reward when OOS IC ≥ train IC


def composite_fitness(*, ts_ics: Sequence[float], xs_ic: float, turnover_val: float,
                      sign_agree: int, n_pairs: int,
                      train_ic: Optional[float] = None,
                      cfg: FitnessConfig) -> Dict[str, float]:
    """Compose one scalar fitness + breakdown. All ingredients must be precomputed."""
    ts_agg = aggregate_multi_period(ts_ics, cfg.stability_mode)
    combined_ic = (1.0 - cfg.xs_weight) * ts_agg + cfg.xs_weight * xs_ic
    cost_mult = cost_multiplier(turnover_val, fee=cfg.fee, slippage=cfg.slippage) \
                if cfg.turnover_weight > 0 else 1.0
    # Cost scaled by its weight
    cost_mult = 1.0 - cfg.turnover_weight * (1.0 - cost_mult)
    cost_mult = max(0.0, min(1.0, cost_mult))
    sign_factor = (sign_agree / max(n_pairs, 1)) if n_pairs > 0 else 0.0
    decay_factor = 1.0
    if cfg.decay_weight > 0 and train_ic is not None and train_ic != 0:
        # ratio in [0, 1.5]; reward when OOS IC maintains strength
        ratio = min(abs(combined_ic) / (abs(train_ic) + 1e-9), 1.5)
        decay_factor = 1.0 + cfg.decay_weight * (ratio - 1.0)
        decay_factor = max(0.1, decay_factor)
    fitness = abs(combined_ic) * sign_factor * cost_mult * decay_factor
    return {
        "fitness": float(fitness),
        "combined_ic": float(combined_ic),
        "ts_agg": float(ts_agg),
        "xs_ic": float(xs_ic),
        "turnover": float(turnover_val),
        "cost_mult": float(cost_mult),
        "sign_agree": int(sign_agree),
        "n_pairs": int(n_pairs),
        "decay_factor": float(decay_factor),
    }
