"""Feature engineering (mtf4h, cross-sectional, pair-relative, funding, microstructure).

All merged causally (no lookahead) onto the 1h feather as extra columns.
Each merger backs up original feather once at *.pre_<kind>.bak.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .paths import KUCOIN_DIR, FUNDING_DIR, resolve_pairs

# Import existing feature engine
import sys
_SRC = str(Path(__file__).resolve().parents[2])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from agent_market.freqai.features import apply_configured_features


# ============================================================
# MTF 4h features
# ============================================================

MTF_CFG = {"features": [
    {"name": "rsi_14",           "type": "rsi",       "period": 14},
    {"name": "adx_14",           "type": "adx",       "period": 14},
    {"name": "atr_norm_14",      "type": "atr_norm",  "period": 14},
    {"name": "ema_pct_12",       "type": "ema_pct",   "period": 12},
    {"name": "ema_pct_48",       "type": "ema_pct",   "period": 48},
    {"name": "cmf_20",           "type": "cmf",       "period": 20},
    {"name": "plus_di_14",       "type": "plus_di",   "period": 14},
    {"name": "minus_di_14",      "type": "minus_di",  "period": 14},
    {"name": "return_zscore_24", "type": "return_zscore", "period": 24},
    {"name": "realized_vol_24",  "type": "realized_vol",  "period": 24},
    {"name": "return_skew_48",   "type": "return_skew",   "period": 48},
    {"name": "donchian_width_48","type": "donchian_width","period": 48},
]}


def _data_root(data_dir: Path | str | None = None) -> Path:
    return Path(data_dir) if data_dir is not None else KUCOIN_DIR


def merge_mtf4h(pairs: Sequence[str] | str = None, *, data_dir: Path | str | None = None) -> dict:
    """Merge 4h-derived features onto 1h feathers (prefix: mtf4h_)."""
    root = _data_root(data_dir)
    pairs = resolve_pairs(pairs, data_dir=root, timeframe="1h")
    results = {}
    for pair in pairs:
        sanitized = pair.replace("/", "_")
        p1 = root / f"{sanitized}-1h.feather"
        p4 = root / f"{sanitized}-4h.feather"
        if not p1.exists() or not p4.exists():
            results[pair] = "missing"; continue

        df1 = pd.read_feather(p1)
        df1["date"] = pd.to_datetime(df1["date"], utc=True)
        df4 = pd.read_feather(p4)
        df4["date"] = pd.to_datetime(df4["date"], utc=True)
        df4 = df4.reset_index(drop=True)

        df4 = apply_configured_features(df4, MTF_CFG)
        feat_cols = [f["name"] for f in MTF_CFG["features"]]
        df4["__close_time__"] = df4["date"] + pd.Timedelta(hours=4)
        rename = {c: f"mtf4h_{c}" for c in feat_cols}
        mtf_df = (df4[["__close_time__"] + feat_cols].rename(columns=rename)
                  .rename(columns={"__close_time__": "date"}).reset_index(drop=True))

        drop = [c for c in mtf_df.columns if c in df1.columns and c.startswith("mtf4h_")]
        if drop: df1 = df1.drop(columns=drop)
        merged = pd.merge_asof(df1, mtf_df, on="date", direction="backward")

        bak = p1.with_suffix(".feather.pre_mtf.bak")
        if not bak.exists(): df1.to_feather(bak)
        merged.to_feather(p1)
        results[pair] = f"+{sum(1 for c in merged.columns if c.startswith('mtf4h_'))} cols"
    return results


# ============================================================
# Cross-sectional ranks
# ============================================================

XS_COLS = ["xs_ret_24h", "xs_ret_72h", "xs_vol_24h", "xs_volume_24h",
           "xs_momentum_168h", "xs_ret_vs_btc", "xs_mtf_rsi_rank"]


def merge_cross_sectional(pairs: Sequence[str] | str = None, *, data_dir: Path | str | None = None) -> dict:
    """Add 7 cross-sectional rank features (0..1 percentile)."""
    root = _data_root(data_dir)
    pairs = resolve_pairs(pairs, data_dir=root, timeframe="1h")
    frames = []
    for pair in pairs:
        p = root / f"{pair.replace('/', '_')}-1h.feather"
        if not p.exists(): continue
        df = pd.read_feather(p)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["__pair__"] = pair
        df["__ret_24h__"] = df["close"].pct_change(24)
        df["__ret_72h__"] = df["close"].pct_change(72)
        df["__vol_24h__"] = df["close"].pct_change().rolling(24).std()
        df["__volume_24h__"] = df["volume"].rolling(24).sum()
        df["__momentum_168h__"] = (df["close"] / df["close"].shift(168)) - 1
        frames.append(df)

    if not frames: return {}
    big = pd.concat(frames, ignore_index=True)
    keep = ["date","__pair__","__ret_24h__","__ret_72h__","__vol_24h__",
            "__volume_24h__","__momentum_168h__"]
    if "mtf4h_rsi_14" in big.columns: keep.append("mtf4h_rsi_14")
    big = big[keep].copy()

    def rank_xs(col): return big.groupby("date")[col].rank(pct=True, method="average")
    big["xs_ret_24h"]       = rank_xs("__ret_24h__")
    big["xs_ret_72h"]       = rank_xs("__ret_72h__")
    big["xs_vol_24h"]       = rank_xs("__vol_24h__")
    big["xs_volume_24h"]    = rank_xs("__volume_24h__")
    big["xs_momentum_168h"] = rank_xs("__momentum_168h__")
    if "mtf4h_rsi_14" in big.columns:
        big["xs_mtf_rsi_rank"] = rank_xs("mtf4h_rsi_14")
    else:
        big["xs_mtf_rsi_rank"] = 0.5

    btc = big[big["__pair__"] == "BTC/USDT"][["date","__ret_24h__"]].rename(columns={"__ret_24h__": "__btc__"})
    big = big.merge(btc, on="date", how="left")
    std = big.groupby("date")["__ret_24h__"].transform("std")
    big["xs_ret_vs_btc"] = (big["__ret_24h__"] - big["__btc__"]) / (std + 1e-9)
    for c in XS_COLS: big[c] = big[c].fillna(0.5)

    results = {}
    for pair in pairs:
        p = root / f"{pair.replace('/', '_')}-1h.feather"
        if not p.exists(): continue
        df = pd.read_feather(p)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        drop = [c for c in XS_COLS if c in df.columns]
        if drop: df = df.drop(columns=drop)
        sub = big[big["__pair__"] == pair][["date"] + XS_COLS]
        merged = df.merge(sub, on="date", how="left")
        bak = p.with_suffix(".feather.pre_xs.bak")
        if not bak.exists(): df.to_feather(bak)
        merged.to_feather(p)
        results[pair] = f"+{len(XS_COLS)} cols"
    return results


# ============================================================
# Pair-relative features (vs reference pair, default BTC/USDT)
# ============================================================

def _reference_suffix(reference_pair: str) -> str:
    return str(reference_pair).split("/")[0].strip().lower().replace("-", "_")


def pair_cols_for_reference(reference_pair: str) -> list[str]:
    suffix = _reference_suffix(reference_pair)
    return [
        f"pair_ret_1_vs_{suffix}",
        f"pair_ret_24h_vs_{suffix}",
        f"pair_ret_72h_vs_{suffix}",
        f"pair_log_ratio_{suffix}",
        f"pair_ratio_z_24_{suffix}",
        f"pair_ratio_z_72_{suffix}",
        f"pair_beta_72_{suffix}",
        f"pair_resid_ret_1_{suffix}",
        f"pair_resid_z_24_{suffix}",
    ]


PAIR_COLS = pair_cols_for_reference("BTC/USDT")


def merge_pair_relative(
    pairs: Sequence[str] | str = None,
    *,
    reference_pair: str = "BTC/USDT",
    reference_pairs: Sequence[str] | str | None = None,
    beta_window: int = 72,
    data_dir: Path | str | None = None,
) -> dict:
    """Add pair-relative features versus one or more reference pairs.

    All features are causal (rolling, no lookahead) and merged onto each pair's
    1h feather.
    """
    root = _data_root(data_dir)
    pairs = resolve_pairs(pairs, data_dir=root, timeframe="1h")
    if reference_pairs is None:
        refs = [reference_pair]
    elif isinstance(reference_pairs, str):
        refs = [part.strip() for part in reference_pairs.split(",") if part.strip()]
    else:
        refs = [str(part).strip() for part in reference_pairs if str(part).strip()]
    refs = list(dict.fromkeys(refs))
    frames = []
    results: dict = {}
    for pair in pairs:
        p = root / f"{pair.replace('/', '_')}-1h.feather"
        if not p.exists():
            continue
        df = pd.read_feather(p)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)
        df = df.drop_duplicates("date", keep="last").reset_index(drop=True)
        df["__pair__"] = pair
        df["__ret_1__"] = df["close"].pct_change(1)
        df["__ret_24h__"] = df["close"].pct_change(24)
        df["__ret_72h__"] = df["close"].pct_change(72)
        df["__log_close__"] = np.log(df["close"].clip(lower=1e-12))
        frames.append(df[["date", "__pair__", "__ret_1__", "__ret_24h__", "__ret_72h__", "__log_close__"]])

    if not frames:
        return {pair: "missing" for pair in pairs}

    big = pd.concat(frames, ignore_index=True)
    big = big.sort_values(["__pair__", "date"]).reset_index(drop=True)

    def _rolling_by_pair(frame: pd.DataFrame, col: str, window: int, min_periods: int, stat: str) -> np.ndarray:
        out = np.full(len(frame), np.nan, dtype=float)
        for _, idx in frame.groupby("__pair__", sort=False).groups.items():
            s = frame.loc[idx, col]
            roll = s.rolling(window, min_periods=min_periods)
            vals = roll.std(ddof=0) if stat == "std" else roll.mean()
            out[np.asarray(idx, dtype=int)] = vals.to_numpy(dtype=float)
        return out

    all_pair_cols: list[str] = []
    missing_refs: list[str] = []
    for ref_pair in refs:
        ref = big.loc[big["__pair__"] == ref_pair, ["date", "__ret_1__", "__ret_24h__", "__ret_72h__", "__log_close__"]]
        if ref.empty:
            missing_refs.append(ref_pair)
            continue
        ref = ref.drop_duplicates("date", keep="last")
        ref = ref.rename(
            columns={
                "__ret_1__": "__ref_ret_1__",
                "__ret_24h__": "__ref_ret_24h__",
                "__ret_72h__": "__ref_ret_72h__",
                "__log_close__": "__ref_log_close__",
            }
        )
        work = big.merge(ref, on="date", how="left")
        cols = pair_cols_for_reference(ref_pair)
        all_pair_cols.extend(cols)

        work[cols[0]] = work["__ret_1__"] - work["__ref_ret_1__"]
        work[cols[1]] = work["__ret_24h__"] - work["__ref_ret_24h__"]
        work[cols[2]] = work["__ret_72h__"] - work["__ref_ret_72h__"]
        work[cols[3]] = work["__log_close__"] - work["__ref_log_close__"]

        mu_24 = _rolling_by_pair(work, cols[3], 24, 12, "mean")
        sd_24 = _rolling_by_pair(work, cols[3], 24, 12, "std")
        mu_72 = _rolling_by_pair(work, cols[3], 72, 24, "mean")
        sd_72 = _rolling_by_pair(work, cols[3], 72, 24, "std")
        work[cols[4]] = (work[cols[3]] - mu_24) / (sd_24 + 1e-9)
        work[cols[5]] = (work[cols[3]] - mu_72) / (sd_72 + 1e-9)

        beta_window = max(24, int(beta_window))
        min_periods = max(12, beta_window // 3)
        beta = np.full(len(work), np.nan, dtype=float)
        for _, idx in work.groupby("__pair__", sort=False).groups.items():
            sub = work.loc[idx]
            cov = sub["__ret_1__"].rolling(beta_window, min_periods=min_periods).cov(sub["__ref_ret_1__"])
            var = sub["__ref_ret_1__"].rolling(beta_window, min_periods=min_periods).var(ddof=0)
            beta[np.asarray(idx, dtype=int)] = (cov / (var + 1e-9)).to_numpy(dtype=float)
        work[cols[6]] = beta
        work[cols[7]] = work["__ret_1__"] - work[cols[6]] * work["__ref_ret_1__"]
        resid_mu = _rolling_by_pair(work, cols[7], 24, 12, "mean")
        resid_sd = _rolling_by_pair(work, cols[7], 24, 12, "std")
        work[cols[8]] = (work[cols[7]] - resid_mu) / (resid_sd + 1e-9)

        ref_mask = work["__pair__"] == ref_pair
        for c in cols:
            work.loc[ref_mask, c] = 0.0
            big[c] = work[c].replace([np.inf, -np.inf], np.nan)

    if not all_pair_cols:
        return {pair: f"missing_reference:{','.join(missing_refs) or ','.join(refs)}" for pair in pairs}
    all_pair_cols = list(dict.fromkeys(all_pair_cols))

    for pair in pairs:
        p = root / f"{pair.replace('/', '_')}-1h.feather"
        if not p.exists():
            results[pair] = "missing"
            continue
        df = pd.read_feather(p)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        drop = [c for c in all_pair_cols if c in df.columns]
        if drop:
            df = df.drop(columns=drop)
        sub = big.loc[big["__pair__"] == pair, ["date"] + all_pair_cols]
        merged = df.merge(sub, on="date", how="left")
        for c in all_pair_cols:
            merged[c] = merged[c].fillna(0.0)

        bak = p.with_suffix(".feather.pre_pair.bak")
        if not bak.exists():
            df.to_feather(bak)
        merged.to_feather(p)
        suffix = f" refs={','.join(refs)}"
        if missing_refs:
            suffix += f" missing_refs={','.join(missing_refs)}"
        results[pair] = f"+{len(all_pair_cols)} cols{suffix}"
    return results


# ============================================================
# Funding rate features
# ============================================================

FUND_COLS = ["funding_rate", "funding_abs_ma24", "funding_cumsum_72",
             "funding_shift_8h", "funding_z_200"]


def merge_funding(pairs: Sequence[str] | str = None, *, data_dir: Path | str | None = None) -> dict:
    """Add 5 funding-rate-derived features (causal merge_asof backward)."""
    root = _data_root(data_dir)
    pairs = resolve_pairs(pairs, data_dir=root, timeframe="1h")
    results = {}
    for pair in pairs:
        sym = pair.split("/")[0]
        p1 = root / f"{pair.replace('/', '_')}-1h.feather"
        pf = FUNDING_DIR / f"{sym}_USDT-funding.feather"
        if not p1.exists() or not pf.exists():
            results[pair] = "missing"; continue

        df1 = pd.read_feather(p1)
        df1["date"] = pd.to_datetime(df1["date"], utc=True)
        fund = pd.read_feather(pf)
        fund["date"] = pd.to_datetime(fund["date"], utc=True)
        fund = fund.reset_index(drop=True)

        fund["abs_rate"] = fund["funding_rate"].abs()
        fund["funding_abs_ma24"] = fund["abs_rate"].rolling(3, min_periods=1).mean()
        fund["funding_cumsum_72"] = fund["funding_rate"].rolling(9, min_periods=1).sum()
        fund["funding_shift_8h"] = fund["funding_rate"].diff(1)
        mu = fund["funding_rate"].rolling(25, min_periods=3).mean()
        sd = fund["funding_rate"].rolling(25, min_periods=3).std(ddof=0)
        fund["funding_z_200"] = (fund["funding_rate"] - mu) / (sd + 1e-9)

        drop = [c for c in FUND_COLS if c in df1.columns]
        if drop: df1 = df1.drop(columns=drop)
        fund_small = fund[["date"] + FUND_COLS].reset_index(drop=True)
        merged = pd.merge_asof(df1, fund_small, on="date", direction="backward")
        for c in FUND_COLS: merged[c] = merged[c].fillna(0)
        bak = p1.with_suffix(".feather.pre_funding.bak")
        if not bak.exists(): df1.to_feather(bak)
        merged.to_feather(p1)
        results[pair] = f"+{len(FUND_COLS)} cols"
    return results


# ============================================================
# Microstructure features (from 1m bars)
# ============================================================

MICRO_COLS = [
    "micro_realized_vol", "micro_range_ratio", "micro_vol_burst",
    "micro_dollar_vol", "micro_up_share", "micro_first_move",
    "micro_last_move", "micro_autocorr_lag1", "micro_zscore_extreme",
    "micro_volume_slope",
]


def _micro_agg_hour(g: pd.DataFrame) -> pd.Series:
    if len(g) < 10: return pd.Series({c: np.nan for c in MICRO_COLS})
    rv = float(g["log_ret"].std(ddof=0))
    h_high = float(g["high"].max()); h_low = float(g["low"].min())
    sum_abs = float(g["bar_abs_move"].sum())
    rng_ratio = (h_high - h_low) / (sum_abs + 1e-9)
    max_v = float(g["volume"].max()); mean_v = float(g["volume"].mean())
    vol_burst = max_v / (mean_v + 1e-9)
    dv = float(g["dollar_vol"].sum())
    up_share = float(g["is_up"].mean())
    first_ret = float(g["log_ret"].iloc[0]); last_ret = float(g["log_ret"].iloc[-1])
    first_m = first_ret / (rv + 1e-9); last_m = last_ret / (rv + 1e-9)
    if len(g) >= 3:
        s = g["log_ret"].values; demean = s - s.mean()
        var = (demean * demean).sum()
        ac = float(((demean[:-1] * demean[1:]).sum()) / (var + 1e-9))
    else: ac = 0.0
    extreme = float((g["log_ret"].abs() > 2 * rv).mean()) if rv > 0 else 0
    cumv = g["volume"].cumsum().values
    if len(cumv) >= 3 and cumv[-1] > 0:
        x = np.arange(len(cumv)) / (len(cumv) - 1)
        y = cumv / cumv[-1]
        slope = float(np.polyfit(x, y, 1)[0])
    else: slope = 1.0
    return pd.Series({
        "micro_realized_vol": rv, "micro_range_ratio": rng_ratio,
        "micro_vol_burst": vol_burst, "micro_dollar_vol": dv,
        "micro_up_share": up_share, "micro_first_move": first_m,
        "micro_last_move": last_m, "micro_autocorr_lag1": ac,
        "micro_zscore_extreme": extreme, "micro_volume_slope": slope,
    })


def merge_micro(pairs: Sequence[str] | str = None, *, data_dir: Path | str | None = None) -> dict:
    """Aggregate 1m bars → 1h micro_* features."""
    root = _data_root(data_dir)
    pairs = resolve_pairs(pairs, data_dir=root, timeframe="1h")
    results = {}
    for pair in pairs:
        sanitized = pair.replace("/", "_")
        p1 = root / f"{sanitized}-1h.feather"
        pm = root / f"{sanitized}-1m.feather"
        if not p1.exists() or not pm.exists():
            results[pair] = "missing"; continue

        df1m = pd.read_feather(pm)
        df1m["date"] = pd.to_datetime(df1m["date"], utc=True)
        df1m = df1m.reset_index(drop=True)
        df1m["log_ret"] = np.log(df1m["close"] / df1m["close"].shift(1)).fillna(0)
        df1m["dollar_vol"] = df1m["close"] * df1m["volume"]
        df1m["is_up"] = (df1m["close"] > df1m["open"]).astype(float)
        df1m["bar_abs_move"] = (df1m["close"] - df1m["open"]).abs()
        hour = df1m["date"].dt.floor("h")
        hour.name = "date"
        hourly = df1m.groupby(hour, group_keys=False).apply(_micro_agg_hour).reset_index()

        df1h = pd.read_feather(p1)
        df1h["date"] = pd.to_datetime(df1h["date"], utc=True)
        drop = [c for c in MICRO_COLS if c in df1h.columns]
        if drop: df1h = df1h.drop(columns=drop)
        merged = df1h.merge(hourly, on="date", how="left")
        for c in MICRO_COLS:
            merged[c] = merged[c].fillna(merged[c].median() if merged[c].notna().any() else 0)

        bak = p1.with_suffix(".feather.pre_micro.bak")
        if not bak.exists(): df1h.to_feather(bak)
        merged.to_feather(p1)
        results[pair] = f"+{len(MICRO_COLS)} cols ({hourly['micro_realized_vol'].notna().sum():,} hours)"
    return results


# ============================================================
# OHLCV micro_feature library (directly on 1h feather)
# ============================================================

OHLCV_MICRO_COLS = [
    "ret_1",
    "logret_1",
    "range_pct",
    "body_pct",
    "wick_up_pct",
    "wick_down_pct",
    "rv_12",
    "rv_24",
    "rv_72",
    "vol_z_12",
    "vol_z_24",
    "vol_z_72",
    "amihud_12",
    "amihud_24",
    "amihud_72",
]


def merge_ohlcv_micro(pairs: Sequence[str] | str = None, *, data_dir: Path | str | None = None) -> dict:
    """Compute and merge OHLCV micro features onto 1h feathers.

    This materializes the same feature names produced by
    `agent_market.microstructure.ohlcv_features.build_ohlcv_micro_features`
    (ret_1/logret_1/rv_*/vol_z_*/amihud_* etc) directly into each
    `*-1h.feather` so factor miners that load feathers can use them.
    """
    root = _data_root(data_dir)
    pairs = resolve_pairs(pairs, data_dir=root, timeframe="1h")
    results: dict = {}
    for pair in pairs:
        sanitized = pair.replace("/", "_")
        p1 = root / f"{sanitized}-1h.feather"
        if not p1.exists():
            results[pair] = "missing"
            continue

        df = pd.read_feather(p1)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            results[pair] = f"missing_cols={sorted(required - set(df.columns))}"
            continue

        close = df["close"].astype("float64")
        open_ = df["open"].astype("float64")
        high = df["high"].astype("float64")
        low = df["low"].astype("float64")
        volume = df["volume"].astype("float64")

        eps = 1e-12
        ret_1 = close.pct_change()
        logret_1 = np.log(close + eps).diff()

        out = pd.DataFrame(
            {
                "date": df["date"],
                "ret_1": ret_1,
                "logret_1": logret_1,
                "range_pct": (high - low) / (close + eps),
                "body_pct": (close - open_) / (open_ + eps),
                "wick_up_pct": (high - np.maximum(open_, close)) / (close + eps),
                "wick_down_pct": (np.minimum(open_, close) - low) / (close + eps),
            }
        )
        for w in (12, 24, 72):
            out[f"rv_{w}"] = logret_1.rolling(int(w)).std(ddof=0)
            out[f"vol_z_{w}"] = (volume - volume.rolling(int(w)).mean()) / (
                volume.rolling(int(w)).std(ddof=0) + eps
            )
            out[f"amihud_{w}"] = (ret_1.abs() / (volume + eps)).rolling(int(w)).mean()

        # Drop existing cols if present (idempotent re-run).
        drop = [c for c in OHLCV_MICRO_COLS if c in df.columns]
        if drop:
            df = df.drop(columns=drop)
        merged = df.merge(out, on="date", how="left")
        merged = merged.replace([np.inf, -np.inf], np.nan)

        bak = p1.with_suffix(".feather.pre_ohlcv_micro.bak")
        if not bak.exists():
            df.to_feather(bak)
        merged.to_feather(p1)
        results[pair] = f"+{len(OHLCV_MICRO_COLS)} cols"
    return results


# ============================================================
# Restore backups
# ============================================================

def restore_backups(kind: str, pairs: Sequence[str] | str = None, *, data_dir: Path | str | None = None):
    """Restore feathers from a backup snapshot (kind: mtf|xs|pair|funding|micro|ohlcv_micro|microstructure)."""
    root = _data_root(data_dir)
    pairs = resolve_pairs(pairs, data_dir=root, timeframe="1h")
    suffix = f".feather.pre_{kind}.bak"
    for pair in pairs:
        p = root / f"{pair.replace('/', '_')}-1h.feather"
        bak = p.with_suffix(suffix)
        if bak.exists():
            import shutil
            shutil.copy(bak, p)
            print(f"  restored {pair} from {suffix}")


# ============================================================
# Microstructure features (LOB+trades parquet → 1h feather)
# ============================================================

def merge_microstructure_parquet(
    *,
    features_parquet: Path,
    target_feather: Path,
    ts_col: str = "ts",
    target_date_col: str = "date",
    symbol: Optional[str] = None,
    agg: str = "mean",
    prefix: str = "",
    backup_suffix: str = ".feather.pre_microstructure.bak",
) -> dict:
    """Merge microstructure (LOB+trades) features.parquet into a 1h OHLCV feather.

    - Aggregates microstructure rows to 1h buckets (mean/median) and merges via backward asof.
    - Keeps only numeric columns (drops identifier fields like ts/symbol/event).
    - By default, merges without prefix so downstream auto-discovery can pick up
      `mid/spread/... depth_* ofi_* l2_ofi_*` column patterns.
    """
    features_parquet = Path(features_parquet).resolve()
    target_feather = Path(target_feather).resolve()

    if not features_parquet.exists():
        raise FileNotFoundError(f"features parquet not found: {features_parquet}")
    if not target_feather.exists():
        raise FileNotFoundError(f"target feather not found: {target_feather}")

    mf = pd.read_parquet(features_parquet)
    if ts_col not in mf.columns:
        if "date" in mf.columns:
            ts_col = "date"
        else:
            raise ValueError(f"microstructure parquet missing timestamp column: {ts_col!r}")

    mf[ts_col] = pd.to_datetime(mf[ts_col], utc=True, errors="coerce").dt.tz_convert(None)
    mf = mf.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    if symbol and "symbol" in mf.columns:
        mf = mf.loc[mf["symbol"].astype(str) == str(symbol)]

    # Avoid poisoning the asof-merge with epoch snapshot rows (often ts=1970-01-01).
    if "event" in mf.columns:
        mf = mf.loc[mf["event"].astype(str).str.lower() != "snapshot"].reset_index(drop=True)

    drop_cols = {ts_col, "symbol", "event", "pair", "date"}
    num_cols = [
        c for c in mf.columns if c not in drop_cols and np.issubdtype(mf[c].dtype, np.number)
    ]
    if not num_cols:
        raise ValueError("no numeric feature columns found in microstructure parquet")

    mf_hour = mf.copy()
    mf_hour["__hour__"] = mf_hour[ts_col].dt.floor("1h")
    if str(agg).strip().lower() == "median":
        hourly = mf_hour.groupby("__hour__", sort=True)[num_cols].median().reset_index()
    else:
        hourly = mf_hour.groupby("__hour__", sort=True)[num_cols].mean().reset_index()
    hourly = hourly.rename(columns={"__hour__": target_date_col})

    if prefix:
        rename = {c: f"{prefix}{c}" for c in num_cols}
        hourly = hourly.rename(columns=rename)
        out_cols = [rename[c] for c in num_cols]
    else:
        out_cols = list(num_cols)

    df = pd.read_feather(target_feather)
    if target_date_col not in df.columns:
        raise ValueError(f"target feather missing date col: {target_date_col!r}")
    df[target_date_col] = pd.to_datetime(df[target_date_col], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.sort_values(target_date_col).reset_index(drop=True)

    drop_existing = [c for c in out_cols if c in df.columns]
    if drop_existing:
        df = df.drop(columns=drop_existing)

    merged = pd.merge_asof(df, hourly, on=target_date_col, direction="backward")
    merged = merged.replace([np.inf, -np.inf], np.nan)

    bak = target_feather.with_suffix(backup_suffix)
    if not bak.exists():
        df.to_feather(bak)
    merged.to_feather(target_feather)

    return {
        "target": str(target_feather),
        "backup": str(bak),
        "features_parquet": str(features_parquet),
        "symbol": symbol,
        "agg": str(agg),
        "prefix": str(prefix),
        "merged_cols": out_cols,
        "rows": int(merged.shape[0]),
    }
