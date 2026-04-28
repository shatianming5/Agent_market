from __future__ import annotations

import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from .aligner import align_to_ohlcv
from .cache import ExternalDataCache
from .providers import fetch_fear_greed, fetch_lunarcrush, fetch_santiment

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = "user_data/external_data"


def apply_external_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    主入口：根据 config['external_data'] 配置向 df 注入外部特征列。
    df 必须含 'date' 列（UTC Timestamp）。
    所有特征均经过 backward merge_asof + shift，无前向偏差。
    """
    ext_cfg = config.get("external_data")
    if not ext_cfg:
        return df

    cache_dir = ext_cfg.get("cache_dir", _DEFAULT_CACHE_DIR)
    cache = ExternalDataCache(cache_dir)

    if ext_cfg.get("fear_greed", {}).get("enabled", False):
        df = _attach_fear_greed(df, ext_cfg["fear_greed"], cache)

    if ext_cfg.get("lunarcrush", {}).get("enabled", False):
        df = _attach_lunarcrush(df, ext_cfg["lunarcrush"], cache)

    if ext_cfg.get("santiment", {}).get("enabled", False):
        df = _attach_santiment(df, ext_cfg["santiment"], cache)

    return df


# ── Fear & Greed ──────────────────────────────────────────────────────────────

def _attach_fear_greed(df: pd.DataFrame, cfg: Dict, cache: ExternalDataCache) -> pd.DataFrame:
    shift = int(cfg.get("shift_periods", 1))
    max_age = float(cfg.get("max_age_hours", 23.0))

    fng = cache.get_or_fetch("fear_greed", fetch_fear_greed, max_age_hours=max_age)

    aligned = align_to_ohlcv(
        df["date"], fng, date_col="timestamp",
        value_cols=["fng_value", "fng_class"], shift_periods=shift,
    )

    df["fng_value"] = aligned["fng_value"].values
    df["fng_class_enc"] = _encode_fng_class(aligned["fng_class"])

    # 派生特征
    fng_val = df["fng_value"]
    window = int(cfg.get("zscore_window", 30))
    df["fng_zscore"] = (fng_val - fng_val.rolling(window, min_periods=5).mean()) / (
        fng_val.rolling(window, min_periods=5).std().replace(0, np.nan)
    )
    df["fng_delta"] = fng_val.diff()

    logger.info("Attached fear_greed features (%d rows valid)", df["fng_value"].notna().sum())
    return df


def _encode_fng_class(series: pd.Series) -> pd.Series:
    mapping = {
        "Extreme Fear": 0,
        "Fear": 1,
        "Neutral": 2,
        "Greed": 3,
        "Extreme Greed": 4,
    }
    return series.map(mapping).astype(float)


# ── LunarCrush ────────────────────────────────────────────────────────────────

def _attach_lunarcrush(df: pd.DataFrame, cfg: Dict, cache: ExternalDataCache) -> pd.DataFrame:
    api_key = cfg.get("api_key") or os.environ.get("LUNARCRUSH_API_KEY", "")
    if not api_key:
        logger.warning("LunarCrush api_key missing, skipping")
        return df

    symbol = cfg.get("symbol", "BTC")
    interval = cfg.get("interval", "hour")
    shift = int(cfg.get("shift_periods", 1))
    max_age = float(cfg.get("max_age_hours", 1.5))
    metrics = cfg.get("metrics", ["galaxy_score", "alt_rank", "sentiment"])

    cache_key = f"lunarcrush_{symbol}_{interval}"
    lc = cache.get_or_fetch(
        cache_key,
        lambda: fetch_lunarcrush(symbol, api_key, interval),
        max_age_hours=max_age,
    )

    avail = [m for m in metrics if m in lc.columns]
    if not avail:
        logger.warning("LunarCrush: none of %s found in response", metrics)
        return df

    aligned = align_to_ohlcv(
        df["date"], lc, date_col="timestamp", value_cols=avail, shift_periods=shift,
    )
    for col in avail:
        df[f"lc_{col}"] = aligned[col].values

    logger.info("Attached LunarCrush features: %s", avail)
    return df


# ── Santiment ─────────────────────────────────────────────────────────────────

def _attach_santiment(df: pd.DataFrame, cfg: Dict, cache: ExternalDataCache) -> pd.DataFrame:
    api_key = cfg.get("api_key") or os.environ.get("SANTIMENT_API_KEY", "")
    if not api_key:
        logger.warning("Santiment api_key missing, skipping")
        return df

    slug = cfg.get("slug", "bitcoin")
    interval = cfg.get("interval", "1d")
    shift = int(cfg.get("shift_periods", 1))
    max_age = float(cfg.get("max_age_hours", 23.0))
    metrics: list[dict] = cfg.get("metrics", [
        {"name": "sentiment_balance_total", "col": "san_sentiment"},
        {"name": "social_volume_total",     "col": "san_social_vol"},
    ])

    dates = pd.to_datetime(df["date"], utc=True)
    from_date = dates.min().strftime("%Y-%m-%d")
    to_date   = dates.max().strftime("%Y-%m-%d")

    for m in metrics:
        metric_name = m["name"]
        col_name    = m.get("col", f"san_{metric_name}")
        cache_key   = f"santiment_{slug}_{metric_name}_{interval}"

        san = cache.get_or_fetch(
            cache_key,
            lambda mn=metric_name: fetch_santiment(mn, slug, api_key, from_date, to_date, interval),
            max_age_hours=max_age,
        )
        san = san.rename(columns={"value": col_name})

        aligned = align_to_ohlcv(
            df["date"], san, date_col="timestamp", value_cols=[col_name], shift_periods=shift,
        )
        df[col_name] = aligned[col_name].values

    logger.info("Attached Santiment features: %s", [m.get("col") for m in metrics])
    return df
