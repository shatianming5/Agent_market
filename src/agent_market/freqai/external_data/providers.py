from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"


def fetch_fear_greed() -> pd.DataFrame:
    """拉取 Alternative.me 全量历史恐慌贪婪指数，返回按 timestamp 升序排列的 DataFrame。"""
    resp = requests.get(_FNG_URL, timeout=15)
    resp.raise_for_status()
    records = resp.json()["data"]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df["fng_value"] = df["value"].astype(float)
    df["fng_class"] = df["value_classification"]
    return df[["timestamp", "fng_value", "fng_class"]].sort_values("timestamp").reset_index(drop=True)


def fetch_lunarcrush(symbol: str, api_key: str, interval: str = "hour") -> pd.DataFrame:
    """
    拉取 LunarCrush v4 时序数据。
    interval: 'hour' | 'day'
    返回含 timestamp / galaxy_score / alt_rank / sentiment 列的 DataFrame。
    """
    url = f"https://lunarcrush.com/api4/public/coins/{symbol}/time-series/v2"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"interval": interval, "bucket": interval}
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        logger.warning("LunarCrush returned empty data for %s", symbol)
        return pd.DataFrame(columns=["timestamp", "galaxy_score", "alt_rank", "sentiment"])
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
    keep = [c for c in ["galaxy_score", "alt_rank", "sentiment"] if c in df.columns]
    return df[["timestamp"] + keep].sort_values("timestamp").reset_index(drop=True)


def fetch_santiment(metric: str, slug: str, api_key: str,
                    from_date: str, to_date: str, interval: str = "1d") -> pd.DataFrame:
    """
    通过 Santiment GraphQL API 拉取单个指标。
    metric: 如 'sentiment_balance_total', 'social_volume_total'
    slug:   如 'bitcoin', 'ethereum'
    返回含 timestamp / value 列的 DataFrame。
    """
    query = f"""
    {{
      getMetric(metric: "{metric}") {{
        timeseriesData(
          slug: "{slug}"
          from: "{from_date}T00:00:00Z"
          to: "{to_date}T00:00:00Z"
          interval: "{interval}"
        ) {{
          datetime
          value
        }}
      }}
    }}
    """
    url = f"https://api.santiment.net/graphql?apikey={api_key}"
    resp = requests.post(url, json={"query": query}, timeout=30)
    resp.raise_for_status()
    rows = resp.json()["data"]["getMetric"]["timeseriesData"]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["datetime"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["timestamp", "value"]].sort_values("timestamp").reset_index(drop=True)
