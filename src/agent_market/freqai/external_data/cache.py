from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExternalDataCache:
    """本地 feather 缓存，按 key 存取外部数据，支持 TTL 刷新判断。"""

    def __init__(self, cache_dir: str):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.feather"

    def load(self, key: str) -> Optional[pd.DataFrame]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            df = pd.read_feather(p)
            logger.debug("External cache hit: %s (%d rows)", key, len(df))
            return df
        except Exception as e:
            logger.warning("Cache read failed for %s: %s", key, e)
            return None

    def save(self, key: str, df: pd.DataFrame) -> None:
        p = self._path(key)
        df.to_feather(p)
        logger.debug("External cache saved: %s (%d rows)", key, len(df))

    def needs_refresh(self, key: str, max_age_hours: float = 23.0) -> bool:
        p = self._path(key)
        if not p.exists():
            return True
        age_seconds = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).total_seconds()
        return age_seconds > max_age_hours * 3600

    def get_or_fetch(self, key: str, fetcher, max_age_hours: float = 23.0) -> pd.DataFrame:
        """若缓存新鲜则直接返回，否则调用 fetcher() 刷新后返回。"""
        if not self.needs_refresh(key, max_age_hours):
            cached = self.load(key)
            if cached is not None:
                return cached
        logger.info("Fetching external data: %s", key)
        df = fetcher()
        self.save(key, df)
        return df
