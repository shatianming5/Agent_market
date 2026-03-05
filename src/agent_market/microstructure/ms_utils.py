"""Shared utilities for microstructure sub-package."""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


def ensure_pandas() -> Any:
    """Import and return pandas, raising a clear error if unavailable."""
    try:
        import pandas as pd  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Missing dependency for microstructure features: {exc}") from exc
    return pd


def iter_ndjson_gz(path: Path) -> Iterator[Dict[str, Any]]:
    """Iterate over newline-delimited JSON records from a gzipped file."""
    with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def topic_symbol(topic: str) -> Optional[str]:
    """Extract symbol from a KuCoin-style topic string like ``/market/level2:BTC-USDT``."""
    t = str(topic or "")
    if ":" not in t:
        return None
    return t.split(":", 1)[1].strip() or None


def utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
