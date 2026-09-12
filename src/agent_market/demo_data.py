from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Sequence

from agent_market import paths


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else paths.resolve_repo_path(p)


def _timeframe_to_pandas_freq(timeframe: str) -> str:
    tf = str(timeframe or "").strip().lower()
    m = re.match(r"^(\d+)([mhdw])$", tf)
    if not m:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "m":
        return f"{n}min"
    if unit == "h":
        return f"{n}h"
    if unit == "d":
        return f"{n}D"
    if unit == "w":
        return f"{n}W"
    raise ValueError(f"Unsupported timeframe: {timeframe!r}")


def _parse_timerange(timerange: str) -> tuple[datetime, datetime]:
    parts = str(timerange or "").split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid timerange: {timerange!r}")
    a = re.sub(r"\D", "", parts[0])[:8]
    b = re.sub(r"\D", "", parts[1])[:8]
    if len(a) != 8 or len(b) != 8:
        raise ValueError(f"Invalid timerange: {timerange!r}")
    start = datetime.strptime(a, "%Y%m%d")
    end_date = datetime.strptime(b, "%Y%m%d")
    end_exclusive = end_date + timedelta(days=1)
    return start, end_exclusive


def ensure_demo_ohlcv_feather(
    *,
    datadir: Path,
    exchange: str,
    pairs: Sequence[str],
    timeframe: str,
    timerange: str,
    start_buffer_days: int = 3,
    end_buffer_days: int = 1,
    min_rows: int = 80,
) -> List[Path]:
    """Create minimal OHLCV feather files (best-effort) when missing.

    This is intended for smoke tests / local development only.
    """
    if not pairs:
        return []

    dataformat = "feather"
    if dataformat != "feather":  # pragma: no cover
        return []

    try:
        import numpy as np  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Missing pandas/numpy for demo data generation: {exc}") from exc

    start_dt, end_exclusive = _parse_timerange(timerange)
    start_dt = start_dt - timedelta(days=int(start_buffer_days))
    end_exclusive = end_exclusive + timedelta(days=int(end_buffer_days))
    freq = _timeframe_to_pandas_freq(timeframe)

    ex_dir = datadir / exchange
    ex_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start=start_dt, end=end_exclusive, freq=freq)
    if len(dates) >= 2 and dates[-1].to_pydatetime() == end_exclusive:
        dates = dates[:-1]
    if len(dates) < int(min_rows):
        return []

    required_start = pd.Timestamp(start_dt, tz="UTC")
    required_end = pd.Timestamp(end_exclusive, tz="UTC")
    required_last = required_end - pd.tseries.frequencies.to_offset(freq)

    created: list[Path] = []
    for pair in pairs:
        if not isinstance(pair, str) or "/" not in pair:
            continue
        sanitized = pair.replace("/", "_")
        out_path = ex_dir / f"{sanitized}-{timeframe}.feather"
        if out_path.exists():
            try:
                existing = pd.read_feather(out_path, columns=["date"])
                if "date" in existing.columns and not existing.empty:
                    s = pd.to_datetime(existing["date"], utc=True, errors="coerce")
                    if s.notna().any() and s.min() <= required_start and s.max() >= required_last:
                        continue
            except Exception:
                pass
            try:
                out_path.unlink()
            except Exception:
                pass

        base_price = 100.0
        if pair.startswith("BTC/"):
            base_price = 50000.0
        elif pair.startswith("ETH/"):
            base_price = 3000.0

        seed = abs(hash((exchange, pair, timeframe, timerange))) % (2**32)
        rs = np.random.RandomState(seed)
        drift = 0.00005
        returns = rs.normal(loc=drift, scale=0.01, size=len(dates))
        close = base_price * np.exp(np.cumsum(returns))
        open_ = np.concatenate([[close[0]], close[:-1]])
        spread = rs.uniform(0.0, 0.003, size=len(dates))
        high = np.maximum(open_, close) * (1.0 + spread)
        low = np.minimum(open_, close) * (1.0 - spread)
        volume = rs.uniform(10.0, 1200.0, size=len(dates))

        df = pd.DataFrame(
            {
                "date": dates,
                "open": open_.astype("float64"),
                "high": high.astype("float64"),
                "low": low.astype("float64"),
                "close": close.astype("float64"),
                "volume": volume.astype("float64"),
            }
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_feather(out_path)
        created.append(out_path)

    return created


def ensure_demo_ohlcv_from_flow_config(
    root: Path,
    flow_config_path: Path,
) -> List[Path]:
    """Best-effort resolve flow config and generate required OHLCV demo data."""
    flow_cfg = _load_json(flow_config_path)
    backtest = flow_cfg.get("backtest") or {}

    ft_cfg = backtest.get("config")
    if not ft_cfg:
        feat_args = (flow_cfg.get("feature") or {}).get("args") or []
        if isinstance(feat_args, list):
            for idx, item in enumerate(feat_args):
                if str(item) == "--config" and idx + 1 < len(feat_args):
                    ft_cfg = str(feat_args[idx + 1])
                    break
    if not ft_cfg:
        return []

    ft_cfg_path = _resolve(root, str(ft_cfg))
    if not ft_cfg_path.exists():
        return []

    ft_payload = _load_json(ft_cfg_path)
    dataformat = str(ft_payload.get("dataformat_ohlcv") or "feather").lower()
    if dataformat != "feather":
        return []

    datadir = _resolve(root, str(ft_payload.get("datadir") or "user_data/data"))
    exchange = str((ft_payload.get("exchange") or {}).get("name") or "unknown")
    pairs = (ft_payload.get("exchange") or {}).get("pair_whitelist") or []
    if not isinstance(pairs, list):
        return []

    timeframe = str(backtest.get("timeframe") or ft_payload.get("timeframe") or "1h")
    timerange = str(backtest.get("timerange") or "")
    if "-" not in timerange:
        return []

    return ensure_demo_ohlcv_feather(
        datadir=datadir,
        exchange=exchange,
        pairs=[str(p) for p in pairs if str(p).strip()],
        timeframe=timeframe,
        timerange=timerange,
    )


__all__ = [
    "ensure_demo_ohlcv_feather",
    "ensure_demo_ohlcv_from_flow_config",
]
