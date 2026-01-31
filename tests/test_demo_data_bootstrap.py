from __future__ import annotations

import sys
from pathlib import Path


def test_demo_data_bootstrap_creates_feather_files(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from agent_market.demo_data import ensure_demo_ohlcv_feather

    datadir = tmp_path / "data"
    created = ensure_demo_ohlcv_feather(
        datadir=datadir,
        exchange="kucoin",
        pairs=["BTC/USDT", "ETH/USDT"],
        timeframe="1h",
        timerange="20260101-20260110",
    )
    assert len(created) == 2
    assert (datadir / "kucoin" / "BTC_USDT-1h.feather").exists()
    assert (datadir / "kucoin" / "ETH_USDT-1h.feather").exists()
