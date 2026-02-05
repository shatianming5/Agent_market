from __future__ import annotations

import json
from pathlib import Path


def test_micro_features_from_freqtrade_config(tmp_path: Path):
    from agent_market.demo_data import ensure_demo_ohlcv_feather
    from agent_market.microstructure.micro_feature import generate_micro_features_from_freqtrade_config

    datadir = tmp_path / "data"
    ensure_demo_ohlcv_feather(
        datadir=datadir,
        exchange="kucoin",
        pairs=["BTC/USDT", "ETH/USDT"],
        timeframe="1h",
        timerange="20260101-20260110",
    )

    cfg = {
        "exchange": {"name": "kucoin", "pair_whitelist": ["BTC/USDT", "ETH/USDT"]},
        "datadir": str(datadir),
        "freqai": {"feature_parameters": {"include_timeframes": ["1h"], "label_period_candles": 12}},
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    out_dir = tmp_path / "out"
    outputs = generate_micro_features_from_freqtrade_config(
        root=tmp_path,
        freqtrade_config=cfg_path,
        run_id="run1",
        out_dir=out_dir,
        timerange="20260102-20260105",
    )

    assert outputs.features_parquet.exists()
    assert outputs.manifest_json.exists()

    import pandas as pd

    df = pd.read_parquet(outputs.features_parquet)
    for col in ["date", "pair", "close", "volume", "ret_1", "rv_24", "vol_z_24", "amihud_24"]:
        assert col in df.columns

