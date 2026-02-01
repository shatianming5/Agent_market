from __future__ import annotations

from pathlib import Path

import pytest


pypfopt = pytest.importorskip("pypfopt")


def test_portfolio_hrp_weights_sum_to_one(tmp_path: Path) -> None:
    from agent_market.demo_data import ensure_demo_ohlcv_feather
    from agent_market.portfolio_opt import (
        compute_returns,
        load_prices_from_feather,
        optimize_hrp,
    )

    timerange = "20260101-20260110"
    datadir = tmp_path / "data"
    ensure_demo_ohlcv_feather(
        datadir=datadir,
        exchange="kucoin",
        pairs=["BTC/USDT", "ETH/USDT"],
        timeframe="1h",
        timerange=timerange,
    )

    prices = load_prices_from_feather(
        tmp_path,
        exchange="kucoin",
        pairs=["BTC/USDT", "ETH/USDT"],
        timeframe="1h",
        timerange=timerange,
        data_dir="data",
    )
    returns = compute_returns(prices, "log")
    result = optimize_hrp(returns)
    weights = result["weights"]

    assert all(k in weights for k in prices.columns)
    total = sum(float(weights.get(k, 0.0)) for k in prices.columns)
    assert 0.999 <= total <= 1.001
    for k in prices.columns:
        w = float(weights.get(k, 0.0))
        assert 0.0 <= w <= 1.0

