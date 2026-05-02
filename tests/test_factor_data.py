from __future__ import annotations

import pandas as pd

from agent_market.factor_lab import data


def test_binance_archive_listing_discovers_usdt_perpetual_like_symbols() -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <CommonPrefixes><Prefix>data/futures/um/monthly/klines/BTCUSDT/</Prefix></CommonPrefixes>
      <CommonPrefixes><Prefix>data/futures/um/monthly/klines/BTCUSDT_250926/</Prefix></CommonPrefixes>
      <CommonPrefixes><Prefix>data/futures/um/monthly/klines/ETHUSDC/</Prefix></CommonPrefixes>
      <CommonPrefixes><Prefix>data/futures/um/monthly/klines/BNXUSDTSETTLED/</Prefix></CommonPrefixes>
      <CommonPrefixes><Prefix>data/futures/um/monthly/klines/1000PEPEUSDT/</Prefix></CommonPrefixes>
    </ListBucketResult>
    """

    assert data._binance_archive_symbols_from_listing(xml) == ["1000PEPEUSDT", "BTCUSDT"]  # noqa: SLF001


def test_usdt_pair_from_symbol_keeps_prefixed_contract_symbols() -> None:
    assert data._usdt_pair_from_symbol("1000PEPEUSDT") == "1000PEPE/USDT"  # noqa: SLF001


def test_futures_downloaders_accept_4h_timeframe() -> None:
    assert data.BINANCE_BAR_MAP["4h"] == ("4h", 4 * 60 * 60_000)
    assert data.BYBIT_BAR_MAP["4h"] == ("240", 4 * 60 * 60_000)
    assert data.OKX_BAR_MAP["4h"] == ("4H", 4 * 60 * 60_000)


def test_regular_bar_gap_detector_catches_internal_missing_candle() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 08:00"], utc=True)})

    assert data._has_regular_bar_gaps(frame, bar_ms=4 * 60 * 60_000) is True  # noqa: SLF001


def test_fill_internal_ohlcv_gaps_uses_flat_zero_volume_bar() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 08:00"], utc=True),
            "open": [100.0, 102.0],
            "high": [101.0, 103.0],
            "low": [99.0, 101.0],
            "close": [100.5, 102.5],
            "volume": [10.0, 12.0],
        }
    )

    filled = data._fill_internal_ohlcv_gaps(frame, bar_ms=4 * 60 * 60_000)  # noqa: SLF001

    assert len(filled) == 3
    assert filled.loc[1, "date"] == pd.Timestamp("2026-01-01 04:00", tz="UTC")
    assert filled.loc[1, ["open", "high", "low", "close"]].tolist() == [100.5, 100.5, 100.5, 100.5]
    assert filled.loc[1, "volume"] == 0.0
