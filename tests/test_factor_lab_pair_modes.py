from __future__ import annotations

import json

import numpy as np
import pandas as pd

from agent_market.factor_lab import features, mining


def _write_ohlcv(path, dates: pd.DatetimeIndex, close: np.ndarray) -> None:
    close = np.asarray(close, dtype=float)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.full(len(close), 1000.0),
        }
    )
    df.to_feather(path)


def test_build_big_pair_spread_label(tmp_path, monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=80, freq="1h", tz="UTC")
    btc = 100.0 * np.cumprod(1.0 + np.full(len(dates), 0.001))
    eth = 200.0 * np.cumprod(1.0 + np.full(len(dates), 0.0015))

    _write_ohlcv(tmp_path / "BTC_USDT-1h.feather", dates, btc)
    _write_ohlcv(tmp_path / "ETH_USDT-1h.feather", dates, eth)
    feat_file = tmp_path / "freqai_features_real.json"
    feat_file.write_text(json.dumps({"features": []}), encoding="utf-8")

    monkeypatch.setattr(mining, "FEATURE_FILE", feat_file)
    monkeypatch.setattr(mining, "apply_configured_features", lambda df, cfg: df)

    big, _ = mining.build_big(
        timeframe="1h",
        label_bars=3,
        label_mode="pair_spread_btc",
        pair_reference="BTC/USDT",
        data_dir=tmp_path,
        pairs="auto",
    )

    btc_rows = big.loc[big["__pair__"] == "BTC/USDT", "__fwd_ret__"]
    assert btc_rows.isna().all()

    eth_rows = big.loc[big["__pair__"] == "ETH/USDT", ["date", "__fwd_ret__"]].set_index("date")["__fwd_ret__"]
    expected = (pd.Series(eth, index=dates).shift(-3) / pd.Series(eth, index=dates) - 1.0) - (
        pd.Series(btc, index=dates).shift(-3) / pd.Series(btc, index=dates) - 1.0
    )
    joined = pd.concat([eth_rows, expected.rename("expected")], axis=1).dropna()
    assert np.allclose(joined["__fwd_ret__"].to_numpy(), joined["expected"].to_numpy(), atol=1e-10)


def test_build_big_pair_beta_resid_label_reduces_spread_variance(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=500, freq="1h", tz="UTC")
    btc_ret = rng.normal(0.0, 0.002, len(dates))
    eth_ret = 2.0 * btc_ret
    btc = 100.0 * np.exp(np.cumsum(btc_ret))
    eth = 200.0 * np.exp(np.cumsum(eth_ret))

    _write_ohlcv(tmp_path / "BTC_USDT-1h.feather", dates, btc)
    _write_ohlcv(tmp_path / "ETH_USDT-1h.feather", dates, eth)
    feat_file = tmp_path / "freqai_features_real.json"
    feat_file.write_text(json.dumps({"features": []}), encoding="utf-8")

    monkeypatch.setattr(mining, "FEATURE_FILE", feat_file)
    monkeypatch.setattr(mining, "apply_configured_features", lambda df, cfg: df)

    big_spread, _ = mining.build_big(
        timeframe="1h",
        label_bars=3,
        label_mode="pair_spread_btc",
        pair_reference="BTC/USDT",
        data_dir=tmp_path,
        pairs="auto",
    )
    big_resid, _ = mining.build_big(
        timeframe="1h",
        label_bars=3,
        label_mode="pair_beta_resid_btc",
        pair_reference="BTC/USDT",
        data_dir=tmp_path,
        pairs="auto",
    )

    spread_lbl = big_spread.loc[big_spread["__pair__"] == "ETH/USDT", "__fwd_ret__"].dropna()
    resid_lbl = big_resid.loc[big_resid["__pair__"] == "ETH/USDT", "__fwd_ret__"].dropna()
    assert resid_lbl.std(ddof=0) < spread_lbl.std(ddof=0) * 0.25

    btc_rows = big_resid.loc[big_resid["__pair__"] == "BTC/USDT", "__fwd_ret__"]
    assert btc_rows.isna().all()


def test_merge_pair_relative_writes_pair_columns(tmp_path) -> None:
    dates = pd.date_range("2025-01-01", periods=120, freq="1h", tz="UTC")
    btc = 100.0 * np.cumprod(1.0 + np.sin(np.linspace(0, 6.0, len(dates))) * 0.001 + 0.0005)
    eth = 220.0 * np.cumprod(1.0 + np.sin(np.linspace(0, 7.0, len(dates))) * 0.0014 + 0.0008)

    _write_ohlcv(tmp_path / "BTC_USDT-1h.feather", dates, btc)
    _write_ohlcv(tmp_path / "ETH_USDT-1h.feather", dates, eth)

    out = features.merge_pair_relative(reference_pair="BTC/USDT", beta_window=24, data_dir=tmp_path, pairs="auto")
    assert out["BTC/USDT"].startswith("+")
    assert out["ETH/USDT"].startswith("+")

    eth_df = pd.read_feather(tmp_path / "ETH_USDT-1h.feather")
    btc_df = pd.read_feather(tmp_path / "BTC_USDT-1h.feather")
    assert set(features.PAIR_COLS).issubset(set(eth_df.columns))
    assert set(features.PAIR_COLS).issubset(set(btc_df.columns))
    assert float(eth_df["pair_log_ratio_btc"].abs().sum()) > 0.0
    assert float(btc_df["pair_log_ratio_btc"].abs().max()) == 0.0


def test_merge_pair_relative_supports_multiple_references(tmp_path) -> None:
    dates = pd.date_range("2025-01-01", periods=120, freq="1h", tz="UTC")
    btc = 100.0 * np.cumprod(1.0 + np.sin(np.linspace(0, 6.0, len(dates))) * 0.001 + 0.0005)
    eth = 220.0 * np.cumprod(1.0 + np.sin(np.linspace(0, 7.0, len(dates))) * 0.0014 + 0.0008)
    sol = 80.0 * np.cumprod(1.0 + np.sin(np.linspace(0, 8.0, len(dates))) * 0.0018 + 0.0009)

    _write_ohlcv(tmp_path / "BTC_USDT-1h.feather", dates, btc)
    _write_ohlcv(tmp_path / "ETH_USDT-1h.feather", dates, eth)
    _write_ohlcv(tmp_path / "SOL_USDT-1h.feather", dates, sol)

    out = features.merge_pair_relative(
        reference_pairs="BTC/USDT,ETH/USDT",
        beta_window=24,
        data_dir=tmp_path,
        pairs="auto",
    )
    assert out["SOL/USDT"].startswith("+18 cols")

    sol_df = pd.read_feather(tmp_path / "SOL_USDT-1h.feather")
    assert "pair_log_ratio_btc" in sol_df.columns
    assert "pair_log_ratio_eth" in sol_df.columns
    assert float(sol_df["pair_log_ratio_eth"].abs().sum()) > 0.0
