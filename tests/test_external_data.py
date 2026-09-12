"""逐一测试 external_data 各模块：cache / aligner / providers / feature_builder。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_cache_dir(tmp_path):
    return str(tmp_path / "ext_cache")


@pytest.fixture()
def sample_fng_df():
    """模拟 Alternative.me 恐慌贪婪数据（日频，30天）。"""
    dates = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    return pd.DataFrame({
        "timestamp": dates,
        "fng_value": np.linspace(20, 80, 30),          # 从恐慌到贪婪
        "fng_class": (["Fear"] * 10 + ["Neutral"] * 10 + ["Greed"] * 10),
    })


@pytest.fixture()
def sample_ohlcv_df():
    """模拟 1h OHLCV 数据（720根K线 = 30天）。"""
    dates = pd.date_range("2024-01-01", periods=720, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    close = 40000 + rng.standard_normal(720).cumsum() * 100
    return pd.DataFrame({
        "date":   dates,
        "open":   close - 50,
        "high":   close + 100,
        "low":    close - 100,
        "close":  close,
        "volume": rng.uniform(1, 10, 720),
    })


# ──────────────────────────────────────────────────────────────────────────────
# 1. ExternalDataCache
# ──────────────────────────────────────────────────────────────────────────────

class TestExternalDataCache:
    def _make(self, cache_dir):
        from agent_market.freqai.external_data.cache import ExternalDataCache
        return ExternalDataCache(cache_dir)

    def test_cache_dir_created(self, tmp_cache_dir):
        cache = self._make(tmp_cache_dir)
        assert Path(tmp_cache_dir).exists()

    def test_load_missing_returns_none(self, tmp_cache_dir):
        cache = self._make(tmp_cache_dir)
        assert cache.load("nonexistent") is None

    def test_save_and_load_roundtrip(self, tmp_cache_dir, sample_fng_df):
        cache = self._make(tmp_cache_dir)
        cache.save("fng", sample_fng_df)
        loaded = cache.load("fng")
        assert loaded is not None
        assert len(loaded) == len(sample_fng_df)
        pd.testing.assert_frame_equal(loaded.reset_index(drop=True),
                                      sample_fng_df.reset_index(drop=True))

    def test_needs_refresh_true_when_missing(self, tmp_cache_dir):
        cache = self._make(tmp_cache_dir)
        assert cache.needs_refresh("fng") is True

    def test_needs_refresh_false_when_fresh(self, tmp_cache_dir, sample_fng_df):
        cache = self._make(tmp_cache_dir)
        cache.save("fng", sample_fng_df)
        assert cache.needs_refresh("fng", max_age_hours=1.0) is False

    def test_needs_refresh_true_when_old(self, tmp_cache_dir, sample_fng_df):
        cache = self._make(tmp_cache_dir)
        cache.save("fng", sample_fng_df)
        # max_age = 0 → 总是过期
        assert cache.needs_refresh("fng", max_age_hours=0.0) is True

    def test_get_or_fetch_calls_fetcher_when_missing(self, tmp_cache_dir, sample_fng_df):
        cache = self._make(tmp_cache_dir)
        fetcher = MagicMock(return_value=sample_fng_df)
        result = cache.get_or_fetch("fng", fetcher)
        fetcher.assert_called_once()
        assert len(result) == len(sample_fng_df)

    def test_get_or_fetch_skips_fetcher_when_fresh(self, tmp_cache_dir, sample_fng_df):
        cache = self._make(tmp_cache_dir)
        cache.save("fng", sample_fng_df)
        fetcher = MagicMock(return_value=sample_fng_df)
        cache.get_or_fetch("fng", fetcher, max_age_hours=1.0)
        fetcher.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# 2. align_to_ohlcv
# ──────────────────────────────────────────────────────────────────────────────

class TestAligner:
    def _align(self, ohlcv_dates, external_df, **kw):
        from agent_market.freqai.external_data.aligner import align_to_ohlcv
        return align_to_ohlcv(ohlcv_dates, external_df, **kw)

    def test_output_length_matches_ohlcv(self, sample_ohlcv_df, sample_fng_df):
        result = self._align(sample_ohlcv_df["date"], sample_fng_df,
                             date_col="timestamp", value_cols=["fng_value"])
        assert len(result) == len(sample_ohlcv_df)

    def test_output_index_matches_ohlcv_index(self, sample_ohlcv_df, sample_fng_df):
        result = self._align(sample_ohlcv_df["date"], sample_fng_df,
                             date_col="timestamp", value_cols=["fng_value"])
        assert list(result.index) == list(sample_ohlcv_df.index)

    def test_no_future_data_with_shift_1(self, sample_ohlcv_df, sample_fng_df):
        """shift=1 时，第0根K线（2024-01-01 00:00）应为 NaN（无历史数据）。"""
        result = self._align(sample_ohlcv_df["date"], sample_fng_df,
                             date_col="timestamp", value_cols=["fng_value"],
                             shift_periods=1)
        assert pd.isna(result["fng_value"].iloc[0])

    def test_shift_0_fills_first_bar(self, sample_ohlcv_df, sample_fng_df):
        """shift=0 时，第0根K线可以有值（当天数据）。"""
        result = self._align(sample_ohlcv_df["date"], sample_fng_df,
                             date_col="timestamp", value_cols=["fng_value"],
                             shift_periods=0)
        # 2024-01-01 00:00 对应 FNG 第0条（2024-01-01），值约为 20.0
        assert not pd.isna(result["fng_value"].iloc[0])

    def test_daily_external_forward_filled_to_hourly(self, sample_ohlcv_df, sample_fng_df):
        """日频 FNG 应向前填充到每小时K线。同一天内所有小时的值应相同。"""
        result = self._align(sample_ohlcv_df["date"], sample_fng_df,
                             date_col="timestamp", value_cols=["fng_value"],
                             shift_periods=0)
        # 2024-01-02 的所有小时（bars 24~47）值应与 2024-01-02 FNG 相同
        day2_bars = result["fng_value"].iloc[24:48].dropna()
        assert day2_bars.nunique() == 1, "同一天所有小时应填充相同的日频值"

    def test_values_are_floats_not_objects(self, sample_ohlcv_df, sample_fng_df):
        result = self._align(sample_ohlcv_df["date"], sample_fng_df,
                             date_col="timestamp", value_cols=["fng_value"])
        assert result["fng_value"].dtype in (np.float32, np.float64, float)

    def test_only_requested_cols_returned(self, sample_ohlcv_df, sample_fng_df):
        result = self._align(sample_ohlcv_df["date"], sample_fng_df,
                             date_col="timestamp", value_cols=["fng_value"])
        assert list(result.columns) == ["fng_value"]

    def test_all_cols_when_value_cols_none(self, sample_ohlcv_df, sample_fng_df):
        result = self._align(sample_ohlcv_df["date"], sample_fng_df,
                             date_col="timestamp", value_cols=None)
        assert "fng_value" in result.columns
        assert "fng_class" in result.columns


# ──────────────────────────────────────────────────────────────────────────────
# 3. providers（mock HTTP）
# ──────────────────────────────────────────────────────────────────────────────

class TestProviders:
    # ── Fear & Greed ──────────────────────────────────────────────────────────

    def _mock_fng_response(self):
        """构造与 Alternative.me 真实响应格式一致的 mock。"""
        import time as _time
        records = []
        for i in range(5):
            ts = int(_time.mktime((2024, 1, i + 1, 0, 0, 0, 0, 0, 0)))
            records.append({
                "value": str(20 + i * 10),
                "value_classification": ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"][i],
                "timestamp": str(ts),
                "time_until_update": "3600",
            })
        return {"name": "Fear and Greed Index", "data": records, "metadata": {"error": None}}

    def test_fetch_fear_greed_columns(self):
        from agent_market.freqai.external_data.providers import fetch_fear_greed
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_fng_response()
        mock_resp.raise_for_status = MagicMock()
        with patch("agent_market.freqai.external_data.providers.requests.get",
                   return_value=mock_resp):
            df = fetch_fear_greed()
        assert set(df.columns) == {"timestamp", "fng_value", "fng_class"}

    def test_fetch_fear_greed_sorted_ascending(self):
        from agent_market.freqai.external_data.providers import fetch_fear_greed
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_fng_response()
        mock_resp.raise_for_status = MagicMock()
        with patch("agent_market.freqai.external_data.providers.requests.get",
                   return_value=mock_resp):
            df = fetch_fear_greed()
        assert df["timestamp"].is_monotonic_increasing

    def test_fetch_fear_greed_value_range(self):
        from agent_market.freqai.external_data.providers import fetch_fear_greed
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_fng_response()
        mock_resp.raise_for_status = MagicMock()
        with patch("agent_market.freqai.external_data.providers.requests.get",
                   return_value=mock_resp):
            df = fetch_fear_greed()
        assert df["fng_value"].between(0, 100).all()

    def test_fetch_fear_greed_timestamp_utc(self):
        from agent_market.freqai.external_data.providers import fetch_fear_greed
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_fng_response()
        mock_resp.raise_for_status = MagicMock()
        with patch("agent_market.freqai.external_data.providers.requests.get",
                   return_value=mock_resp):
            df = fetch_fear_greed()
        assert str(df["timestamp"].dt.tz) == "UTC"

    # ── LunarCrush ────────────────────────────────────────────────────────────

    def _mock_lc_response(self):
        import time as _time
        data = []
        for i in range(24):
            ts = int(_time.mktime((2024, 1, 1, i, 0, 0, 0, 0, 0)))
            data.append({"time": ts, "galaxy_score": 60 + i,
                          "alt_rank": 5 - i * 0.1, "sentiment": 0.5 + i * 0.01})
        return {"data": data}

    def test_fetch_lunarcrush_columns(self):
        from agent_market.freqai.external_data.providers import fetch_lunarcrush
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_lc_response()
        mock_resp.raise_for_status = MagicMock()
        with patch("agent_market.freqai.external_data.providers.requests.get",
                   return_value=mock_resp):
            df = fetch_lunarcrush("BTC", "fake_key")
        assert "timestamp" in df.columns
        assert "galaxy_score" in df.columns

    def test_fetch_lunarcrush_empty_response(self):
        from agent_market.freqai.external_data.providers import fetch_lunarcrush
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()
        with patch("agent_market.freqai.external_data.providers.requests.get",
                   return_value=mock_resp):
            df = fetch_lunarcrush("BTC", "fake_key")
        assert len(df) == 0

    # ── Santiment ─────────────────────────────────────────────────────────────

    def _mock_santiment_response(self):
        rows = [{"datetime": f"2024-01-0{i+1}T00:00:00Z", "value": float(i * 0.1)}
                for i in range(5)]
        return {"data": {"getMetric": {"timeseriesData": rows}}}

    def test_fetch_santiment_columns(self):
        from agent_market.freqai.external_data.providers import fetch_santiment
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_santiment_response()
        mock_resp.raise_for_status = MagicMock()
        with patch("agent_market.freqai.external_data.providers.requests.post",
                   return_value=mock_resp):
            df = fetch_santiment("sentiment_balance_total", "bitcoin", "fake_key",
                                 "2024-01-01", "2024-01-05")
        assert set(df.columns) == {"timestamp", "value"}

    def test_fetch_santiment_sorted(self):
        from agent_market.freqai.external_data.providers import fetch_santiment
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_santiment_response()
        mock_resp.raise_for_status = MagicMock()
        with patch("agent_market.freqai.external_data.providers.requests.post",
                   return_value=mock_resp):
            df = fetch_santiment("sentiment_balance_total", "bitcoin", "fake_key",
                                 "2024-01-01", "2024-01-05")
        assert df["timestamp"].is_monotonic_increasing


# ──────────────────────────────────────────────────────────────────────────────
# 4. FNG 编码：_encode_fng_class
# ──────────────────────────────────────────────────────────────────────────────

class TestFngEncoding:
    def _encode(self, values):
        from agent_market.freqai.external_data.feature_builder import _encode_fng_class
        return _encode_fng_class(pd.Series(values))

    def test_extreme_fear_is_0(self):
        assert self._encode(["Extreme Fear"]).iloc[0] == 0.0

    def test_extreme_greed_is_4(self):
        assert self._encode(["Extreme Greed"]).iloc[0] == 4.0

    def test_full_mapping(self):
        classes = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
        result = self._encode(classes)
        assert list(result) == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_unknown_class_becomes_nan(self):
        result = self._encode(["Unknown"])
        assert pd.isna(result.iloc[0])


# ──────────────────────────────────────────────────────────────────────────────
# 5. apply_external_features（端到端，mock 拉取）
# ──────────────────────────────────────────────────────────────────────────────

class TestFeatureBuilder:
    def _run(self, ohlcv_df, config, fng_df):
        from agent_market.freqai.external_data.feature_builder import apply_external_features
        with patch("agent_market.freqai.external_data.feature_builder.fetch_fear_greed",
                   return_value=fng_df):
            return apply_external_features(ohlcv_df.copy(), config)

    def test_noop_when_no_external_data_key(self, sample_ohlcv_df):
        from agent_market.freqai.external_data.feature_builder import apply_external_features
        out = apply_external_features(sample_ohlcv_df.copy(), {})
        assert list(out.columns) == list(sample_ohlcv_df.columns)

    def test_noop_when_fear_greed_disabled(self, sample_ohlcv_df, sample_fng_df, tmp_cache_dir):
        cfg = {"external_data": {"cache_dir": tmp_cache_dir,
                                  "fear_greed": {"enabled": False}}}
        out = self._run(sample_ohlcv_df, cfg, sample_fng_df)
        assert "fng_value" not in out.columns

    def test_fng_columns_added(self, sample_ohlcv_df, sample_fng_df, tmp_cache_dir):
        cfg = {"external_data": {"cache_dir": tmp_cache_dir,
                                  "fear_greed": {"enabled": True, "shift_periods": 1}}}
        out = self._run(sample_ohlcv_df, cfg, sample_fng_df)
        for col in ["fng_value", "fng_class_enc", "fng_zscore", "fng_delta"]:
            assert col in out.columns, f"缺少列: {col}"

    def test_fng_value_range(self, sample_ohlcv_df, sample_fng_df, tmp_cache_dir):
        cfg = {"external_data": {"cache_dir": tmp_cache_dir,
                                  "fear_greed": {"enabled": True, "shift_periods": 1}}}
        out = self._run(sample_ohlcv_df, cfg, sample_fng_df)
        valid = out["fng_value"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_fng_no_forward_bias(self, sample_ohlcv_df, sample_fng_df, tmp_cache_dir):
        """shift=1 时，第0行 fng_value 应为 NaN（无前一天数据）。"""
        cfg = {"external_data": {"cache_dir": tmp_cache_dir,
                                  "fear_greed": {"enabled": True, "shift_periods": 1}}}
        out = self._run(sample_ohlcv_df, cfg, sample_fng_df)
        assert pd.isna(out["fng_value"].iloc[0])

    def test_fng_class_enc_in_0_4(self, sample_ohlcv_df, sample_fng_df, tmp_cache_dir):
        cfg = {"external_data": {"cache_dir": tmp_cache_dir,
                                  "fear_greed": {"enabled": True, "shift_periods": 1}}}
        out = self._run(sample_ohlcv_df, cfg, sample_fng_df)
        valid = out["fng_class_enc"].dropna()
        assert valid.isin([0, 1, 2, 3, 4]).all()

    def test_fng_zscore_has_nans_at_start(self, sample_ohlcv_df, sample_fng_df, tmp_cache_dir):
        """zscore 在窗口期内（不足 min_periods）应为 NaN。"""
        cfg = {"external_data": {"cache_dir": tmp_cache_dir,
                                  "fear_greed": {"enabled": True, "shift_periods": 1,
                                                  "zscore_window": 30}}}
        out = self._run(sample_ohlcv_df, cfg, sample_fng_df)
        # 最初几行（不足 5 个有效样本）应为 NaN
        assert out["fng_zscore"].iloc[:5].isna().all()

    def test_row_count_unchanged(self, sample_ohlcv_df, sample_fng_df, tmp_cache_dir):
        cfg = {"external_data": {"cache_dir": tmp_cache_dir,
                                  "fear_greed": {"enabled": True, "shift_periods": 1}}}
        out = self._run(sample_ohlcv_df, cfg, sample_fng_df)
        assert len(out) == len(sample_ohlcv_df)

    def test_original_ohlcv_columns_preserved(self, sample_ohlcv_df, sample_fng_df, tmp_cache_dir):
        cfg = {"external_data": {"cache_dir": tmp_cache_dir,
                                  "fear_greed": {"enabled": True, "shift_periods": 1}}}
        out = self._run(sample_ohlcv_df, cfg, sample_fng_df)
        for col in ["date", "open", "high", "low", "close", "volume"]:
            assert col in out.columns


# ──────────────────────────────────────────────────────────────────────────────
# 6. 真实网络测试（需联网，默认跳过）
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.network
class TestFearGreedLive:
    def test_live_fetch_returns_data(self):
        from agent_market.freqai.external_data.providers import fetch_fear_greed
        df = fetch_fear_greed()
        assert len(df) > 100, "历史数据应超过 100 条"
        assert df["fng_value"].between(0, 100).all()
        assert df["timestamp"].is_monotonic_increasing

    def test_live_fetch_recent_date(self):
        from agent_market.freqai.external_data.providers import fetch_fear_greed
        df = fetch_fear_greed()
        latest = df["timestamp"].max()
        days_ago = (pd.Timestamp.now(tz="UTC") - latest).days
        assert days_ago <= 2, f"最新数据距今 {days_ago} 天，应 ≤ 2 天"
