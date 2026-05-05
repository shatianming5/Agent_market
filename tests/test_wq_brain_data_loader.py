"""data_loader tests (no network — uses tmp paths + bundled list)."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agent_market.wq_brain.data_loader import (
    StooqAPIKeyRequired,
    _fetch_stooq_one,
    _resolve_backend,
    _stooq_symbol,
    bundled_tickers,
    load_tickers,
    metadata_path,
    ohlcv_cache_path,
    sectors_cache_path,
)


def test_bundled_tickers_includes_majors():
    tickers = bundled_tickers()
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "GOOGL" in tickers
    assert len(tickers) >= 200


def test_bundled_tickers_no_duplicates():
    tickers = bundled_tickers()
    assert len(tickers) == len(set(tickers))


def test_load_tickers_from_file(tmp_path: Path):
    f = tmp_path / "tickers.txt"
    f.write_text("AAPL\nMSFT\n# comment\nGOOGL\n", encoding="utf-8")
    tickers = load_tickers(f)
    assert tickers == ["AAPL", "MSFT", "GOOGL"]


def test_load_tickers_skips_blank_and_comment(tmp_path: Path):
    f = tmp_path / "tickers.txt"
    f.write_text("\n# header\nAAPL\n\nMSFT\n# trailing\n", encoding="utf-8")
    assert load_tickers(f) == ["AAPL", "MSFT"]


def test_load_tickers_csv_takes_first_column(tmp_path: Path):
    f = tmp_path / "tickers.csv"
    f.write_text("AAPL,Apple Inc.\nMSFT,Microsoft Corp.\n", encoding="utf-8")
    assert load_tickers(f) == ["AAPL", "MSFT"]


def test_load_tickers_falls_back_to_bundled_when_none():
    out = load_tickers(None)
    assert "AAPL" in out


def test_load_tickers_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_tickers(tmp_path / "nonexistent.txt")


def test_cache_paths_under_data_root(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    p = ohlcv_cache_path()
    assert p.parent.name == "data"
    assert "wq_brain" in str(p)
    assert sectors_cache_path().name == "sectors.json"
    assert metadata_path().name == "_last_update.json"


# ── Stooq backend ────────────────────────────────────────────────────────


def test_stooq_symbol_lowercases_and_appends_us():
    assert _stooq_symbol("AAPL") == "aapl.us"
    assert _stooq_symbol("MSFT") == "msft.us"


def test_stooq_symbol_replaces_dot_with_dash():
    # Stooq uses dash not dot for share-class suffixes (BRK.B → brk-b.us)
    assert _stooq_symbol("BRK.B") == "brk-b.us"
    assert _stooq_symbol("BRK-B") == "brk-b.us"


def test_resolve_backend_default_is_stooq(monkeypatch):
    monkeypatch.delenv("WQB_DATA_BACKEND", raising=False)
    assert _resolve_backend(None) == "stooq"


def test_resolve_backend_honors_env(monkeypatch):
    monkeypatch.setenv("WQB_DATA_BACKEND", "yfinance")
    assert _resolve_backend(None) == "yfinance"


def test_resolve_backend_kwarg_wins_over_env(monkeypatch):
    monkeypatch.setenv("WQB_DATA_BACKEND", "yfinance")
    assert _resolve_backend("auto") == "auto"


def test_resolve_backend_rejects_unknown(monkeypatch):
    with pytest.raises(ValueError, match="WQB_DATA_BACKEND"):
        _resolve_backend("invalid")


class _FakeResp:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


_STOOQ_OK_CSV = (
    b"Date,Open,High,Low,Close,Volume\n"
    b"2024-01-02,150.0,151.5,149.5,151.0,1000000\n"
    b"2024-01-03,151.0,152.0,150.5,151.5,1100000\n"
    b"2024-01-04,151.5,153.0,151.0,152.8,1250000\n"
)


def test_fetch_stooq_one_parses_csv():
    pd = pytest.importorskip("pandas")
    with patch("urllib.request.urlopen", return_value=_FakeResp(_STOOQ_OK_CSV)):
        df = _fetch_stooq_one("AAPL", "2024-01-01", "2024-01-31")
    assert df is not None
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 3
    assert df.index[0] == pd.Timestamp("2024-01-02")
    assert df.iloc[0]["close"] == pytest.approx(151.0)
    assert df.iloc[-1]["volume"] == pytest.approx(1250000)


def test_fetch_stooq_one_returns_none_on_no_data():
    with patch("urllib.request.urlopen", return_value=_FakeResp(b"No data")):
        assert _fetch_stooq_one("ZZZZ", "2024-01-01", "2024-01-31") is None


def test_fetch_stooq_one_returns_none_on_html_error():
    with patch(
        "urllib.request.urlopen",
        return_value=_FakeResp(b"<html><body>error</body></html>"),
    ):
        assert _fetch_stooq_one("AAPL", "2024-01-01", "2024-01-31") is None


def test_fetch_stooq_one_url_uses_compact_dates(monkeypatch):
    captured: dict[str, Any] = {}

    def _spy(req, timeout: float = 30.0):
        captured["url"] = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        return _FakeResp(_STOOQ_OK_CSV)

    monkeypatch.setattr("urllib.request.urlopen", _spy)
    _fetch_stooq_one("AAPL", "2024-01-02", "2024-12-31")
    url = captured["url"]
    assert "s=aapl.us" in url
    assert "i=d" in url
    # dates flatten to YYYYMMDD
    assert "d1=20240102" in url
    assert "d2=20241231" in url


def test_fetch_stooq_one_appends_apikey_when_set(monkeypatch):
    captured: dict[str, Any] = {}

    def _spy(req, timeout: float = 30.0):
        captured["url"] = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        return _FakeResp(_STOOQ_OK_CSV)

    monkeypatch.setenv("STOOQ_APIKEY", "abc123")
    monkeypatch.setattr("urllib.request.urlopen", _spy)
    _fetch_stooq_one("AAPL", "2024-01-02", "2024-12-31")
    assert "apikey=abc123" in captured["url"]


def test_fetch_stooq_one_omits_apikey_when_unset(monkeypatch):
    captured: dict[str, Any] = {}

    def _spy(req, timeout: float = 30.0):
        captured["url"] = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        return _FakeResp(_STOOQ_OK_CSV)

    monkeypatch.delenv("STOOQ_APIKEY", raising=False)
    monkeypatch.setattr("urllib.request.urlopen", _spy)
    _fetch_stooq_one("AAPL", "2024-01-02", "2024-12-31")
    assert "apikey" not in captured["url"]


_STOOQ_APIKEY_PROMPT = (
    b"Get your apikey:\n\n"
    b"1. Open https://stooq.com/q/d/?s=aapl.us&get_apikey\n"
    b"2. Enter the captcha code.\n"
)


def test_fetch_stooq_one_raises_on_apikey_prompt(monkeypatch):
    monkeypatch.delenv("STOOQ_APIKEY", raising=False)
    with patch("urllib.request.urlopen", return_value=_FakeResp(_STOOQ_APIKEY_PROMPT)):
        with pytest.raises(StooqAPIKeyRequired):
            _fetch_stooq_one("AAPL", "2024-01-01", "2024-01-31")


def test_fetch_stooq_one_raises_on_quota_exhausted(monkeypatch):
    body = b"Exceeded the daily hits limit. Try again tomorrow."
    monkeypatch.setenv("STOOQ_APIKEY", "abc")
    with patch("urllib.request.urlopen", return_value=_FakeResp(body)):
        with pytest.raises(RuntimeError, match="quota exhausted"):
            _fetch_stooq_one("AAPL", "2024-01-01", "2024-01-31")
