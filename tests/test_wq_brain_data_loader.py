"""data_loader tests (no network — uses tmp paths + bundled list)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_market.wq_brain.data_loader import (
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
