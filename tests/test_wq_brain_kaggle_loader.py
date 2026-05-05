"""Kaggle loader tests — credential lookup, auth header, ZIP I/O, CSV import."""
from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agent_market.wq_brain.kaggle_loader import (
    KaggleCredentialsError,
    _basic_auth_header,
    download_kaggle_dataset,
    extract_kaggle_zip,
    import_kaggle_to_cache,
    kaggle_credentials,
)


# ── Credentials ──────────────────────────────────────────────────────────


def test_kaggle_credentials_from_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "alice")
    monkeypatch.setenv("KAGGLE_KEY", "secret123")
    assert kaggle_credentials() == ("alice", "secret123")


def test_kaggle_credentials_from_file(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    cfg = tmp_path / "kaggle.json"
    cfg.write_text(json.dumps({"username": "bob", "key": "xyz"}), encoding="utf-8")
    assert kaggle_credentials(config_dir=tmp_path) == ("bob", "xyz")


def test_kaggle_credentials_env_wins_over_file(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KAGGLE_USERNAME", "envuser")
    monkeypatch.setenv("KAGGLE_KEY", "envkey")
    cfg = tmp_path / "kaggle.json"
    cfg.write_text(json.dumps({"username": "fileuser", "key": "filekey"}), encoding="utf-8")
    assert kaggle_credentials(config_dir=tmp_path) == ("envuser", "envkey")


def test_kaggle_credentials_missing_raises(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    with pytest.raises(KaggleCredentialsError, match="kaggle.json"):
        kaggle_credentials(config_dir=tmp_path)


def test_kaggle_credentials_malformed_raises(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    cfg = tmp_path / "kaggle.json"
    cfg.write_text("{not json", encoding="utf-8")
    with pytest.raises(KaggleCredentialsError):
        kaggle_credentials(config_dir=tmp_path)


def test_kaggle_credentials_missing_field_raises(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    cfg = tmp_path / "kaggle.json"
    cfg.write_text(json.dumps({"username": "alice"}), encoding="utf-8")  # no key
    with pytest.raises(KaggleCredentialsError, match="missing"):
        kaggle_credentials(config_dir=tmp_path)


# ── Basic auth header ────────────────────────────────────────────────────


def test_basic_auth_header_uses_base64():
    header = _basic_auth_header("alice", "secret")
    assert header.startswith("Basic ")
    decoded = base64.b64decode(header.removeprefix("Basic ")).decode("ascii")
    assert decoded == "alice:secret"


# ── Download ─────────────────────────────────────────────────────────────


class _FakeBinaryResp:
    def __init__(self, body: bytes) -> None:
        self._buf = io.BytesIO(body)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n if n >= 0 else None)

    def __enter__(self) -> "_FakeBinaryResp":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def test_download_kaggle_dataset_writes_zip(tmp_path: Path, monkeypatch):
    payload = b"PK\x05\x06" + b"\x00" * 18  # minimal empty-zip-ish bytes
    captured: dict[str, Any] = {}

    def _spy(req, timeout: float = 300.0):
        captured["url"] = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        captured["auth"] = req.headers.get("Authorization")
        return _FakeBinaryResp(payload)

    monkeypatch.setattr("urllib.request.urlopen", _spy)
    out = download_kaggle_dataset("alice/cool", tmp_path, credentials=("alice", "key123"))
    assert out.exists()
    assert out.read_bytes() == payload
    assert captured["url"].endswith("/datasets/download/alice/cool")
    assert captured["auth"].startswith("Basic ")


def test_download_kaggle_dataset_skips_when_present(tmp_path: Path, monkeypatch):
    existing = tmp_path / "alice__cool.zip"
    existing.write_bytes(b"already-here")
    called = {"n": 0}

    def _spy(*_a: Any, **_k: Any):
        called["n"] += 1
        return _FakeBinaryResp(b"new")

    monkeypatch.setattr("urllib.request.urlopen", _spy)
    out = download_kaggle_dataset("alice/cool", tmp_path, credentials=("alice", "k"))
    assert called["n"] == 0
    assert out.read_bytes() == b"already-here"


def test_download_kaggle_dataset_force_overwrites(tmp_path: Path, monkeypatch):
    existing = tmp_path / "alice__cool.zip"
    existing.write_bytes(b"old")

    def _spy(*_a: Any, **_k: Any):
        return _FakeBinaryResp(b"fresh-bytes")

    monkeypatch.setattr("urllib.request.urlopen", _spy)
    out = download_kaggle_dataset("alice/cool", tmp_path, credentials=("a", "k"), force=True)
    assert out.read_bytes() == b"fresh-bytes"


def test_download_kaggle_dataset_rejects_bad_slug(tmp_path: Path):
    with pytest.raises(ValueError, match="owner/dataset"):
        download_kaggle_dataset("just-a-slug", tmp_path, credentials=("a", "k"))


def test_download_kaggle_dataset_401_translates_to_credentials_error(tmp_path: Path, monkeypatch):
    import urllib.error

    def _raise_401(*_a: Any, **_k: Any):
        raise urllib.error.HTTPError(
            "https://kaggle/x", 401, "Unauthorized", {}, io.BytesIO(b"")
        )

    monkeypatch.setattr("urllib.request.urlopen", _raise_401)
    with pytest.raises(KaggleCredentialsError, match="HTTP 401"):
        download_kaggle_dataset("alice/cool", tmp_path, credentials=("a", "k"))


# ── ZIP extraction ───────────────────────────────────────────────────────


def _make_zip(tmp_path: Path, files: dict[str, bytes]) -> Path:
    zp = tmp_path / "data.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        for name, body in files.items():
            zf.writestr(name, body)
    return zp


def test_extract_kaggle_zip_creates_named_dir(tmp_path: Path):
    zp = _make_zip(tmp_path, {"a.csv": b"hi"})
    out = extract_kaggle_zip(zp, tmp_path / "out")
    assert out.is_dir()
    assert out.name == "data"
    assert (out / "a.csv").read_bytes() == b"hi"


# ── Import to cache ──────────────────────────────────────────────────────


def _world_stock_csv() -> bytes:
    # Mirrors nelgiriyewithana/world-stock-prices-daily-updating shape:
    # one big CSV with a Ticker column and Date / Open / High / Low / Close / Volume.
    return (
        b"Date,Open,High,Low,Close,Volume,Ticker\n"
        b"2024-01-02,150.0,151.5,149.5,151.0,1000000,AAPL\n"
        b"2024-01-03,151.0,152.0,150.5,151.5,1100000,AAPL\n"
        b"2024-01-02,300.0,302.0,299.5,301.0,500000,MSFT\n"
        b"2024-01-03,301.0,303.0,300.5,302.5,540000,MSFT\n"
    )


def _per_ticker_csvs() -> dict[str, bytes]:
    # Mirrors williecosta/russell-3000-stock-history shape: one file per ticker,
    # filename=ticker, no Ticker column inside.
    aapl = (
        b"Date,Open,High,Low,Close,Volume\n"
        b"2024-01-02,150.0,151.5,149.5,151.0,1000000\n"
        b"2024-01-03,151.0,152.0,150.5,151.5,1100000\n"
    )
    msft = (
        b"Date,Open,High,Low,Close,Volume\n"
        b"2024-01-02,300.0,302.0,299.5,301.0,500000\n"
    )
    return {"AAPL.csv": aapl, "MSFT.csv": msft}


def test_import_kaggle_to_cache_one_big_csv(tmp_path: Path):
    pytest.importorskip("pandas")
    pd = __import__("pandas")
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    (extract_dir / "world_stocks.csv").write_bytes(_world_stock_csv())
    cache = tmp_path / "ohlcv.parquet"

    summary = import_kaggle_to_cache(
        extract_dir,
        cache_path=cache,
        files_glob="*.csv",
        ticker_col="Ticker",
        date_col="Date",
    )
    assert summary["ok"] is True
    assert summary["files_imported"] == 1
    assert summary["tickers_total"] == 2
    df = pd.read_parquet(cache)
    assert len(df) == 4
    assert set(df.index.get_level_values("ticker").unique()) == {"AAPL", "MSFT"}
    assert df.loc[(pd.Timestamp("2024-01-02"), "AAPL")]["close"] == pytest.approx(151.0)


def test_import_kaggle_to_cache_one_csv_per_ticker(tmp_path: Path):
    pytest.importorskip("pandas")
    pd = __import__("pandas")
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    for name, body in _per_ticker_csvs().items():
        (extract_dir / name).write_bytes(body)
    cache = tmp_path / "ohlcv.parquet"

    summary = import_kaggle_to_cache(
        extract_dir,
        cache_path=cache,
        files_glob="*.csv",
        ticker_col=None,
        date_col="Date",
        ticker_from_filename=True,
    )
    assert summary["files_imported"] == 2
    assert summary["tickers_total"] == 2
    df = pd.read_parquet(cache)
    assert {"AAPL", "MSFT"}.issubset(df.index.get_level_values("ticker").unique())


def test_import_kaggle_to_cache_merges_existing_cache(tmp_path: Path):
    pytest.importorskip("pandas")
    pd = __import__("pandas")
    cache = tmp_path / "ohlcv.parquet"
    # Pre-existing cache row for AAPL on a *different* date — must survive merge
    seed = pd.DataFrame(
        [{"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 5_000.0}],
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2023-12-29"), "AAPL")], names=["date", "ticker"],
        ),
    )
    seed.to_parquet(cache)

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    (extract_dir / "world.csv").write_bytes(_world_stock_csv())
    summary = import_kaggle_to_cache(
        extract_dir,
        cache_path=cache,
        files_glob="*.csv",
        ticker_col="Ticker",
        date_col="Date",
    )
    df = pd.read_parquet(cache)
    assert len(df) == 5  # 1 seeded + 4 imported
    assert (pd.Timestamp("2023-12-29"), "AAPL") in df.index


def test_import_kaggle_to_cache_raises_when_glob_matches_nothing(tmp_path: Path):
    pytest.importorskip("pandas")
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    cache = tmp_path / "ohlcv.parquet"
    with pytest.raises(FileNotFoundError, match="files matching"):
        import_kaggle_to_cache(extract_dir, cache_path=cache, files_glob="*.parquet")


def test_import_kaggle_to_cache_raises_when_zero_rows_parsed(tmp_path: Path):
    pytest.importorskip("pandas")
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    # CSV exists but missing required columns → all rows skipped
    (extract_dir / "junk.csv").write_bytes(b"foo,bar\n1,2\n")
    cache = tmp_path / "ohlcv.parquet"
    with pytest.raises(RuntimeError, match="0 rows"):
        import_kaggle_to_cache(
            extract_dir, cache_path=cache, files_glob="*.csv",
            ticker_col="Ticker", date_col="Date",
        )
