"""Kaggle dataset backend for OHLCV cache.

Designed for the case where yfinance is rate-limited / Stooq wants an apikey
but Kaggle has a community-maintained dataset that we can pull whole, in one
HTTP roundtrip. The Kaggle API is authenticated with HTTP Basic auth using a
``kaggle.json`` credential file (see https://www.kaggle.com/docs/api).

Typical flow:

    creds = kaggle_credentials()                       # read ~/.kaggle/kaggle.json
    zip_path = download_kaggle_dataset(slug, dest)     # one HTTP GET, ~tens of MB
    extract_dir = extract_kaggle_zip(zip_path, dest)
    summary = import_kaggle_to_cache(                  # merge into ohlcv.parquet
        extract_dir,
        files_glob="*.csv",
        ticker_col="Ticker",
        date_col="Date",
    )

The import step keeps ``DataFrame.rename`` flexible so different Kaggle
datasets (one big CSV vs. one file per ticker) can both flow through.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _pd() -> Any:
    import pandas as pd
    return pd


KAGGLE_API_BASE = "https://www.kaggle.com/api/v1"


class KaggleCredentialsError(RuntimeError):
    """Raised when ``kaggle.json`` cannot be located or parsed."""


def kaggle_credentials(
    *,
    config_dir: Optional[Path] = None,
) -> tuple[str, str]:
    """Return ``(username, key)`` from ``KAGGLE_USERNAME``/``KAGGLE_KEY`` env or
    ``$KAGGLE_CONFIG_DIR/kaggle.json`` (defaults to ``~/.kaggle/kaggle.json``).

    Raises :class:`KaggleCredentialsError` with a friendly message pointing to
    the Kaggle account-token UI.
    """
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return env_user, env_key

    cfg_dir = config_dir or Path(os.environ.get("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")
    cfg = Path(cfg_dir) / "kaggle.json"
    if not cfg.exists():
        raise KaggleCredentialsError(
            f"Kaggle credentials not found at {cfg}. Visit "
            "https://www.kaggle.com/settings/account → Create New API Token, "
            "place the downloaded kaggle.json at ~/.kaggle/kaggle.json (chmod 600), "
            "or export KAGGLE_USERNAME / KAGGLE_KEY."
        )
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KaggleCredentialsError(f"Failed to read {cfg}: {exc}") from exc
    user, key = data.get("username"), data.get("key")
    if not user or not key:
        raise KaggleCredentialsError(f"{cfg} missing 'username' or 'key' fields")
    return user, key


def _basic_auth_header(username: str, key: str) -> str:
    raw = f"{username}:{key}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def download_kaggle_dataset(
    slug: str,
    dest_dir: Path,
    *,
    credentials: Optional[tuple[str, str]] = None,
    force: bool = False,
    timeout: float = 300.0,
) -> Path:
    """Download ``owner/dataset`` to ``dest_dir/dataset.zip``.

    Returns the ZIP path. If the file already exists and ``force=False``,
    skips the network call.
    """
    if "/" not in slug:
        raise ValueError(f"Kaggle slug must be 'owner/dataset', got {slug!r}")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = slug.replace("/", "__") + ".zip"
    zip_path = dest_dir / safe_name
    if zip_path.exists() and not force:
        logger.info("Kaggle ZIP already present, skipping download: %s", zip_path)
        return zip_path

    user, key = credentials or kaggle_credentials()
    url = f"{KAGGLE_API_BASE}/datasets/download/{slug}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": _basic_auth_header(user, key),
            "User-Agent": "agent-market-wq-brain/1.0",
        },
    )
    logger.info("Downloading Kaggle dataset %s …", slug)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(zip_path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 64)
                if not chunk:
                    break
                out.write(chunk)
    except urllib.error.HTTPError as exc:
        zip_path.unlink(missing_ok=True)
        if exc.code in (401, 403):
            raise KaggleCredentialsError(
                f"Kaggle rejected credentials (HTTP {exc.code}). Re-issue token "
                "at https://www.kaggle.com/settings/account."
            ) from exc
        raise
    return zip_path


def extract_kaggle_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Extract the dataset ZIP into ``dest_dir/<dataset_stem>/`` and return it."""
    zip_path = Path(zip_path)
    extract_dir = Path(dest_dir) / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


# ── Cache import ────────────────────────────────────────────────────────

# Common column aliases seen in Kaggle stock datasets, normalized to our schema
_DEFAULT_COLUMN_MAP = {
    "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume",
    "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    "OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close", "VOLUME": "volume",
    "Adj Close": "close", "adj_close": "close",  # prefer adjusted close as our `close`
}

_OUR_COLUMNS = ("open", "high", "low", "close", "volume")


def _read_one_kaggle_csv(
    path: Path,
    *,
    pd: Any,
    ticker_col: Optional[str],
    date_col: str,
    column_map: dict[str, str],
    ticker_from_filename: bool,
) -> Any:
    """Read a single CSV from a Kaggle dump and normalize to our long-format
    schema (index=(date, ticker), columns=open/high/low/close/volume)."""
    df = pd.read_csv(path)
    if df.empty:
        return None
    rename = {src: dst for src, dst in column_map.items() if src in df.columns}
    df = df.rename(columns=rename)
    if date_col not in df.columns:
        raise ValueError(f"{path.name}: missing date column {date_col!r}; got {list(df.columns)[:8]}")
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])
    if ticker_from_filename:
        ticker = path.stem.upper().split(".")[0]
        df["ticker"] = ticker
    elif ticker_col and ticker_col in df.columns:
        df["ticker"] = df[ticker_col].astype(str).str.upper()
    else:
        raise ValueError(
            f"{path.name}: cannot derive ticker — neither ticker_from_filename nor "
            f"a ticker_col {ticker_col!r} present in {list(df.columns)[:8]}"
        )
    keep = ["date", "ticker"] + [c for c in _OUR_COLUMNS if c in df.columns]
    df = df[keep].set_index(["date", "ticker"]).sort_index()
    return df


def import_kaggle_to_cache(
    extract_dir: Path,
    *,
    cache_path: Optional[Path] = None,
    files_glob: str = "*.csv",
    ticker_col: Optional[str] = "Ticker",
    date_col: str = "Date",
    column_map: Optional[dict[str, str]] = None,
    ticker_from_filename: bool = False,
) -> dict[str, Any]:
    """Walk extracted dataset, parse each matching CSV, merge into ohlcv.parquet.

    Returns a summary dict with file/row counts. Raises if no usable CSV is
    found (typo in glob, layered subdirs, etc.) so the caller can fix the args.
    """
    pd = _pd()
    from .data_loader import ohlcv_cache_path

    extract_dir = Path(extract_dir)
    if cache_path is None:
        cache_path = ohlcv_cache_path()
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in extract_dir.rglob(files_glob) if p.is_file())
    if not files:
        raise FileNotFoundError(
            f"No files matching {files_glob!r} under {extract_dir}; check the dataset layout."
        )
    cmap = {**_DEFAULT_COLUMN_MAP, **(column_map or {})}

    frames: list[Any] = []
    n_files_ok = 0
    for p in files:
        try:
            df = _read_one_kaggle_csv(
                p, pd=pd,
                ticker_col=ticker_col,
                date_col=date_col,
                column_map=cmap,
                ticker_from_filename=ticker_from_filename,
            )
        except Exception as exc:
            logger.warning("Skip %s: %s", p.name, exc)
            continue
        if df is None or df.empty:
            continue
        frames.append(df)
        n_files_ok += 1

    if not frames:
        raise RuntimeError(
            f"Parsed {len(files)} files but produced 0 rows — check ticker_col / date_col / "
            f"ticker_from_filename for this dataset."
        )

    merged = pd.concat(frames)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
        merged = pd.concat([existing, merged])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_parquet(cache_path)

    return {
        "ok": True,
        "files_total": len(files),
        "files_imported": n_files_ok,
        "rows_total": int(len(merged)),
        "tickers_total": int(merged.index.get_level_values("ticker").nunique()),
        "cache_path": str(cache_path),
    }
