"""US stock OHLCV + sector data loader (yfinance backend with retry + parquet cache).

Used by local_sim and anti_overfit modules. Caches under
artifacts/wq_brain/data/{ohlcv.parquet, sectors.json, _last_update.json}.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lazy imports (yfinance/pandas not strictly needed for unit tests)
def _yfinance():
    import yfinance as yf
    return yf


def _pd():
    import pandas as pd
    return pd


def data_root() -> Path:
    from .paths import wq_brain_root
    return wq_brain_root() / "data"


def ohlcv_cache_path() -> Path:
    return data_root() / "ohlcv.parquet"


def sectors_cache_path() -> Path:
    return data_root() / "sectors.json"


def universe_cache_path() -> Path:
    return data_root() / "universe.txt"


def metadata_path() -> Path:
    return data_root() / "_last_update.json"


# ── Bundled fallback ticker list ────────────────────────────────────────
# Top liquid US stocks — used when no --tickers-file is provided. Spans
# major sectors. ~300 tickers; for full Russell 3000 the user should pass
# --tickers-file pointing at a CSV/TXT with one symbol per line.
_BUNDLED_TICKERS = """
AAPL MSFT GOOGL AMZN NVDA META TSLA BRK-B JPM V UNH XOM JNJ WMT MA PG
HD CVX LLY ABBV BAC PFE KO PEP MRK COST AVGO CSCO ADBE TMO ACN MCD
CRM NKE ORCL AMD QCOM DHR LIN UPS PM TXN HON SBUX UNP LOW BMY IBM CAT
NEE INTC AXP RTX GS DE AMGN T NOW BA SPGI MDT BLK ISRG GE C MS PYPL
INTU ADI MMM CVS LMT TJX SCHW PLD ABT TFC SO MDLZ ZTS DUK CB SYK PNC
USB ADP CME CL EQIX ICE CSX BSX MMC FISV BMRN APD ETN ECL PSX EMR FCX
GD GIS WM HUM ILMN ITW KMB MAR MO NSC OXY PEG ROST SLB STT TGT TRV TSN
VLO WBA WELL WFC AEP AIG ALL AMT ANTM AON APH ATVI ADM AZO BAX BIIB BK
BKR BBY BDX BX BRK.A CAH CARR CCI CHRW CHTR CI CINF CMA CMCSA CMG COF
COP COR CTAS CTSH CTVA D DIS DLR DOW DOV EA EBAY EFX EIX EL EMN EOG
EPD ES ESS EW EXC EXPE F FANG FAST FDX FE FFIV FIS FITB FLEX FMC FRT
FTI FTNT GLW GMAB GPC GPN GPS GRMN HAL HAS HBAN HCA HCP HD HES HIG HII
HOLX HPE HPQ HRB HRL HSIC HST HSY IBM IDXX IEX IFF INCY IP IPG IQV IR
IRM IT JBHT JBL JCI JKHY JNPR K KEY KEYS KHC KIM KLAC KMB KMI KMX KO
KR L LDOS LEN LH LIN LKQ LLY LMT LNC LNT LOW LRCX LUMN LUV LVS LW LYB
LYV MA MAA MAS MCD MCHP MCK MCO MDLZ MDT MET MGM MHK MKC MKTX MLM MMC
MMM MNST MO MOH MOS MPC MPWR MRK MRNA MRO MS MSCI MSFT MSI MTB MTD MU
NCLH NDAQ NEE NEM NFLX NI NKE NLOK NOC NOW NRG NSC NTAP NTRS NUE NVDA
NVR NWL NWSA NXPI O ODFL OKE OMC ON ORCL ORLY OXY PAYC PAYX PCAR PCG
PEAK PEG PEP PFE PFG PG PGR PH PHM PKG PKI PLD PLTR PM PNC PNR PNW POOL
PPG PPL PRGO PRU PSA PSX PTC PVH PWR PXD QCOM QRVO RCL RE REG REGN RF
RHI RJF RL RMD ROK ROL ROP ROST RSG RTN RTX SBNY SBUX SCHW SEDG SEE
SHW SIVB SJM SLB SNA SNPS SO SPG SPGI SRE SRPT STT STX STZ SWK SWKS
SYF SYK SYY T TAP TDG TDY TECH TEL TER TFC TFX TGT TJX TMO TMUS TPR
TRMB TROW TRV TSCO TSLA TSN TT TTWO TWTR TXN TXT TYL UA UAA UAL UDR
UHS ULTA UNH UNP UPS URI USB V VFC VIAC VLO VMC VNO VRSK VRSN VRTX
VTR VTRS VZ WAB WAT WBA WBD WDC WEC WELL WFC WHR WLTW WM WMB WMT WRB
WRK WST WY WYNN XEL XOM XRAY XYL YUM ZBH ZBRA ZION ZTS
""".split()


def bundled_tickers() -> list[str]:
    """Top US-stock fallback list (~300 names). Used when no file given."""
    seen: set[str] = set()
    out: list[str] = []
    for t in _BUNDLED_TICKERS:
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def load_tickers(file_path: Optional[Path] = None) -> list[str]:
    """Load tickers from file (one per line) or fall back to bundled list."""
    if file_path is None:
        return bundled_tickers()
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"tickers file not found: {p}")
    out: list[str] = []
    seen: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        t = line.strip().upper().split(",")[0]  # tolerate CSV
        if not t or t.startswith("#") or t in seen:
            continue
        out.append(t)
        seen.add(t)
    return out


# ── OHLCV fetch with retry ──────────────────────────────────────────────

def _fetch_batch(
    tickers: list[str],
    start: str,
    end: str,
    *,
    max_retries: int = 4,
    polite_sleep: float = 1.5,
) -> dict[str, Any]:
    """Fetch one batch via yfinance.download with retry on transient errors.

    Returns dict mapping ticker → DataFrame (date-indexed, columns:
    Open/High/Low/Close/Volume). Tickers that fail every retry are absent.
    """
    yf = _yfinance()
    pd = _pd()

    delays = [0, 5, 15, 60, 180]
    for attempt, delay in enumerate(delays[: max_retries + 1]):
        if delay:
            logger.info("Retry %d after %ds for batch [%s..%s]", attempt, delay,
                        tickers[0], tickers[-1])
            time.sleep(delay)
        try:
            df = yf.download(
                tickers,
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception as exc:
            logger.warning("yfinance.download raised: %s", exc)
            continue

        if df is None or df.empty:
            continue

        out: dict[str, Any] = {}
        if isinstance(df.columns, pd.MultiIndex):
            for t in tickers:
                if t in df.columns.get_level_values(0):
                    sub = df[t].dropna(how="all")
                    if not sub.empty:
                        out[t] = sub
        else:
            # Single ticker — yfinance flattens columns
            if not df.dropna(how="all").empty and len(tickers) == 1:
                out[tickers[0]] = df.dropna(how="all")

        if out:
            time.sleep(polite_sleep)
            return out

    logger.warning("Batch fetch failed completely: %s..%s", tickers[0], tickers[-1])
    return {}


def fetch_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    *,
    batch_size: int = 50,
    cache_path: Optional[Path] = None,
) -> Any:
    """Fetch OHLCV for a list of tickers. Returns a long-format DataFrame.

    Schema: index unique on (date, ticker); columns open/high/low/close/volume.

    Uses cache_path (default: artifacts/wq_brain/data/ohlcv.parquet) — only
    fetches tickers / date ranges not already cached.
    """
    pd = _pd()
    cache_path = Path(cache_path) if cache_path else ohlcv_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
    else:
        cached = pd.DataFrame()

    cached_tickers = set(cached.index.get_level_values("ticker").unique()) if not cached.empty else set()
    missing = [t for t in tickers if t not in cached_tickers]

    if missing:
        logger.info("Fetching %d new tickers in batches of %d", len(missing), batch_size)
        new_rows: list[Any] = []
        for i in range(0, len(missing), batch_size):
            batch = missing[i: i + batch_size]
            res = _fetch_batch(batch, start, end)
            for t, df in res.items():
                df = df.rename(columns=str.lower)
                df["ticker"] = t
                df.index.name = "date"
                df = df.reset_index().set_index(["date", "ticker"])
                new_rows.append(df[["open", "high", "low", "close", "volume"]])
            logger.info("Batch %d/%d: fetched %d/%d", i // batch_size + 1,
                        (len(missing) + batch_size - 1) // batch_size,
                        len(res), len(batch))

        if new_rows:
            new_df = pd.concat(new_rows)
            cached = pd.concat([cached, new_df]) if not cached.empty else new_df
            cached = cached[~cached.index.duplicated(keep="last")]
            cached = cached.sort_index()
            cached.to_parquet(cache_path)
            logger.info("Wrote %d total rows to %s", len(cached), cache_path)

    # Filter to requested tickers + date range
    if cached.empty:
        return cached
    mask = cached.index.get_level_values("ticker").isin(tickers)
    sub = cached.loc[mask]
    sub = sub.loc[(sub.index.get_level_values("date") >= start)
                  & (sub.index.get_level_values("date") <= end)]
    return sub


# ── Sector lookup ───────────────────────────────────────────────────────

def fetch_sectors(
    tickers: list[str],
    *,
    cache_path: Optional[Path] = None,
    polite_sleep: float = 0.3,
) -> dict[str, str]:
    """Fetch sector metadata per ticker. Slow (~1s each) but cached, so once-only.

    Returns dict {ticker: sector}. Empty string for tickers without metadata.
    """
    yf = _yfinance()
    cache_path = Path(cache_path) if cache_path else sectors_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache: dict[str, str] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    missing = [t for t in tickers if t not in cache]
    if missing:
        logger.info("Looking up sector for %d new tickers", len(missing))
        for i, t in enumerate(missing):
            try:
                info = yf.Ticker(t).info or {}
                cache[t] = info.get("sector", "") or ""
            except Exception as exc:
                logger.warning("sector lookup failed for %s: %s", t, exc)
                cache[t] = ""
            if (i + 1) % 50 == 0:
                cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
                logger.info("Sector progress: %d/%d", i + 1, len(missing))
            time.sleep(polite_sleep)

        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    return {t: cache.get(t, "") for t in tickers}


# ── Top-level orchestration ─────────────────────────────────────────────

def fetch_data(
    tickers: list[str],
    start: str,
    end: str,
    *,
    skip_sectors: bool = False,
) -> dict[str, Any]:
    """Top-level: fetch OHLCV + sectors + write metadata. Returns summary dict."""
    t0 = time.time()
    pd = _pd()
    ohlcv = fetch_ohlcv(tickers, start, end)
    n_rows = len(ohlcv) if ohlcv is not None else 0
    actual_tickers = (
        sorted(ohlcv.index.get_level_values("ticker").unique().tolist())
        if n_rows else []
    )

    sectors: dict[str, str] = {}
    if not skip_sectors:
        sectors = fetch_sectors(actual_tickers or tickers)

    universe_cache_path().write_text("\n".join(actual_tickers or tickers) + "\n",
                                     encoding="utf-8")
    summary = {
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
        "ticker_count": len(actual_tickers),
        "row_count": n_rows,
        "sector_count": sum(1 for s in sectors.values() if s),
        "start": start,
        "end": end,
        "ohlcv_path": str(ohlcv_cache_path()),
        "sectors_path": str(sectors_cache_path()),
    }
    metadata_path().write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_cached_ohlcv() -> Any:
    """Load cached OHLCV; returns empty DataFrame if missing."""
    pd = _pd()
    p = ohlcv_cache_path()
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def load_cached_sectors() -> dict[str, str]:
    p = sectors_cache_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
