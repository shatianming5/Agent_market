"""US stock OHLCV + sector data loader.

Two backends, selected at fetch time by env var ``WQB_DATA_BACKEND`` or kwarg:
  - ``stooq`` (default, free, no API key) — pulls CSV from
    ``https://stooq.com/q/d/l/?s=<ticker>.us&i=d&d1=YYYYMMDD&d2=YYYYMMDD``
    Per-ticker CSV endpoint is NOT behind the CAPTCHA (only the bulk ZIP at
    /db/h is). Polite ~3-5s between calls; works at scale where yfinance fails.
  - ``yfinance`` — kept as fallback when stooq returns no data for a symbol.
    Use ``WQB_DATA_BACKEND=yfinance`` to force.

Caches under artifacts/wq_brain/data/{ohlcv.parquet, sectors.json,
_last_update.json}. Used by local_sim and anti_overfit modules.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import time
import urllib.error
import urllib.parse
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

def _make_yf_session():
    """Return a curl_cffi browser-fingerprint session if available, else None.

    Yahoo aggressively rate-limits Python's default requests session by user-agent
    fingerprint. curl_cffi's `impersonate="chrome120"` mimics a real browser TLS +
    HTTP/2 fingerprint and bypasses most anti-bot measures.
    """
    try:
        from curl_cffi import requests as cffi_req
        return cffi_req.Session(impersonate="chrome120")
    except ImportError:
        logger.warning("curl_cffi not available — falling back to default session "
                       "(yfinance rate-limit risk is high)")
        return None


def _fetch_one_ticker(
    ticker: str,
    start: str,
    end: str,
    *,
    session: Any = None,
    max_retries: int = 3,
) -> Any:
    """Fetch a SINGLE ticker via yfinance.Ticker(t).history() — more reliable
    than batch yf.download() under rate limits."""
    yf = _yfinance()
    pd = _pd()
    for attempt, delay in enumerate([0, 10, 30, 90][: max_retries + 1]):
        if delay:
            time.sleep(delay)
        try:
            t = yf.Ticker(ticker, session=session) if session else yf.Ticker(ticker)
            df = t.history(start=start, end=end, auto_adjust=True, raise_errors=False)
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            logger.debug("Ticker %s attempt %d failed: %s", ticker, attempt, exc)
    return None


# ── Stooq backend (default) ─────────────────────────────────────────────

_STOOQ_BASE = "https://stooq.com/q/d/l/"
_STOOQ_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


class StooqAPIKeyRequired(RuntimeError):
    """Raised when Stooq CSV endpoint demands an apikey we don't have set.

    Pass the apikey via ``STOOQ_APIKEY`` env var. Get one (free) by visiting
    ``https://stooq.com/q/d/?s=aapl.us&get_apikey`` in a browser, solving the
    CAPTCHA once, and copying the apikey from the resulting CSV download URL.
    """


def _stooq_symbol(ticker: str) -> str:
    """Map our ticker (e.g. ``BRK-B``, ``BRK.A``) to Stooq's lower-case form."""
    return ticker.lower().replace(".", "-") + ".us"


def _fetch_stooq_one(
    ticker: str,
    start: str,
    end: str,
    *,
    max_retries: int = 3,
    timeout: float = 30.0,
    apikey: Optional[str] = None,
) -> Any:
    """Fetch a single US ticker's daily OHLCV from Stooq's CSV endpoint.

    Returns a pandas DataFrame indexed by date with columns
    open/high/low/close/volume, or ``None`` if no data. Uses urllib (no
    third-party dependency).

    Stooq requires an apikey for CSV downloads as of 2025+. Pass via the
    ``apikey`` kwarg or ``STOOQ_APIKEY`` env var. Without it, raises
    :class:`StooqAPIKeyRequired` so callers can stop fast and prompt the user.
    """
    pd = _pd()
    sym = _stooq_symbol(ticker)
    d1 = start.replace("-", "")
    d2 = end.replace("-", "")
    params: dict[str, str] = {"s": sym, "i": "d", "d1": d1, "d2": d2}
    key = apikey if apikey is not None else os.environ.get("STOOQ_APIKEY", "")
    if key:
        params["apikey"] = key
    url = f"{_STOOQ_BASE}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _STOOQ_UA})

    last_err: Optional[Exception] = None
    for attempt, delay in enumerate([0, 3, 10, 30][: max_retries + 1]):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_err = exc
            continue

        head = body.lstrip()[:200].lower()
        # Apikey-prompt page (we have no key or wrong key) — fail loud
        if "get your apikey" in head or "get_apikey" in head:
            raise StooqAPIKeyRequired(
                "Stooq CSV endpoint requires an apikey. Visit "
                "https://stooq.com/q/d/?s=aapl.us&get_apikey in a browser, "
                "solve the CAPTCHA, copy the apikey, and export it via "
                "STOOQ_APIKEY=<key>."
            )
        # Quota exhausted — distinct from missing apikey
        if "exceeded the daily hits limit" in head:
            raise RuntimeError(
                f"Stooq daily-hit quota exhausted (ticker={ticker}); "
                "rerun tomorrow or supply a different STOOQ_APIKEY"
            )
        # No data for this symbol / range
        if not body or head.startswith("no data") or "<html" in head:
            return None
        # Header row: Date,Open,High,Low,Close,Volume
        rows: list[dict[str, Any]] = []
        reader = csv.DictReader(io.StringIO(body))
        for r in reader:
            d = (r.get("Date") or "").strip()
            if not d:
                continue
            try:
                rows.append({
                    "date": pd.Timestamp(d),
                    "open": float(r["Open"]),
                    "high": float(r["High"]),
                    "low": float(r["Low"]),
                    "close": float(r["Close"]),
                    "volume": float(r.get("Volume") or 0.0),
                })
            except (KeyError, ValueError):
                continue
        if not rows:
            return None
        df = pd.DataFrame(rows).set_index("date").sort_index()
        return df

    if last_err is not None:
        logger.debug("Stooq fetch %s failed after retries: %s", ticker, last_err)
    return None


def _fetch_batch(
    tickers: list[str],
    start: str,
    end: str,
    *,
    max_retries: int = 3,
    polite_sleep: float = 5.0,
    use_single_ticker: bool = True,
    session: Any = None,
) -> dict[str, Any]:
    """Fetch a batch of tickers. By default uses single-ticker mode with
    a polite delay between each call — much more rate-limit tolerant than
    yf.download(batch).

    Returns dict mapping ticker → DataFrame (date-indexed, columns:
    Open/High/Low/Close/Volume). Tickers that fail every retry are absent.
    """
    yf = _yfinance()
    pd = _pd()

    if session is None:
        session = _make_yf_session()

    if use_single_ticker:
        out: dict[str, Any] = {}
        for ticker in tickers:
            df = _fetch_one_ticker(ticker, start, end, session=session, max_retries=max_retries)
            if df is not None:
                out[ticker] = df
            time.sleep(polite_sleep)
        return out

    # Legacy batch mode (kept for compatibility, but rarely succeeds today)
    delays = [0, 10, 30, 90, 300]
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
                threads=False,  # IMPORTANT: thread=True makes rate-limit MUCH worse
                progress=False,
                session=session,
            )
        except Exception as exc:
            logger.warning("yfinance.download raised: %s", exc)
            continue

        if df is None or df.empty:
            continue

        out2: dict[str, Any] = {}
        if isinstance(df.columns, pd.MultiIndex):
            for t in tickers:
                if t in df.columns.get_level_values(0):
                    sub = df[t].dropna(how="all")
                    if not sub.empty:
                        out2[t] = sub
        else:
            if not df.dropna(how="all").empty and len(tickers) == 1:
                out2[tickers[0]] = df.dropna(how="all")

        if out2:
            time.sleep(polite_sleep)
            return out2

    logger.warning("Batch fetch failed completely: %s..%s", tickers[0], tickers[-1])
    return {}


def _resolve_backend(backend: Optional[str]) -> str:
    """Resolve fetch backend: explicit kwarg > env > default ``stooq``."""
    raw = (backend or os.environ.get("WQB_DATA_BACKEND") or "stooq").lower().strip()
    if raw not in ("stooq", "yfinance", "auto"):
        raise ValueError(f"unknown WQB_DATA_BACKEND={raw!r}; want stooq|yfinance|auto")
    return raw


def _fetch_one_routed(
    ticker: str,
    start: str,
    end: str,
    *,
    backend: str,
    yf_session: Any = None,
) -> Any:
    """Single-ticker dispatch honoring the resolved backend.

    - ``stooq``: propagates ``StooqAPIKeyRequired`` so the caller stops fast.
    - ``yfinance``: never touches stooq.
    - ``auto``: tries stooq, but on missing apikey or quota exhaustion silently
      falls back to yfinance (logged once).
    """
    if backend == "stooq":
        return _fetch_stooq_one(ticker, start, end)
    if backend == "yfinance":
        return _fetch_one_ticker(ticker, start, end, session=yf_session)
    # auto
    try:
        df = _fetch_stooq_one(ticker, start, end)
    except (StooqAPIKeyRequired, RuntimeError) as exc:
        logger.warning("Stooq unavailable, falling back to yfinance: %s", exc)
        df = None
    if df is not None and not df.empty:
        return df
    return _fetch_one_ticker(ticker, start, end, session=yf_session)


def fetch_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    *,
    cache_path: Optional[Path] = None,
    polite_sleep: float = 2.0,
    save_every: int = 25,
    backend: Optional[str] = None,
) -> Any:
    """Fetch OHLCV for a list of tickers. Returns a long-format DataFrame.

    Schema: index unique on (date, ticker); columns open/high/low/close/volume.

    Backend dispatch (override via env ``WQB_DATA_BACKEND`` or kwarg):
      * ``stooq`` (default) — free CSV endpoint, much more rate-limit tolerant
        than yfinance in 2026
      * ``yfinance`` — legacy fallback
      * ``auto`` — try stooq, fall back to yfinance per failed ticker

    Uses cache_path (default: artifacts/wq_brain/data/ohlcv.parquet) — only
    fetches tickers / date ranges not already cached. Saves every `save_every`
    tickers so partial progress survives interruption.
    """
    pd = _pd()
    cache_path = Path(cache_path) if cache_path else ohlcv_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_backend = _resolve_backend(backend)

    # Load existing cache
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
    else:
        cached = pd.DataFrame()

    cached_tickers = set(cached.index.get_level_values("ticker").unique()) if not cached.empty else set()
    missing = [t for t in tickers if t not in cached_tickers]

    if missing:
        logger.info("Fetching %d new tickers via backend=%s (%.1fs polite sleep)",
                    len(missing), resolved_backend, polite_sleep)
        yf_session = _make_yf_session() if resolved_backend in ("yfinance", "auto") else None
        new_rows: list[Any] = []
        ok = 0
        fail = 0
        for i, ticker in enumerate(missing, 1):
            df = _fetch_one_routed(
                ticker, start, end, backend=resolved_backend, yf_session=yf_session,
            )
            if df is not None and not df.empty:
                df = df.rename(columns=str.lower)
                df["ticker"] = ticker
                df.index.name = "date"
                df = df.reset_index().set_index(["date", "ticker"])
                cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
                new_rows.append(df[cols])
                ok += 1
            else:
                fail += 1
            time.sleep(polite_sleep)
            # Periodic save so we don't lose progress
            if i % save_every == 0 or i == len(missing):
                if new_rows:
                    new_df = pd.concat(new_rows)
                    cached = pd.concat([cached, new_df]) if not cached.empty else new_df
                    cached = cached[~cached.index.duplicated(keep="last")]
                    cached = cached.sort_index()
                    cached.to_parquet(cache_path)
                    new_rows = []
                logger.info("Progress %d/%d (ok=%d fail=%d cached_rows=%d)",
                            i, len(missing), ok, fail, len(cached))

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
    polite_sleep: float = 2.0,
    backend: Optional[str] = None,
) -> dict[str, Any]:
    """Top-level: fetch OHLCV + sectors + write metadata. Returns summary dict.

    See ``fetch_ohlcv`` for backend semantics. ``backend=None`` honors the
    ``WQB_DATA_BACKEND`` env var (default ``stooq``).
    """
    t0 = time.time()
    pd = _pd()
    ohlcv = fetch_ohlcv(tickers, start, end, polite_sleep=polite_sleep, backend=backend)
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
