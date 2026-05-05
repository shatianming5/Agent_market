"""Data quality audit for the OHLCV cache.

Five checks, each producing structured findings (severity + count + sample
rows). Audits never mutate the cache — they only emit a report so the user
can decide whether to re-import / filter / drop.

Checks:

1. **OHLC invariant** — every row should satisfy
   ``low <= min(open, close)`` and ``high >= max(open, close)``;
   ``volume >= 0``. Catches obvious corruption.
2. **Split sanity** — the cache uses ``adjusted`` as the close column, so
   split events should already be smoothed. If we see a |close_t / close_{t-1}|
   ratio ∈ (0, 0.5) ∪ (2.0, ∞), the adjustment likely failed for that
   ticker. Cross-checks against 5 well-known splits as ground truth.
3. **Ticker reuse / data gaps** — same symbol re-listed under a different
   company (e.g. GM pre/post-2009 bankruptcy) appears as a > 365-day gap
   in trading dates within one ticker.
4. **Survivor bias** — current ticker count vs the canonical Russell 3000
   roster. Coverage < 70% suggests stale or filtered data.
5. **Outlier rows** — ``volume == 0`` and ``close <= 0`` rows; counted per
   ticker.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _pd():
    import pandas as pd
    return pd


# ── Known historical splits (used for split-sanity ground truth) ────────
# (ticker, date_str, split_ratio_old_to_new) — date_str is the *ex-split* date.
KNOWN_SPLITS: tuple[tuple[str, str, float], ...] = (
    ("AAPL", "2020-08-31", 4.0),    # 4-for-1
    ("TSLA", "2020-08-31", 5.0),    # 5-for-1
    ("GOOGL", "2014-04-03", 2.0),   # 1998 split, then class C separation
    ("AMZN", "2022-06-06", 20.0),   # 20-for-1
    ("NVDA", "2024-06-10", 10.0),   # 10-for-1
)


# Approximate canonical Russell 3000 size (the index targets ~3000 names)
EXPECTED_RUSSELL_3000_SIZE = 3000


@dataclass
class AuditFinding:
    name: str
    severity: str  # "info" | "warn" | "error"
    summary: str
    count: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditReport:
    rows_total: int
    tickers_total: int
    date_min: Optional[str]
    date_max: Optional[str]
    findings: list[AuditFinding] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_total": self.rows_total,
            "tickers_total": self.tickers_total,
            "date_min": self.date_min,
            "date_max": self.date_max,
            "elapsed_sec": round(self.elapsed_sec, 1),
            "findings": [f.to_dict() for f in self.findings],
            "summary": {
                f.name: {"severity": f.severity, "count": f.count}
                for f in self.findings
            },
        }


# ── Individual checks ───────────────────────────────────────────────────


def check_ohlc_invariant(df: Any, *, sample_size: int = 5) -> AuditFinding:
    """``low <= min(open, close)`` and ``high >= max(open, close)``."""
    pd = _pd()
    pieces = []
    bad_low = (df["low"] > df[["open", "close"]].min(axis=1)) & df["low"].notna()
    bad_high = (df["high"] < df[["open", "close"]].max(axis=1)) & df["high"].notna()
    bad_vol = (df["volume"] < 0) & df["volume"].notna()
    bad = bad_low | bad_high | bad_vol
    n = int(bad.sum())
    if n:
        sub = df.loc[bad].head(sample_size).reset_index()
        pieces = sub.to_dict(orient="records")
    return AuditFinding(
        name="ohlc_invariant",
        severity="error" if n > 100 else ("warn" if n else "info"),
        summary=f"{n} rows violate low<=min(o,c) / high>=max(o,c) / volume>=0",
        count=n,
        samples=pieces,
    )


def check_split_sanity(
    df: Any,
    *,
    extreme_ratio_low: float = 0.5,
    extreme_ratio_high: float = 2.0,
    known_splits: tuple = KNOWN_SPLITS,
    sample_size: int = 10,
) -> AuditFinding:
    """Adjacent-day close ratio outside [0.5, 2.0] is a missing-split signal.

    Also reports check status against ``KNOWN_SPLITS``: at the ex-split
    date for a known split, the ratio in our (adjusted) cache should be
    near 1.0 (because adjusted prices smooth out the split), NOT the raw
    split ratio.
    """
    pd = _pd()
    g = df["close"].groupby(level="ticker")
    ratios = (g.shift(0) / g.shift(1)).dropna()
    extreme_mask = (ratios < extreme_ratio_low) | (ratios > extreme_ratio_high)
    extreme = ratios[extreme_mask]
    n_extreme = int(len(extreme))
    samples: list[dict[str, Any]] = []
    if n_extreme:
        extreme_sorted = extreme.abs().sort_values(ascending=False).head(sample_size)
        for idx, ratio in extreme_sorted.items():
            samples.append({
                "date": str(idx[0]) if isinstance(idx, tuple) else str(idx),
                "ticker": idx[1] if isinstance(idx, tuple) else "",
                "ratio": float(ratio),
            })

    # Check known splits — ratio in our adjusted cache should be near 1.0
    known_check: list[dict[str, Any]] = []
    for ticker, date_s, expected_split in known_splits:
        try:
            ts = pd.Timestamp(date_s)
            mask = (
                (df.index.get_level_values("ticker") == ticker)
                & (df.index.get_level_values("date") == ts)
            )
            if not mask.any():
                known_check.append({"ticker": ticker, "date": date_s,
                                    "found": False, "note": "row missing"})
                continue
            row = df[mask]
            close_at = float(row["close"].iloc[0])
            # The day before
            tm1 = ts - pd.Timedelta(days=10)  # window in case of weekend
            prev_mask = (
                (df.index.get_level_values("ticker") == ticker)
                & (df.index.get_level_values("date") < ts)
                & (df.index.get_level_values("date") >= tm1)
            )
            prev_rows = df[prev_mask].sort_index()
            if prev_rows.empty:
                known_check.append({"ticker": ticker, "date": date_s,
                                    "found": False, "note": "no prior row"})
                continue
            prev_close = float(prev_rows["close"].iloc[-1])
            ratio = close_at / prev_close if prev_close else 0.0
            # Adjusted-close ratio at known split should be ≈ 1.0; raw would be 1/expected
            adjusted_ok = 0.7 <= ratio <= 1.3
            known_check.append({
                "ticker": ticker, "date": date_s,
                "found": True,
                "ratio": round(ratio, 3),
                "expected_split": expected_split,
                "adjusted_smooth": adjusted_ok,
            })
        except Exception as exc:
            known_check.append({"ticker": ticker, "date": date_s,
                                "error": str(exc)})

    smooth_count = sum(1 for k in known_check if k.get("adjusted_smooth"))
    severity = "info" if (n_extreme < 50 and smooth_count >= 3) else "warn"
    return AuditFinding(
        name="split_sanity",
        severity=severity,
        summary=(
            f"{n_extreme} adjacent-day close ratios outside [0.5, 2.0]; "
            f"{smooth_count}/{len(known_check)} known splits look adjusted"
        ),
        count=n_extreme,
        samples=samples,
        extra={"known_splits_check": known_check},
    )


def check_ticker_reuse(df: Any, *, gap_days: int = 365) -> AuditFinding:
    """Within-ticker date gap > N days suggests symbol reuse / re-listing."""
    pd = _pd()
    by_ticker = df.reset_index().sort_values(["ticker", "date"])
    by_ticker["gap"] = (
        by_ticker.groupby("ticker")["date"].diff().dt.days
    )
    bad = by_ticker[by_ticker["gap"] > gap_days]
    samples: list[dict[str, Any]] = [
        {"ticker": r["ticker"],
         "gap_start": str(r["date"]),
         "gap_days": int(r["gap"])}
        for _, r in bad.head(15).iterrows()
    ]
    return AuditFinding(
        name="ticker_reuse",
        severity="warn" if len(bad) else "info",
        summary=f"{len(bad)} within-ticker date gaps > {gap_days} days",
        count=int(len(bad)),
        samples=samples,
        extra={"gap_threshold_days": gap_days},
    )


def check_survivor_bias(
    df: Any,
    *,
    expected_size: int = EXPECTED_RUSSELL_3000_SIZE,
    coverage_warn_below: float = 0.70,
) -> AuditFinding:
    """Tickers count vs Russell 3000 expected size."""
    actual = int(df.index.get_level_values("ticker").nunique())
    coverage = actual / expected_size if expected_size else 0.0
    severity = "warn" if coverage < coverage_warn_below else "info"
    return AuditFinding(
        name="survivor_bias",
        severity=severity,
        summary=(
            f"{actual} unique tickers ({coverage*100:.1f}% of expected "
            f"{expected_size}). Below {coverage_warn_below*100:.0f}% suggests "
            f"the Kaggle dump has filtered out delisted names."
        ),
        count=expected_size - actual if actual < expected_size else 0,
        samples=[],
        extra={
            "expected_size": expected_size,
            "actual_size": actual,
            "coverage_pct": round(coverage * 100, 1),
        },
    )


def check_outliers(df: Any, *, sample_size: int = 10) -> AuditFinding:
    """Zero-volume and non-positive-close rows."""
    zero_vol = (df["volume"] == 0)
    bad_close = (df["close"] <= 0)
    bad = zero_vol | bad_close
    n = int(bad.sum())
    samples: list[dict[str, Any]] = []
    if n:
        sub = df.loc[bad].head(sample_size).reset_index()
        for _, r in sub.iterrows():
            samples.append({
                "date": str(r["date"]),
                "ticker": r["ticker"],
                "volume": float(r.get("volume", float("nan"))),
                "close": float(r.get("close", float("nan"))),
            })
    return AuditFinding(
        name="outliers",
        severity="warn" if n > 1000 else "info",
        summary=f"{n} rows with volume==0 or close<=0",
        count=n,
        samples=samples,
    )


# ── Top-level orchestration ─────────────────────────────────────────────


def run_audit(
    df: Any,
    *,
    sample_size: int = 10,
) -> AuditReport:
    pd = _pd()
    t0 = time.time()
    if df is None or len(df) == 0:
        return AuditReport(
            rows_total=0, tickers_total=0, date_min=None, date_max=None,
            findings=[AuditFinding(
                name="empty_cache", severity="error",
                summary="OHLCV cache is empty; nothing to audit"
            )],
            elapsed_sec=time.time() - t0,
        )
    dates = df.index.get_level_values("date")
    tickers = df.index.get_level_values("ticker")
    findings = [
        check_ohlc_invariant(df, sample_size=sample_size),
        check_split_sanity(df, sample_size=sample_size),
        check_ticker_reuse(df),
        check_survivor_bias(df),
        check_outliers(df, sample_size=sample_size),
    ]
    return AuditReport(
        rows_total=int(len(df)),
        tickers_total=int(tickers.nunique()),
        date_min=str(dates.min()),
        date_max=str(dates.max()),
        findings=findings,
        elapsed_sec=time.time() - t0,
    )


def write_audit_artifacts(report: AuditReport, *, out_dir: Optional[Path] = None) -> dict[str, str]:
    from .data_loader import data_root
    out = Path(out_dir) if out_dir else data_root()
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "_audit.json"
    md_path = out / "_audit.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")

    # Markdown report
    lines = [
        "# OHLCV Cache Audit",
        "",
        f"_rows = {report.rows_total:,} · tickers = {report.tickers_total} · "
        f"dates = {report.date_min} → {report.date_max} · elapsed = {report.elapsed_sec:.1f}s_",
        "",
        "| check | severity | count | summary |",
        "|---|---|---|---|",
    ]
    for f in report.findings:
        lines.append(f"| `{f.name}` | **{f.severity}** | {f.count} | {f.summary} |")
    lines.append("")
    for f in report.findings:
        if not f.samples and not f.extra:
            continue
        lines.append(f"## `{f.name}`")
        lines.append("")
        if f.samples:
            keys = list(f.samples[0].keys())
            lines.append("| " + " | ".join(keys) + " |")
            lines.append("|" + "|".join("---" for _ in keys) + "|")
            for s in f.samples[:10]:
                lines.append("| " + " | ".join(str(s.get(k, "")) for k in keys) + " |")
            lines.append("")
        if f.extra:
            lines.append("```json")
            lines.append(json.dumps(f.extra, indent=2, default=str))
            lines.append("```")
            lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}
