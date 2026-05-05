"""data_audit tests — synthetic OHLCV fixtures hit each check."""
from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from agent_market.wq_brain.data_audit import (
    AuditFinding,
    AuditReport,
    check_ohlc_invariant,
    check_outliers,
    check_split_sanity,
    check_survivor_bias,
    check_ticker_reuse,
    run_audit,
    write_audit_artifacts,
)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "ticker"]).sort_index()


# ── ohlc_invariant ─────────────────────────────────────────────────────


def test_ohlc_invariant_passes_clean_data():
    df = _make_df([
        {"date": "2024-01-02", "ticker": "A", "open": 10, "high": 11, "low": 9.5, "close": 10.5, "volume": 1000},
        {"date": "2024-01-03", "ticker": "A", "open": 10.5, "high": 12, "low": 10, "close": 11, "volume": 1100},
    ])
    f = check_ohlc_invariant(df)
    assert f.count == 0
    assert f.severity == "info"


def test_ohlc_invariant_catches_inverted_high_low():
    df = _make_df([
        {"date": "2024-01-02", "ticker": "X", "open": 10, "high": 9, "low": 11, "close": 10, "volume": 1},
    ])
    f = check_ohlc_invariant(df)
    assert f.count >= 1
    assert f.severity in ("warn", "error")


def test_ohlc_invariant_catches_negative_volume():
    df = _make_df([
        {"date": "2024-01-02", "ticker": "X", "open": 10, "high": 11, "low": 9, "close": 10, "volume": -50},
    ])
    f = check_ohlc_invariant(df)
    assert f.count == 1


# ── split_sanity ───────────────────────────────────────────────────────


def test_split_sanity_passes_smooth_close():
    rows = []
    for i, dt in enumerate(pd.date_range("2024-01-02", periods=20)):
        rows.append({"date": dt, "ticker": "A",
                     "open": 100 + i, "high": 100 + i + 1, "low": 100 + i - 1,
                     "close": 100 + i, "volume": 1000})
    df = _make_df(rows)
    f = check_split_sanity(df)
    assert f.count == 0


def test_split_sanity_flags_unadjusted_split():
    """If a 4-for-1 split shows up as a raw 4x drop in close, flag it."""
    rows = [
        {"date": "2024-08-30", "ticker": "X", "open": 400, "high": 410, "low": 395, "close": 400, "volume": 100},
        {"date": "2024-09-03", "ticker": "X", "open": 100, "high": 105, "low": 99, "close": 100, "volume": 100},
    ]
    df = _make_df(rows)
    f = check_split_sanity(df)
    assert f.count >= 1
    assert f.samples and abs(f.samples[0]["ratio"]) <= 0.5


# ── ticker_reuse ───────────────────────────────────────────────────────


def test_ticker_reuse_no_gap():
    rows = []
    for dt in pd.date_range("2024-01-01", "2024-02-29"):
        rows.append({"date": dt, "ticker": "A", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1})
    df = _make_df(rows)
    f = check_ticker_reuse(df)
    assert f.count == 0


def test_ticker_reuse_detects_long_gap():
    rows = [
        {"date": pd.Timestamp("2010-01-04"), "ticker": "GM", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"date": pd.Timestamp("2014-01-04"), "ticker": "GM", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"date": pd.Timestamp("2014-01-05"), "ticker": "GM", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
    ]
    df = _make_df(rows)
    f = check_ticker_reuse(df, gap_days=365)
    assert f.count == 1
    assert f.samples[0]["ticker"] == "GM"


# ── survivor_bias ──────────────────────────────────────────────────────


def test_survivor_bias_low_coverage_warns():
    rows = [{"date": "2024-01-02", "ticker": f"T{i}",
             "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
            for i in range(100)]
    df = _make_df(rows)
    f = check_survivor_bias(df, expected_size=3000)
    assert f.severity == "warn"
    assert f.extra["coverage_pct"] < 70


def test_survivor_bias_high_coverage_info():
    rows = [{"date": "2024-01-02", "ticker": f"T{i}",
             "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
            for i in range(2900)]
    df = _make_df(rows)
    f = check_survivor_bias(df, expected_size=3000)
    assert f.severity == "info"
    assert f.extra["coverage_pct"] >= 90


# ── outliers ───────────────────────────────────────────────────────────


def test_outliers_flags_zero_volume():
    rows = [{"date": "2024-01-02", "ticker": "X", "open": 1, "high": 1, "low": 1,
             "close": 1, "volume": 0}]
    df = _make_df(rows)
    f = check_outliers(df)
    assert f.count == 1


def test_outliers_flags_negative_close():
    rows = [{"date": "2024-01-02", "ticker": "X", "open": 1, "high": 1, "low": 1,
             "close": -5, "volume": 100}]
    df = _make_df(rows)
    f = check_outliers(df)
    assert f.count == 1


# ── orchestration & artifact write ─────────────────────────────────────


def test_run_audit_returns_report_with_all_checks():
    rows = [{"date": "2024-01-02", "ticker": "A", "open": 1, "high": 2, "low": 0.5,
             "close": 1.5, "volume": 1000}]
    df = _make_df(rows)
    report = run_audit(df)
    assert report.rows_total == 1
    assert report.tickers_total == 1
    names = [f.name for f in report.findings]
    assert {"ohlc_invariant", "split_sanity", "ticker_reuse", "survivor_bias", "outliers"}.issubset(names)


def test_run_audit_empty_df():
    df = pd.DataFrame()
    report = run_audit(df)
    assert report.rows_total == 0
    assert any(f.name == "empty_cache" for f in report.findings)


def test_write_audit_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    rows = [{"date": "2024-01-02", "ticker": "A", "open": 1, "high": 2, "low": 0.5,
             "close": 1.5, "volume": 1000}]
    df = _make_df(rows)
    report = run_audit(df)
    paths = write_audit_artifacts(report)
    assert Path(paths["json"]).exists()
    assert Path(paths["md"]).exists()
    md = Path(paths["md"]).read_text()
    assert "OHLCV Cache Audit" in md
    assert "ohlc_invariant" in md
