"""Edit-altitude classifier + taxonomy + distribution-hint tests."""
from __future__ import annotations

from agent_market.wq_brain.prompt_builder import (
    _altitude_taxonomy_block,
    _classify_edit_altitude,
    _extract_field_set,
    _extract_numbers,
    _recent_altitude_distribution_hint,
)


# ── helpers ────────────────────────────────────────────────────────────


def _row(expr: str, *, ts: float) -> dict:
    return {"expr": expr, "ts": ts, "sharpe": 1.0, "fitness": 1.0,
            "turnover": 0.3, "alpha_id": "X", "status": "COMPLETE"}


# ── _extract_numbers / _extract_field_set ──────────────────────────────


def test_extract_numbers_sorted_dedup_unaware():
    """Floats are sorted; duplicates preserved (multiset semantics)."""
    out = _extract_numbers("ts_rank(close, 60) - ts_rank(volume, 20)")
    assert out == (20.0, 60.0)


def test_extract_numbers_handles_decimals():
    out = _extract_numbers("signed_power(returns, 0.5) + power(rank(returns), 2)")
    assert 0.5 in out and 2.0 in out


def test_extract_numbers_empty_when_no_constants():
    assert _extract_numbers("rank(close)") == ()
    assert _extract_numbers("") == ()


def test_extract_field_set_only_canonical_fields():
    out = _extract_field_set("rank(close - vwap) * ts_corr(close, volume, 20)")
    assert "close" in out and "vwap" in out and "volume" in out
    assert "ts_corr" not in out  # operator, not field
    assert "20" not in out


def test_extract_field_set_empty_for_blank():
    assert _extract_field_set("") == frozenset()


# ── _classify_edit_altitude ─────────────────────────────────────────────


def test_altitude_l1_when_family_changes():
    """ts_corr_pv → intraday_range = different family bucket."""
    out = _classify_edit_altitude(
        "rank(ts_corr(close, volume, 20))",
        "rank((high - low) / close)",
    )
    assert out == "L1"


def test_altitude_l2_when_skeleton_changes_in_same_family():
    """Both match ts_corr_pv but operator multiset differs."""
    out = _classify_edit_altitude(
        "rank(ts_corr(close, volume, 20))",
        "rank(ts_corr(close, volume, 60) * ts_zscore(volume, 20))",
    )
    assert out == "L2"


def test_altitude_l3_when_only_window_changes():
    """Same family, same skeleton, different numeric constants."""
    out = _classify_edit_altitude(
        "rank(ts_corr(close, volume, 20))",
        "rank(ts_corr(close, volume, 60))",
    )
    assert out == "L3"


def test_altitude_l4_when_only_field_swap():
    """Same family ('other'), same skeleton, no numbers, only fields differ."""
    out = _classify_edit_altitude("rank(close)", "rank(vwap)")
    assert out == "L4"


def test_altitude_dash_when_identical():
    out = _classify_edit_altitude("rank(close)", "rank(close)")
    assert out == "—"


def test_altitude_dash_when_blank():
    assert _classify_edit_altitude("", "rank(close)") == "—"
    assert _classify_edit_altitude("rank(close)", "") == "—"


# ── _altitude_taxonomy_block ────────────────────────────────────────────


def test_taxonomy_block_lists_all_four_levels():
    blk = _altitude_taxonomy_block()
    assert "L1 family swap" in blk
    assert "L2 skeleton swap" in blk
    assert "L3 calibration" in blk
    assert "L4 refinement" in blk
    assert "Bubble-up rule" in blk


def test_taxonomy_block_deterministic():
    """Same call should produce identical output (no time-based state)."""
    assert _altitude_taxonomy_block() == _altitude_taxonomy_block()


# ── _recent_altitude_distribution_hint ──────────────────────────────────


def test_distribution_hint_empty_when_too_few_records():
    assert _recent_altitude_distribution_hint([]) == ""
    assert _recent_altitude_distribution_hint([_row("rank(close)", ts=1)]) == ""


def test_distribution_hint_counts_l1_l2_l3_l4():
    """Window of 5 records → 4 transitions: L1, L2, L3, L4."""
    rows = [
        _row("rank(ts_corr(close, volume, 20))",                       ts=1),
        _row("rank((high - low) / close)",                              ts=2),  # L1 ts_corr_pv→intraday_range
        _row("rank((high - low) / vwap)",                               ts=3),  # L4 same skeleton, field swap
        _row("rank((high - low) / vwap) - rank(volume / adv20)",        ts=4),  # L2 skeleton change
        _row("rank((high - low) / vwap) - rank(volume / adv60)",        ts=5),  # L4 field swap (adv20→adv60)
    ]
    out = _recent_altitude_distribution_hint(rows)
    assert "RECENT EDIT-ALTITUDE DISTRIBUTION" in out
    assert "L1 family-swap × 1" in out
    assert "L2 skeleton-swap × 1" in out
    assert "L4 refinement × 2" in out


def test_distribution_hint_fires_bubble_up_when_last_two_l4():
    """Last 2 transitions are L4 → bubble-up advisory rendered."""
    rows = [
        _row("rank(ts_corr(close, volume, 20))",   ts=1),
        _row("rank(ts_corr(close, volume, 60))",   ts=2),  # L3
        _row("rank(close)",                         ts=3),  # L1 (other family)
        _row("rank(vwap)",                          ts=4),  # L4
        _row("rank(returns)",                       ts=5),  # L4
    ]
    out = _recent_altitude_distribution_hint(rows)
    assert "BUBBLE-UP ADVISORY" in out


def test_distribution_hint_no_bubble_up_when_last_includes_l3():
    """Last L3 breaks the L4 streak → no bubble-up."""
    rows = [
        _row("rank(close)",                                   ts=1),
        _row("rank(vwap)",                                    ts=2),  # L4
        _row("rank(ts_corr(close, volume, 20))",              ts=3),  # L1
        _row("rank(ts_corr(close, volume, 60))",              ts=4),  # L3
    ]
    out = _recent_altitude_distribution_hint(rows)
    assert "BUBBLE-UP ADVISORY" not in out


def test_distribution_hint_no_high_advisory_when_only_l3_l4():
    """Window without any L1 or L2 transition → no-high advisory."""
    rows = [
        _row("rank(ts_corr(close, volume, 20))",   ts=1),
        _row("rank(ts_corr(close, volume, 60))",   ts=2),  # L3
        _row("rank(ts_corr(close, volume, 252))",  ts=3),  # L3
    ]
    out = _recent_altitude_distribution_hint(rows)
    assert "No L1/L2 edits" in out


def test_distribution_hint_window_caps_recent():
    """Only last `window` records considered."""
    # 12 records, window=4 → only last 4 → 3 transitions
    rows = [_row(f"rank(close - {i})", ts=i) for i in range(12)]
    out = _recent_altitude_distribution_hint(rows, window=4)
    # last 4 are: rank(close - 8), rank(close - 9), rank(close - 10), rank(close - 11)
    # L3 transitions (numeric constant changes only)
    assert "(last 3 transitions)" in out
    assert "L3 calibration × 3" in out
