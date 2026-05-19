"""Slot cool-down (anti-flip) hint + find_recent_revisits primitive tests."""
from __future__ import annotations

from agent_market.wq_brain.prompt_builder import (
    _record_slot_key,
    _slot_cooldown_hint,
    _slot_key,
)
from agent_market.wq_brain.tried_log import find_recent_revisits


# ── helpers ────────────────────────────────────────────────────────────


def _row(expr: str, *, ts: float, fitness: float = 1.0,
         sharpe: float = 1.0) -> dict:
    return {"expr": expr, "ts": ts, "sharpe": sharpe, "fitness": fitness,
            "turnover": 0.3, "alpha_id": "X", "status": "COMPLETE"}


# ── find_recent_revisits ───────────────────────────────────────────────


def test_revisits_groups_records_within_window():
    rows = [
        _row("rank(close)", ts=1),
        _row("rank(vwap)",  ts=2),
        _row("rank(close)", ts=3),
        _row("rank(close)", ts=4),
    ]
    out = find_recent_revisits(rows, lambda r: r["expr"], window=4, min_revisits=2)
    assert ("rank(close)" in out) and len(out["rank(close)"]) == 3
    assert "rank(vwap)" not in out  # only 1 visit


def test_revisits_caps_to_window():
    """Older records sliced off by window; only tail considered."""
    rows = [_row("rank(close)", ts=i) for i in range(10)]
    out = find_recent_revisits(rows, lambda r: r["expr"], window=3, min_revisits=2)
    # Last 3 records → all "rank(close)" → 3 visits
    assert len(out["rank(close)"]) == 3


def test_revisits_skips_none_keys():
    """key_fn returning None means 'skip this record'."""
    rows = [
        _row("rank(close)", ts=1),
        _row("",            ts=2),  # empty expr → key_fn returns None
        _row("rank(close)", ts=3),
    ]
    out = find_recent_revisits(
        rows,
        lambda r: r["expr"] or None,
        window=5, min_revisits=2,
    )
    assert "rank(close)" in out and len(out["rank(close)"]) == 2


def test_revisits_handles_keyerror_in_key_fn():
    """key_fn raising is treated as None (skip)."""
    rows = [_row("rank(close)", ts=1), _row("rank(close)", ts=2)]
    def boom(r): raise KeyError("missing")
    assert find_recent_revisits(rows, boom, window=5) == {}


def test_revisits_returns_empty_when_no_groups_meet_min():
    rows = [_row(f"rank(f{i})", ts=i) for i in range(5)]
    assert find_recent_revisits(
        rows, lambda r: r["expr"], window=5, min_revisits=2,
    ) == {}


def test_revisits_invalid_params_return_empty():
    rows = [_row("rank(close)", ts=1)]
    assert find_recent_revisits(rows, lambda r: r["expr"], window=0) == {}
    assert find_recent_revisits(rows, lambda r: r["expr"], min_revisits=0) == {}


# ── _slot_key ──────────────────────────────────────────────────────────


def test_slot_key_returns_none_for_empty_or_atomic():
    """Expressions with no operators have no skeleton → no slot."""
    assert _slot_key("") is None
    # Bare field/literal with no function call has no skeleton ops
    assert _slot_key("close") is None


def test_slot_key_includes_family_and_field_set():
    """Codex review R2-#6: slot now requires identical (family, skeleton,
    field-kinds, fields). `rank(close-vwap)` matches the vwap_dev family
    pattern; `rank(open-high)` lands in 'other' — different slots."""
    a = _slot_key("rank(close - vwap)")
    b = _slot_key("rank(open - high)")
    assert a != b


def test_slot_key_same_family_same_fields_same_slot():
    """Codex review R2-#6: identical exprs differing only by numeric
    constants share a slot (family, skeleton, fields all equal)."""
    a = _slot_key("rank(ts_corr(close, volume, 20))")
    b = _slot_key("rank(ts_corr(close, volume, 60))")
    assert a == b
    assert a is not None
    assert a[0] == "ts_corr_pv"  # family


def test_slot_key_different_fund_ratios_different_slot():
    """Codex review R2-#6: sales/assets vs debt/equity vs fcf/cap are
    DIFFERENT slots now (different field sets), preventing legitimate
    fundamental exploration from being flagged as anti-flip."""
    a = _slot_key("rank(sales / assets)")
    b = _slot_key("rank(debt / equity)")
    c = _slot_key("rank(fcf / cap)")
    assert a != b and b != c and a != c
    # All in same family though (fundamental_ratio)
    assert a[0] == b[0] == c[0] == "fundamental_ratio"


def test_slot_key_different_field_kinds_different_slot():
    """rank x1 with P/V vs FUND fields ⇒ different slots."""
    pv  = _slot_key("rank(close)")
    fnd = _slot_key("rank(sales / assets)")
    assert pv != fnd
    assert pv is not None and fnd is not None


def test_slot_key_different_operator_count_different_slot():
    a = _slot_key("rank(close)")                              # rank x1
    b = _slot_key("rank(close - ts_mean(close, 20))")         # rank x1, ts_mean x1
    assert a != b


# ── _slot_cooldown_hint ────────────────────────────────────────────────


def test_cooldown_silent_when_only_one_visit_per_slot():
    """Each row has a distinct (skeleton, kinds) tuple → no slot revisited."""
    rows = [
        _row("rank(close)",                             ts=1),
        _row("ts_rank(close, 20)",                      ts=2),
        _row("rank(sales / assets)",                    ts=3),    # FUND kinds
        _row("ts_corr(close, volume, 20)",              ts=4),
        _row("group_zscore(close - vwap, sector)",      ts=5),
    ]
    assert _slot_cooldown_hint(rows) == ""


def test_cooldown_fires_on_plateau_after_min_revisits():
    """Codex review #9 + #10 + R2-#6: needs ≥ min_revisits visits to the
    SAME slot (family + skeleton + fields), AND last K visits each below
    running-best - epsilon. Use a single ts_corr_pv slot with varying
    windows so all rows collapse to one slot."""
    rows = [
        _row("rank(ts_corr(close, volume, 20))",  ts=1, fitness=0.95),  # slot A
        _row("rank(ts_corr(close, volume, 40))",  ts=2, fitness=0.85),  # slot A
        _row("rank((high - low) / close)",        ts=3, fitness=1.20),  # slot B
        _row("rank(ts_corr(close, volume, 60))",  ts=4, fitness=0.80),  # slot A
        _row("rank(ts_corr(close, volume, 120))", ts=5, fitness=0.75),  # slot A
    ]
    # Slot A (ts_corr_pv, rankx1|ts_corrx1, {close, volume}) visits at
    # ts=1,2,4,5 → fi=[0.95, 0.85, 0.80, 0.75]
    # running_best=[0.95, 0.95, 0.95, 0.95]; last 2 (0.80, 0.75) ≤ 0.95-0.05
    out = _slot_cooldown_hint(rows, window=5, min_revisits=3,
                               consecutive_non_improvements=2, epsilon=0.05)
    assert "SLOT COOL-DOWN" in out
    assert "ts_corr_pv" in out
    assert "P/V" in out


def test_cooldown_silent_when_revisit_improves_fitness():
    """If second visit beats first, slot is still 'productive' → silent."""
    rows = [
        _row("rank(close)", ts=1, fitness=0.95),
        _row("rank(vwap)",  ts=2, fitness=1.40),  # same slot, but fi up → no flag
    ]
    assert _slot_cooldown_hint(rows, window=5, min_revisits=2) == ""


def test_cooldown_silent_on_single_regression():
    """Codex review #10: a single dip after a peak is NOT enough — could be
    noise. Need K consecutive non-improvements (default 2)."""
    rows = [
        _row("rank(close)", ts=1, fitness=0.95),
        _row("rank(vwap)",  ts=2, fitness=1.50),
        _row("rank(returns)", ts=3, fitness=1.10),
    ]
    # 3 visits to rankx1+{P/V}: fi=[0.95, 1.50, 1.10]
    # only 1 visit (1.10) below best (1.50); 1.50 itself is the peak so K=2
    # plateau requires BOTH last 2 below running best → 1.50 is not, so silent.
    out = _slot_cooldown_hint(rows, window=5, min_revisits=2,
                               consecutive_non_improvements=2)
    assert out == ""


def test_cooldown_silent_on_recovery_at_latest():
    """Codex review #10: noisy 1.0 → 1.5 → 1.4 → 1.6 → 1.5 should NOT freeze
    (still oscillating around best). The peak is recent."""
    rows = [
        _row("rank(close)",   ts=1, fitness=1.00),
        _row("rank(vwap)",    ts=2, fitness=1.50),
        _row("rank(open)",    ts=3, fitness=1.40),
        _row("rank(returns)", ts=4, fitness=1.60),
        _row("rank(high)",    ts=5, fitness=1.55),
    ]
    # rankx1+{P/V} 5 visits: [1.00, 1.50, 1.40, 1.60, 1.55]
    # running_best=[1.00, 1.50, 1.50, 1.60, 1.60]
    # last 2 (1.60, 1.55): 1.60 == running_best[-1], not ≤ best-ε → silent.
    out = _slot_cooldown_hint(rows, window=5, min_revisits=3,
                               consecutive_non_improvements=2, epsilon=0.05)
    assert out == ""


def test_cooldown_skips_failed_simulate_rows():
    """Codex review #10: rows with status != COMPLETE or non-finite fitness
    don't count as 'no progress' — agent never had data to compare."""
    rows = [
        _row("rank(close)",   ts=1, fitness=0.95),
        _row("rank(vwap)",    ts=2, fitness=0.50),
    ]
    rows.append({"expr": "rank(open)", "ts": 3, "fitness": None,
                 "sharpe": None, "turnover": None, "alpha_id": "X",
                 "status": "ERROR"})
    rows.append({"expr": "rank(high)", "ts": 4, "fitness": float("nan"),
                 "sharpe": 1.0, "turnover": 0.3, "alpha_id": "X",
                 "status": "COMPLETE"})
    # Only ts=1,2 have valid fitness → fi_series=[0.95, 0.50]
    # < min_revisits=3 → silent.
    out = _slot_cooldown_hint(rows, window=5, min_revisits=3)
    assert out == ""


def test_cooldown_window_cap_excludes_old_visits():
    """Codex review R2-#6: window slicing happens BEFORE slot grouping —
    older same-slot visits drop off. Use 4 same-slot rows and a window
    that excludes the first."""
    rows = [
        _row("rank(ts_corr(close, volume, 20))",  ts=1, fitness=1.50),  # peak (excluded)
        _row("rank(ts_corr(close, volume, 40))",  ts=2, fitness=1.20),
        _row("rank(ts_corr(close, volume, 60))",  ts=3, fitness=1.15),
        _row("rank(ts_corr(close, volume, 120))", ts=4, fitness=1.10),
    ]
    # window=3 slice = ts=2,3,4 → fi=[1.20, 1.15, 1.10]; running_best=[1.20, 1.20, 1.20]
    # last K=2 (1.15, 1.10) ≤ 1.20-0.05=1.15 → 1.15 <= 1.15 ✓; 1.10 <= 1.15 ✓ → plateau
    out = _slot_cooldown_hint(rows, window=3, min_revisits=3,
                               consecutive_non_improvements=2, epsilon=0.05)
    assert "SLOT COOL-DOWN" in out
    assert "visits | best fi" in out
    # window=4 includes ts=1 peak (1.50) → running_best=[1.50,1.50,1.50,1.50];
    # last 2 (1.15, 1.10) ≤ 1.50-0.05=1.45 → strong plateau, also flagged.
    out_wide = _slot_cooldown_hint(rows, window=4, min_revisits=3,
                                    consecutive_non_improvements=2, epsilon=0.05)
    assert "SLOT COOL-DOWN" in out_wide


def test_cooldown_record_slot_key_handles_blank_record():
    """The dict adapter returns None for records with empty expr."""
    assert _record_slot_key({"expr": ""}) is None
    assert _record_slot_key({}) is None


def test_cooldown_silent_when_records_too_few():
    rows = [_row("rank(close)", ts=1)]
    assert _slot_cooldown_hint(rows, min_revisits=2) == ""
