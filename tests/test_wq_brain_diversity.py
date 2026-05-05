"""Diversity tests — semantic_jaccard catches 'same skeleton, different fields'."""
from __future__ import annotations

import pytest

from agent_market.wq_brain.diversity import (
    CANONICAL_FAMILIES,
    family_distribution,
    max_semantic_similarity,
    next_target_family,
    op_field_multiset,
    semantic_jaccard,
)


# ── op_field_multiset ────────────────────────────────────────────────────


def test_op_field_multiset_separates_ops_and_fields():
    ops, fields = op_field_multiset("rank(ts_corr(close, volume, 20))")
    assert ops == {"rank": 1, "ts_corr": 1}
    assert fields == {"close": 1, "volume": 1}


def test_op_field_multiset_counts_repetition():
    ops, fields = op_field_multiset("rank(rank(close)) + rank(close)")
    assert ops["rank"] == 3
    assert fields["close"] == 2


def test_op_field_multiset_drops_numeric_and_unknown():
    ops, fields = op_field_multiset("rank(close * 0.5 + xyz)")
    assert ops == {"rank": 1}
    assert fields == {"close": 1}  # `xyz` not in KNOWN_FIELDS, `0.5` is numeric


def test_op_field_multiset_handles_empty():
    ops, fields = op_field_multiset("")
    assert ops == {}
    assert fields == {}


def test_op_field_multiset_groups_by_sector():
    ops, fields = op_field_multiset("group_zscore(close - vwap, sector)")
    assert ops == {"group_zscore": 1}
    assert fields == {"close": 1, "vwap": 1, "sector": 1}


# ── semantic_jaccard ────────────────────────────────────────────────────


def test_semantic_jaccard_identical_is_one():
    e = "rank(ts_corr(close, volume, 20))"
    assert semantic_jaccard(e, e) == pytest.approx(1.0)


def test_semantic_jaccard_disjoint_low():
    a = "rank(close)"
    b = "ts_corr(high, low, 20)"
    sim = semantic_jaccard(a, b)
    assert sim < 0.5


def test_semantic_jaccard_catches_field_swap():
    """rank(ts_rank(close, 252)) vs rank(ts_rank(vwap, 252)) should be near 1
    on the operator side — token jaccard misses this pattern."""
    a = "rank(ts_rank(close, 252) * (-ts_delta(close, 3) / close))"
    b = "rank(ts_rank(vwap, 252) * (-ts_delta(vwap, 3) / vwap))"
    sim = semantic_jaccard(a, b)
    assert sim > 0.7, f"expected high similarity (operator skeleton same), got {sim}"


def test_semantic_jaccard_distinguishes_different_skeletons():
    """Same fields but different operator stack → similarity should drop."""
    a = "rank(ts_rank(close, 252))"
    b = "ts_decay_linear(close, 20)"
    sim = semantic_jaccard(a, b)
    assert sim < 0.7


def test_semantic_jaccard_op_weight_zero_uses_only_fields():
    a = "rank(close)"
    b = "ts_decay_linear(close, 5)"
    # ops differ, but fields ({close}) identical → with op_weight=0 should be 1.0
    sim = semantic_jaccard(a, b, op_weight=0.0)
    assert sim == pytest.approx(1.0)


def test_semantic_jaccard_op_weight_one_uses_only_ops():
    a = "rank(close)"
    b = "rank(volume)"
    sim = semantic_jaccard(a, b, op_weight=1.0)
    assert sim == pytest.approx(1.0)  # only ops, ops are identical


def test_semantic_jaccard_validates_op_weight():
    with pytest.raises(ValueError):
        semantic_jaccard("a", "b", op_weight=1.5)


# ── family_distribution / next_target_family ────────────────────────────


def test_family_distribution_skips_empty():
    exprs = ["rank(close)", "", None, "rank(ts_corr(close, volume, 20))"]
    dist = family_distribution(e for e in exprs if e)
    assert "ts_corr_pv" in dist
    assert "other" in dist  # rank(close) → other


def test_next_target_family_picks_first_canonical_when_empty():
    assert next_target_family([]) == CANONICAL_FAMILIES[0]


def test_next_target_family_returns_none_when_uniform():
    """If every canonical family has same count → no recommendation."""
    one_per_family = [
        "rank(ts_corr(close, volume, 20))",            # ts_corr_pv
        "rank((high - low) / close)",                  # intraday_range
        "rank(close / vwap)",                          # vwap_dev
        "rank(ts_rank(volume, 20))",                   # volume_rank
        "rank(open - ts_delay(close, 1))",             # open_gap
        "hump(rank(close))",                           # humped
        "rank(close) + 0.5 * rank(volume)",            # multi_signal
        "group_zscore(close, sector)",                 # sector_relative
    ]
    assert next_target_family(one_per_family) is None


def test_next_target_family_picks_missing_when_pool_skewed():
    """All ts_corr_pv → recommend the next canonical family (intraday_range)."""
    pool = ["rank(ts_corr(close, volume, 20))"] * 5
    nxt = next_target_family(pool)
    assert nxt == "intraday_range"


# ── max_semantic_similarity ─────────────────────────────────────────────


def test_max_semantic_similarity_empty_pool_returns_zero():
    sim, blocker = max_semantic_similarity("rank(close)", [])
    assert sim == 0.0
    assert blocker is None


def test_max_semantic_similarity_finds_skeleton_dup():
    pool = [
        "rank(close)",                                    # different
        "rank(ts_rank(close, 252) * (-ts_delta(close, 3) / close))",  # ACTIVE-like
    ]
    candidate = "rank(ts_rank(vwap, 252) * (-ts_delta(vwap, 3) / vwap))"  # field swap
    sim, blocker = max_semantic_similarity(candidate, pool)
    assert sim > 0.7
    assert blocker == pool[1]
