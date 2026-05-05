"""Crossover engine tests."""
from __future__ import annotations

from agent_market.wq_brain.crossover import (
    Segment,
    extract_top_segments,
    format_crossover_block,
    infer_family,
)


def test_infer_family_classifies_known_patterns():
    assert infer_family("rank(ts_corr(close, volume, 20))") == "ts_corr_pv"
    assert infer_family("rank(ts_rank(close, 252))") == "ts_rank_close"
    assert infer_family("rank((high - low) / close)") == "intraday_range"
    assert infer_family("rank(close / vwap)") == "vwap_dev"
    assert infer_family("rank(ts_decay_linear(returns, 10))") == "decay_linear"
    assert infer_family("hump(rank(close), 0.01)") == "humped"
    # group_zscore against a generic group → group_neutral
    assert infer_family("rank(group_zscore(close, market))") == "group_neutral"
    assert infer_family("rank(ts_rank(volume, 20))") == "volume_rank"
    assert infer_family("rank(close + 1)") == "other"


def test_infer_family_recognizes_sector_relative():
    """group_* with sector / industry / subindustry is its own family."""
    assert infer_family("rank(group_zscore(returns, sector))") == "sector_relative"
    assert infer_family("group_neutralize(close - vwap, industry)") == "sector_relative"
    assert infer_family("group_rank(ts_corr(close, volume, 20), subindustry)") == "sector_relative"


def test_infer_family_recognizes_multi_signal():
    """Linear combinations of rank()'d signals are multi_signal."""
    assert infer_family("rank(close) + 0.5 * rank(volume)") == "multi_signal"
    assert infer_family("rank(close) - 0.3 * rank(-ts_corr(close, volume, 20))") == "multi_signal"
    assert infer_family("rank(close) + 0.5 * (-rank(volume))") == "multi_signal"
    # Composite that contains ts_corr but is dominated by linear combo → still multi_signal
    assert infer_family(
        "rank(ts_rank(close, 252)) + 0.5 * rank(-ts_corr(close, volume, 20))"
    ) == "multi_signal"


def test_extract_top_segments_filters_by_score_and_status():
    records = [
        {"expr": "rank(close)", "sharpe": 0.05, "fitness": 0.01, "turnover": 0.2,
         "status": "COMPLETE"},  # quick_score too low
        {"expr": "rank(ts_rank(close, 252) * (-ts_delta(close, 3)/close))",
         "sharpe": 1.47, "fitness": 0.77, "turnover": 0.46, "status": "COMPLETE"},
        {"expr": "rank(-ts_corr(close, volume, 20))",
         "sharpe": 1.02, "fitness": 0.56, "turnover": 0.17, "status": "COMPLETE"},
        {"expr": "broken_thing", "sharpe": None, "fitness": None, "turnover": None,
         "status": "ERROR"},  # status filter
    ]
    segs = extract_top_segments(records, min_score=20, top_n=5)
    exprs = [s.expr for s in segs]
    assert any("ts_rank(close, 252)" in e for e in exprs)
    assert any("ts_corr(close, volume, 20)" in e for e in exprs)
    assert "rank(close)" not in exprs  # too low
    assert "broken_thing" not in exprs  # ERROR


def test_extract_top_segments_diversifies_by_family():
    # 3 ts_corr_pv + 1 intraday_range + 1 group_neutral; with diversify_by_family
    # only one ts_corr_pv should appear in the first round
    records = [
        {"expr": "rank(ts_corr(close, volume, 20))", "sharpe": 1.0, "fitness": 0.55,
         "turnover": 0.18, "status": "COMPLETE"},
        {"expr": "rank(ts_corr(close, volume, 60))", "sharpe": 0.9, "fitness": 0.50,
         "turnover": 0.15, "status": "COMPLETE"},
        {"expr": "rank(ts_corr(vwap, volume, 20))", "sharpe": 0.85, "fitness": 0.48,
         "turnover": 0.16, "status": "COMPLETE"},
        {"expr": "rank((high - low) / close)", "sharpe": 0.8, "fitness": 0.40,
         "turnover": 0.20, "status": "COMPLETE"},
        {"expr": "rank(group_zscore(returns, sector))", "sharpe": 0.7, "fitness": 0.30,
         "turnover": 0.18, "status": "COMPLETE"},
    ]
    segs = extract_top_segments(records, min_score=20, top_n=3, diversify_by_family=True)
    families = [s.family for s in segs]
    # First 3 should cover 3 different families
    assert len(set(families)) == 3


def test_extract_top_segments_returns_empty_when_no_records():
    assert extract_top_segments([]) == []


def test_format_crossover_block_renders_table():
    segs = [
        Segment(expr="rank(ts_corr(close, volume, 20))", score=70.0,
                sharpe=1.02, fitness=0.56, turnover=0.17, family="ts_corr_pv"),
        Segment(expr="rank(ts_rank(close, 252) * (-ts_delta(close, 3) / close))",
                score=72.0, sharpe=1.47, fitness=0.77, turnover=0.46,
                family="ts_rank_close"),
    ]
    out = format_crossover_block(segs)
    assert "## Cross-Over Candidates" in out
    assert "ts_corr_pv" in out
    assert "ts_rank_close" in out
    assert "1.47" in out
    assert "Recombination patterns" in out


def test_format_crossover_block_empty_returns_empty_string():
    assert format_crossover_block([]) == ""
