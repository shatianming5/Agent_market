"""Tests for the directed mutation engine."""
from __future__ import annotations

from agent_market.wq_brain.mutation import (
    FailureContext,
    MutationEngine,
    MutationStrategy,
    diagnose_from_record,
    render_top_failures_block,
)


def _make_ctx(expr="rank(close)", **kw):
    return FailureContext(expr=expr, **kw)


def test_negative_sharpe_recommends_signal_flip():
    ctx = _make_ctx(expr="rank(ts_corr(close, volume, 20))",
                    sharpe=-1.0, fitness=-0.5, turnover=0.17)
    diag = MutationEngine(ctx).diagnose()
    assert diag.strategy == MutationStrategy.MUTATE_SIGNAL_TYPE


def test_high_turnover_blocking_fitness_recommends_reduce_turnover():
    # Top alpha case: sh=1.47 fi=0.77 to=0.46 — should suggest hump
    ctx = _make_ctx(expr="rank(ts_rank(close, 252) * (-ts_delta(close, 3) / close))",
                    sharpe=1.47, fitness=0.77, turnover=0.46)
    diag = MutationEngine(ctx).diagnose()
    assert diag.strategy == MutationStrategy.REDUCE_TURNOVER
    assert any("hump" in str(c) for c in diag.details["candidates"])


def test_already_humped_alpha_does_not_recommend_reduce_turnover():
    ctx = _make_ctx(expr="hump(rank(ts_corr(close, volume, 20)), 0.01)",
                    sharpe=1.10, fitness=0.65, turnover=0.45)
    diag = MutationEngine(ctx).diagnose()
    # Should fall through to other strategies — NOT reduce_turnover (already wrapped)
    assert diag.strategy != MutationStrategy.REDUCE_TURNOVER


def test_weak_signal_recommends_operator_swap():
    ctx = _make_ctx(expr="rank(close)", sharpe=0.05, fitness=0.01, turnover=0.20)
    diag = MutationEngine(ctx).diagnose()
    assert diag.strategy == MutationStrategy.MUTATE_OPERATOR


def test_mid_tier_no_nonlinear_recommends_nonlinear():
    ctx = _make_ctx(expr="rank(ts_mean(close, 20) - close)",
                    sharpe=1.20, fitness=0.70, turnover=0.18)
    diag = MutationEngine(ctx).diagnose()
    assert diag.strategy == MutationStrategy.MUTATE_NONLINEAR


def test_default_falls_back_to_window():
    # sharpe ≥ 1.5 (skips mid-tier nonlinear branch), has normalization,
    # multi-signal, low turnover → falls through to default window-tuning.
    ctx = _make_ctx(expr="rank(group_zscore(ts_mean(close, 20) * volume, sector))",
                    sharpe=1.55, fitness=0.95, turnover=0.18)
    diag = MutationEngine(ctx).diagnose()
    assert diag.strategy == MutationStrategy.MUTATE_WINDOW
    assert "current_windows" in diag.details


def test_error_status_recommends_regenerate():
    ctx = _make_ctx(expr="rank(close)", status="ERROR", error="syntax")
    diag = MutationEngine(ctx).diagnose()
    assert diag.strategy == MutationStrategy.REGENERATE_FULL


def test_format_for_prompt_renders_markdown():
    ctx = _make_ctx(expr="rank(ts_rank(close, 252) * (-ts_delta(close, 3) / close))",
                    sharpe=1.47, fitness=0.77, turnover=0.46)
    md = MutationEngine(ctx).format_for_prompt()
    assert "### Mutation Engine Diagnosis" in md
    assert "reduce_turnover" in md
    assert "hump" in md.lower()


def test_diagnose_from_record_works_with_jsonl_row():
    row = {
        "expr": "rank(ts_corr(close, volume, 20))",
        "sharpe": -1.02,
        "fitness": -0.56,
        "turnover": 0.17,
        "status": "COMPLETE",
    }
    diag = diagnose_from_record(row)
    assert diag is not None
    assert diag.strategy == MutationStrategy.MUTATE_SIGNAL_TYPE


def test_diagnose_from_record_returns_none_for_blank():
    assert diagnose_from_record({}) is None
    assert diagnose_from_record({"expr": ""}) is None


def test_render_top_failures_picks_near_misses():
    records = [
        {"expr": "rank(close)", "sharpe": 0.05, "fitness": 0.01,
         "turnover": 0.2, "status": "COMPLETE"},  # too weak — sharpe filter
        {"expr": "rank(ts_rank(close, 252) * (-ts_delta(close, 3) / close))",
         "sharpe": 1.47, "fitness": 0.77, "turnover": 0.46, "status": "COMPLETE"},  # passes filter
        {"expr": "rank(volume)", "sharpe": 1.0, "fitness": 0.5,
         "turnover": 0.30, "status": "COMPLETE"},  # passes filter
        {"expr": "rank(closed_thing)", "sharpe": None, "fitness": None,
         "turnover": None, "status": "ERROR"},  # error filter
    ]
    out = render_top_failures_block(records, top_n=3)
    assert "Mutation Hints" in out
    # Only the 2 passing-filter records should appear
    assert "ts_rank(close, 252)" in out
    assert "rank(volume)" in out
    assert out.count("### Mutation Engine Diagnosis") == 2


def test_render_top_failures_empty_when_no_candidates():
    records = [
        {"expr": "rank(close)", "sharpe": 1.5, "fitness": 1.2, "turnover": 0.18,
         "status": "COMPLETE"},  # already passes — filtered out
    ]
    out = render_top_failures_block(records)
    assert out == ""


def test_quick_score_synthetic():
    # Best alpha quick-score should be > 50
    ctx = _make_ctx(sharpe=1.47, fitness=0.77, turnover=0.46)
    score = ctx.quick_score()
    assert score > 50

    # Failing alpha quick-score lower
    ctx2 = _make_ctx(sharpe=0.1, fitness=0.05, turnover=0.5)
    assert ctx2.quick_score() < ctx.quick_score()
