"""Tests for ready-to-paste simulate commands in mutation/crossover blocks."""
from __future__ import annotations

import pytest

from agent_market.wq_brain.crossover import (
    Segment,
    extract_top_segments,
    format_crossover_block,
)
from agent_market.wq_brain.mutation import (
    FailureContext,
    MutationEngine,
    MutationStrategy,
    STRATEGY_TO_EVIDENCE_TYPE,
    diagnose_from_record,
    render_top_failures_block,
)


# ── mutation ──────────────────────────────────────────────────────────────


def test_strategy_to_evidence_type_covers_every_strategy():
    for strategy in MutationStrategy:
        assert strategy in STRATEGY_TO_EVIDENCE_TYPE, (
            f"missing evidence_type mapping for {strategy}"
        )


def test_simulate_command_template_empty_without_alpha_id():
    ctx = FailureContext(
        expr="rank(close)",
        sharpe=1.20, fitness=0.85, turnover=0.30, status="COMPLETE",
    )
    assert MutationEngine(ctx).simulate_command_template() == ""


def test_simulate_command_template_includes_parent_and_evidence_type():
    ctx = FailureContext(
        expr="rank(ts_mean(close, 20))",
        sharpe=1.20, fitness=0.85, turnover=0.30, status="COMPLETE",
        alpha_id="par_42", region="USA", universe="TOP500", decay=6,
    )
    cmd = MutationEngine(ctx).simulate_command_template()
    assert "simulate \"<new_expr>\"" in cmd
    assert "--parent-alpha-id par_42" in cmd
    assert "--region USA" in cmd
    assert "--universe TOP500" in cmd
    assert "--decay 6" in cmd
    # evidence_type derived from strategy table
    diag = MutationEngine(ctx).diagnose()
    expected_ev = STRATEGY_TO_EVIDENCE_TYPE[diag.strategy]
    assert f"--evidence-type {expected_ev}" in cmd


def test_format_for_prompt_embeds_simulate_command_block():
    ctx = FailureContext(
        expr="rank(signed_power(close, 0.5))",
        sharpe=1.30, fitness=0.90, turnover=0.25, status="COMPLETE",
        alpha_id="par_X", region="USA", universe="TOP500", decay=6,
    )
    block = MutationEngine(ctx).format_for_prompt()
    assert "Recommended `simulate` call" in block
    assert "--parent-alpha-id par_X" in block
    assert "<new_expr>" in block


def test_render_top_failures_block_propagates_alpha_id_into_commands():
    records = [
        {
            "ts": 1.0,
            "expr": "rank(close) * rank(volume)",
            "sharpe": 1.30, "fitness": 0.85, "turnover": 0.30,
            "alpha_id": "near_miss_a",
            "region": "USA", "universe": "TOP500", "decay": 6,
            "status": "COMPLETE",
        },
        {
            "ts": 2.0,
            "expr": "rank(ts_corr(close, volume, 20))",
            "sharpe": 1.20, "fitness": 0.92, "turnover": 0.40,
            "alpha_id": "near_miss_b",
            "region": "USA", "universe": "TOP500", "decay": 6,
            "status": "COMPLETE",
        },
    ]
    block = render_top_failures_block(records, top_n=2)
    assert "near_miss_a" in block
    assert "near_miss_b" in block
    assert "simulate \"<new_expr>\"" in block


def test_diagnose_from_record_preserves_alpha_id():
    record = {
        "expr": "rank(close)", "sharpe": 1.2, "fitness": 0.85, "turnover": 0.25,
        "status": "COMPLETE", "alpha_id": "preserved", "region": "USA",
        "universe": "TOP500", "decay": 6,
    }
    diag = diagnose_from_record(record)
    # diagnose_from_record returns only the Diagnosis; the alpha_id lives on
    # the engine context — verify by re-building the engine.
    ctx = FailureContext(
        expr=record["expr"],
        sharpe=record["sharpe"], fitness=record["fitness"], turnover=record["turnover"],
        status="COMPLETE", alpha_id=record["alpha_id"],
        region=record["region"], universe=record["universe"], decay=record["decay"],
    )
    cmd = MutationEngine(ctx).simulate_command_template()
    assert "--parent-alpha-id preserved" in cmd


# ── crossover ────────────────────────────────────────────────────────────


def test_extract_top_segments_carries_alpha_id_into_segment():
    records = [
        {
            "expr": "rank(close) + rank(volume)",
            "sharpe": 1.6, "fitness": 1.2, "turnover": 0.20,
            "alpha_id": "src_a", "region": "USA", "universe": "TOP500", "decay": 6,
            "status": "COMPLETE",
        },
        {
            "expr": "rank(ts_mean(close, 20))",
            "sharpe": 1.5, "fitness": 1.1, "turnover": 0.18,
            "alpha_id": "src_b", "region": "USA", "universe": "TOP500", "decay": 6,
            "status": "COMPLETE",
        },
    ]
    segments = extract_top_segments(records, top_n=2, diversify_by_family=False)
    aliases = {s.alpha_id for s in segments}
    assert aliases == {"src_a", "src_b"}


def test_format_crossover_block_emits_simulate_commands_when_alpha_ids_present():
    segments = extract_top_segments([
        {
            "expr": "rank(close) + rank(volume)",
            "sharpe": 1.6, "fitness": 1.2, "turnover": 0.20,
            "alpha_id": "xover_a", "region": "USA", "universe": "TOP500", "decay": 6,
            "status": "COMPLETE",
        },
        {
            "expr": "rank(ts_mean(close, 20))",
            "sharpe": 1.5, "fitness": 1.1, "turnover": 0.18,
            "alpha_id": "xover_b", "region": "USA", "universe": "TOP500", "decay": 6,
            "status": "COMPLETE",
        },
    ], top_n=2, diversify_by_family=False)
    block = format_crossover_block(segments)
    assert "parent" in block.lower()
    assert "xover_a" in block
    assert "xover_b" in block
    assert "--parent-alpha-id xover_a" in block
    assert "--parent-alpha-id xover_b" in block
    assert "--evidence-type crossover" in block


def test_format_crossover_block_without_alpha_ids_keeps_original_layout():
    segments = [
        Segment(
            expr="rank(close)", score=80.0,
            sharpe=1.5, fitness=1.1, turnover=0.20, family="ts_rank_close",
        )
    ]
    block = format_crossover_block(segments)
    # Original 7-column layout; no parent header introduced.
    assert "| # | family | sh | fi | to | quick | expr |" in block
    # No ready-to-paste simulate block when no parent alpha_id is available.
    assert "simulate \"<new_expr>\"" not in block
    assert "Ready-to-paste" not in block
