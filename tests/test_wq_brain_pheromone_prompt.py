"""Tests for the typed pheromone-trail block in prompt_builder."""
from __future__ import annotations

from agent_market.wq_brain.prompt_builder import _pheromone_trail_block


def test_pheromone_trail_block_empty_when_no_metadata():
    rows = [
        {"expr": "rank(close)", "sharpe": 1.3, "fitness": 1.0, "turnover": 0.20,
         "ts": 100.0, "status": "COMPLETE", "region": "USA", "universe": "TOP500"},
    ]
    assert _pheromone_trail_block(rows) == ""


def test_pheromone_trail_block_orders_by_delta_U_desc_and_includes_summary():
    rows = [
        {
            "expr": "rank(close) * rank(volume)",
            "sharpe": 1.6, "fitness": 1.2, "turnover": 0.18,
            "ts": 200.0, "status": "COMPLETE",
            "region": "USA", "universe": "TOP500", "decay": 6,
            "evidence_type": "op_swap", "altitude": "L2_op_family",
            "parent_alpha_id": "parent_A", "delta_U": 0.45,
        },
        {
            "expr": "rank(ts_mean(close, 30))",
            "sharpe": 1.42, "fitness": 1.05, "turnover": 0.20,
            "ts": 250.0, "status": "COMPLETE",
            "region": "USA", "universe": "TOP500", "decay": 6,
            "evidence_type": "numeric_tweak", "altitude": "L4_numeric_tweak",
            "parent_alpha_id": "parent_B", "delta_U": 0.07,
        },
        {
            "expr": "rank(close)",
            "sharpe": 1.20, "fitness": 0.95, "turnover": 0.25,
            "ts": 100.0, "status": "COMPLETE",
            "region": "USA", "universe": "TOP500", "decay": 6,
            "evidence_type": "mutation", "altitude": "L3_slot_param",
            "parent_alpha_id": "parent_C", "delta_U": -0.18,
        },
        # legacy row — should be ignored
        {"expr": "rank(open)", "sharpe": 1.1, "fitness": 0.9, "turnover": 0.30,
         "ts": 110.0, "status": "COMPLETE", "region": "USA", "universe": "TOP500"},
    ]
    block = _pheromone_trail_block(rows)
    assert "PHEROMONE TRAILS" in block
    # 3 enriched rows, legacy row excluded:
    assert "n=3" in block
    # sorted by ΔU descending: 0.45 row appears before 0.07 row appears before -0.18 row
    pos_high = block.index("+0.450")
    pos_mid = block.index("+0.070")
    pos_low = block.index("-0.180")
    assert pos_high < pos_mid < pos_low
    # summary line lists most-common altitude and evidence types
    assert "Logged altitude mix" in block
    assert "L2_op_family" in block
    assert "L4_numeric_tweak" in block
    assert "L3_slot_param" in block
    assert "Evidence-type mix" in block
    assert "op_swap" in block
    # legacy expression must not be rendered
    assert "rank(open)" not in block
