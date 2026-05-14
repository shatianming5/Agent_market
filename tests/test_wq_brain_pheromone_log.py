"""Tests for the typed pheromone-link fields on tried_exprs.jsonl."""
from __future__ import annotations

import json
from pathlib import Path

from agent_market.wq_brain.tried_log import (
    ALTITUDE_L1_REGION_UNIVERSE,
    ALTITUDE_L2_OP_FAMILY,
    ALTITUDE_L3_SLOT_PARAM,
    ALTITUDE_L4_NUMERIC_TWEAK,
    ALTITUDES,
    EVIDENCE_TYPES,
    _op_signature,
    _strip_numbers,
    append_tried,
    classify_altitude,
    read_tried,
    utility_score,
)


def test_append_tried_preserves_legacy_schema_when_pheromone_fields_omitted(tmp_path):
    path = tmp_path / "tried.jsonl"
    append_tried(
        path,
        expr="rank(close)",
        sharpe=1.4,
        fitness=1.1,
        turnover=0.21,
        alpha_id="aBcD1234",
        status="COMPLETE",
        region="USA",
        universe="TOP500",
        decay=6,
    )
    rows = read_tried(path)
    assert len(rows) == 1
    row = rows[0]
    assert row["expr"] == "rank(close)"
    assert row["sharpe"] == 1.4
    assert row["region"] == "USA"
    assert "evidence_type" not in row
    assert "altitude" not in row
    assert "parent_alpha_id" not in row
    assert "delta_U" not in row


def test_append_tried_records_pheromone_fields_when_supplied(tmp_path):
    path = tmp_path / "tried.jsonl"
    append_tried(
        path,
        expr="rank(close) * rank(volume)",
        sharpe=1.51,
        fitness=1.18,
        turnover=0.22,
        alpha_id="aBcD1234",
        status="COMPLETE",
        region="USA",
        universe="TOP500",
        decay=6,
        evidence_type="op_swap",
        altitude=ALTITUDE_L2_OP_FAMILY,
        parent_alpha_id="parent_xyz",
        delta_U=0.42,
    )
    row = read_tried(path)[0]
    assert row["evidence_type"] == "op_swap"
    assert row["altitude"] == ALTITUDE_L2_OP_FAMILY
    assert row["parent_alpha_id"] == "parent_xyz"
    assert row["delta_U"] == 0.42
    assert "op_swap" in EVIDENCE_TYPES
    assert ALTITUDE_L2_OP_FAMILY in ALTITUDES


def test_read_tried_tolerates_mixed_legacy_and_pheromone_rows(tmp_path):
    path = tmp_path / "tried.jsonl"
    legacy = {
        "ts": 100.0, "expr": "rank(close)", "sharpe": 1.30, "fitness": 1.0,
        "turnover": 0.30, "alpha_id": "old", "status": "COMPLETE", "error": None,
        "region": "USA", "universe": "TOP500", "decay": 6,
    }
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    append_tried(
        path,
        expr="rank(close) * rank(volume)",
        sharpe=1.45, fitness=1.10, turnover=0.20,
        alpha_id="new", status="COMPLETE",
        region="USA", universe="TOP500", decay=6,
        evidence_type="mutation", altitude=ALTITUDE_L3_SLOT_PARAM,
        parent_alpha_id="old", delta_U=0.15,
    )
    rows = read_tried(path)
    assert len(rows) == 2
    legacy_row, new_row = rows
    assert "evidence_type" not in legacy_row
    assert new_row["evidence_type"] == "mutation"
    assert new_row["parent_alpha_id"] == "old"


def test_utility_score_rewards_sharpe_fitness_penalises_turnover():
    base = utility_score(sharpe=1.5, fitness=1.2, turnover=0.20)
    higher_sharpe = utility_score(sharpe=1.6, fitness=1.2, turnover=0.20)
    higher_turnover = utility_score(sharpe=1.5, fitness=1.2, turnover=0.40)
    assert higher_sharpe > base
    assert higher_turnover < base
    # delta_U convention: child - parent
    assert higher_sharpe - base > 0
    # zero-defaults must not raise even with missing pieces
    assert utility_score(sharpe=None, fitness=None, turnover=None) == 0.0


def test_classify_altitude_region_swap_returns_L1():
    parent = {"expr": "rank(close)", "region": "USA", "universe": "TOP500"}
    child = {"expr": "rank(close)", "region": "CHN", "universe": "TOP500"}
    assert classify_altitude(parent, child) == ALTITUDE_L1_REGION_UNIVERSE


def test_classify_altitude_op_family_change_returns_L2():
    parent = {"expr": "rank(close)", "region": "USA", "universe": "TOP500"}
    child = {"expr": "ts_rank(close, 5)", "region": "USA", "universe": "TOP500"}
    assert classify_altitude(parent, child) == ALTITUDE_L2_OP_FAMILY


def test_classify_altitude_param_only_returns_L3_or_L4():
    parent = {
        "expr": "rank(ts_mean(close, 20))",
        "region": "USA", "universe": "TOP500",
    }
    # Numeric-only tweak (window 20 → 30): same ops, same shape, just number.
    numeric_only = {
        "expr": "rank(ts_mean(close, 30))",
        "region": "USA", "universe": "TOP500",
    }
    assert classify_altitude(parent, numeric_only) == ALTITUDE_L4_NUMERIC_TWEAK
    # Structural slot change without changing operator family.
    structural = {
        "expr": "rank(ts_mean(open, 20))",
        "region": "USA", "universe": "TOP500",
    }
    assert classify_altitude(parent, structural) == ALTITUDE_L3_SLOT_PARAM


def test_classify_altitude_seed_when_no_parent_returns_L1():
    assert classify_altitude(None, {"expr": "rank(close)", "region": "USA"}) == ALTITUDE_L1_REGION_UNIVERSE


def test_op_signature_extracts_function_names_in_order():
    assert _op_signature("rank(ts_mean(close, 20)) + signed_power(volume, 0.3)") == (
        "rank", "ts_mean", "signed_power",
    )


def test_strip_numbers_normalises_literals():
    assert _strip_numbers("ts_mean(close, 20) * 0.5") == "ts_mean(close, #) * #"
