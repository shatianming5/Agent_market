"""Tests for the routing policy and panel routing advisory wiring."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agent_market.wq_brain.routing import (
    ACTION_BUBBLE_UP,
    ACTION_DEEPER,
    ACTION_JUMP_ROOT,
    ACTION_STAY,
    RoutingDecision,
    RoutingState,
    RoutingThresholds,
    decide,
    state_from_rows,
)
from agent_market.wq_brain.tried_log import (
    ALTITUDE_L1_REGION_UNIVERSE,
    ALTITUDE_L2_OP_FAMILY,
    ALTITUDE_L3_SLOT_PARAM,
    ALTITUDE_L4_NUMERIC_TWEAK,
)


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


def test_decide_jump_root_when_cross_panel_conflict_exceeds_tau_root():
    state = RoutingState(
        last_delta_U=0.01,
        cross_panel_conflict=0.8,
        current_altitude=ALTITUDE_L2_OP_FAMILY,
    )
    d = decide(state)
    assert d.action == ACTION_JUMP_ROOT
    assert d.target_altitude == ALTITUDE_L1_REGION_UNIVERSE


def test_decide_jump_root_when_diagnosis_is_global():
    state = RoutingState(
        last_delta_U=0.2,
        diagnosis_scope="global",
        current_altitude=ALTITUDE_L3_SLOT_PARAM,
    )
    assert decide(state).action == ACTION_JUMP_ROOT


def test_decide_bubble_up_when_stalled():
    state = RoutingState(
        last_delta_U=0.0,
        stall_count=3,
        current_altitude=ALTITUDE_L4_NUMERIC_TWEAK,
    )
    d = decide(state)
    assert d.action == ACTION_BUBBLE_UP
    assert d.target_altitude == ALTITUDE_L3_SLOT_PARAM


def test_decide_bubble_up_when_oscillating():
    state = RoutingState(
        last_delta_U=0.0,
        osc_count=2,
        current_altitude=ALTITUDE_L3_SLOT_PARAM,
    )
    d = decide(state)
    assert d.action == ACTION_BUBBLE_UP
    assert d.target_altitude == ALTITUDE_L2_OP_FAMILY


def test_decide_deeper_when_child_diagnosis_with_positive_gain():
    state = RoutingState(
        last_delta_U=0.3,
        diagnosis_scope="child",
        current_altitude=ALTITUDE_L2_OP_FAMILY,
    )
    d = decide(state)
    assert d.action == ACTION_DEEPER
    assert d.target_altitude == ALTITUDE_L3_SLOT_PARAM


def test_decide_stay_when_recent_gain_positive_and_local():
    state = RoutingState(
        last_delta_U=0.08,
        mean_recent_delta_U=0.05,
        diagnosis_scope="local",
        current_altitude=ALTITUDE_L3_SLOT_PARAM,
    )
    d = decide(state)
    assert d.action == ACTION_STAY
    assert d.target_altitude == ALTITUDE_L3_SLOT_PARAM


def test_thresholds_can_be_overridden():
    state = RoutingState(stall_count=1, current_altitude=ALTITUDE_L3_SLOT_PARAM)
    # Default stall_threshold = 3 → would stay
    assert decide(state).action == ACTION_STAY
    # Tighten threshold to 1 → bubble_up
    d = decide(state, RoutingThresholds(stall_threshold=1))
    assert d.action == ACTION_BUBBLE_UP


def test_state_from_rows_computes_stall_and_osc_counts():
    now = time.time()
    rows = [
        # Sign-flipping deltas: +0.1, -0.05, +0.1 → 2 sign flips
        {"ts": now - 4, "delta_U": 0.10, "altitude": ALTITUDE_L3_SLOT_PARAM},
        {"ts": now - 3, "delta_U": -0.05, "altitude": ALTITUDE_L3_SLOT_PARAM},
        {"ts": now - 2, "delta_U": 0.10, "altitude": ALTITUDE_L3_SLOT_PARAM},
        # Small ΔU rows feed stall_count
        {"ts": now - 1, "delta_U": 0.01, "altitude": ALTITUDE_L3_SLOT_PARAM},
        {"ts": now,     "delta_U": 0.02, "altitude": ALTITUDE_L3_SLOT_PARAM},
    ]
    state = state_from_rows(rows, window=5)
    assert state.last_delta_U == pytest.approx(0.02)
    assert state.osc_count == 2
    assert state.stall_count == 2  # last 3 are 0.10, 0.01, 0.02 -> two with |ΔU| < 0.05
    assert state.current_altitude == ALTITUDE_L3_SLOT_PARAM


def test_state_from_rows_handles_legacy_rows_without_delta_U():
    """Backfill ΔU from utility_score(child) − utility_score(parent)."""
    now = time.time()
    rows = [
        {"ts": now - 1, "sharpe": 1.0, "fitness": 0.8, "turnover": 0.30},
        {"ts": now,     "sharpe": 1.5, "fitness": 1.1, "turnover": 0.25},
    ]
    state = state_from_rows(rows, window=2)
    # ΔU = (1.5+1.1-0.5*0.25) - (1.0+0.8-0.5*0.30) = 2.475 - 1.65 = 0.825
    assert state.last_delta_U == pytest.approx(0.825, abs=0.01)


def test_state_from_rows_empty_returns_defaults():
    state = state_from_rows([])
    assert state.last_delta_U == 0.0
    assert state.stall_count == 0
    assert state.osc_count == 0


def test_routing_decision_renders_prompt_block_with_action_and_altitude():
    decision = RoutingDecision(
        action=ACTION_BUBBLE_UP,
        target_altitude=ALTITUDE_L2_OP_FAMILY,
        rationale="example",
        inputs={"stall_count": 3},
    )
    block = decision.to_prompt_block()
    assert "COLONY ROUTING ADVISORY" in block
    assert "BUBBLE_UP" in block
    assert ALTITUDE_L2_OP_FAMILY in block
    assert "example" in block


def test_compute_panel_routing_advisory_persists_file(isolated_artifacts):
    """Colony wiring: advisory JSON gets written next to the colony tag."""
    from agent_market.wq_brain.colony import (
        PanelSpec,
        compute_panel_routing_advisory,
        routing_advisory_path,
        write_panel_routing_advisory,
    )
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.tried_log import append_tried

    panel = PanelSpec(tag="routing_panel", region="USA", universe="TOP500")
    # Seed with a stalled history to force BUBBLE_UP.
    for i in range(4):
        append_tried(
            tried_exprs_path("routing_panel"),
            expr=f"rank(close) + {i}",
            sharpe=1.30 + i * 0.001, fitness=1.00, turnover=0.20,
            alpha_id=f"r{i}", status="COMPLETE",
            region="USA", universe="TOP500", decay=6,
            evidence_type="numeric_tweak", altitude=ALTITUDE_L4_NUMERIC_TWEAK,
            parent_alpha_id=None, delta_U=0.0,
        )
    decision = compute_panel_routing_advisory("routing_colony", panel)
    write_panel_routing_advisory("routing_colony", panel.tag, decision)
    path = routing_advisory_path("routing_colony", panel.tag)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["action"] == ACTION_BUBBLE_UP


def test_routing_advisory_block_renders_into_prompt(isolated_artifacts):
    """prompt_builder picks up the advisory file when present."""
    from agent_market.wq_brain.colony import (
        PanelSpec,
        compute_panel_routing_advisory,
        write_panel_routing_advisory,
    )
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.prompt_builder import _routing_advisory_block
    from agent_market.wq_brain.tried_log import append_tried

    panel = PanelSpec(tag="prompt_panel", region="USA", universe="TOP500")
    append_tried(
        tried_exprs_path("prompt_panel"),
        expr="rank(close)",
        sharpe=1.30, fitness=1.00, turnover=0.30,
        alpha_id="root", status="COMPLETE",
        region="USA", universe="TOP500", decay=6,
        evidence_type="seed", altitude=ALTITUDE_L1_REGION_UNIVERSE,
        parent_alpha_id=None, delta_U=0.0,
    )
    decision = compute_panel_routing_advisory("prompt_colony", panel)
    write_panel_routing_advisory("prompt_colony", panel.tag, decision)
    block = _routing_advisory_block("prompt_panel")
    assert "COLONY ROUTING ADVISORY" in block
    assert decision.action.upper() in block
