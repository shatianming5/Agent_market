"""Tests for the learned routing policy + hybrid decider + trainer CLI."""
from __future__ import annotations

import importlib.util
import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from agent_market.wq_brain.routing import (
    ACTION_BUBBLE_UP,
    ACTION_DEEPER,
    ACTION_JUMP_ROOT,
    ACTION_STAY,
    RoutingState,
)
from agent_market.wq_brain.routing_policy import (
    ACTIONS,
    FEATURE_NAMES,
    LearnedPolicy,
    featurise,
    hybrid_decide,
    policy_path,
    samples_from_history,
)
from agent_market.wq_brain.tried_log import (
    ALTITUDE_L1_REGION_UNIVERSE,
    ALTITUDE_L2_OP_FAMILY,
    ALTITUDE_L3_SLOT_PARAM,
    ALTITUDE_L4_NUMERIC_TWEAK,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "wq_brain.py"


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


def test_empty_policy_has_zero_weights_per_action():
    p = LearnedPolicy.empty()
    for a in ACTIONS:
        for f in FEATURE_NAMES:
            assert p.weights[a][f] == 0.0
        assert p.weights[a]["_bias"] == 0.0


def test_featurise_clips_each_axis_into_expected_range():
    state = RoutingState(
        last_delta_U=10.0,  # → clipped to 2.0
        mean_recent_delta_U=-5.0,  # → -2.0
        stall_count=20,  # → 5.0
        osc_count=3,
        cross_panel_conflict=2.0,  # → 1.0
        diagnosis_scope="local",
        current_altitude=ALTITUDE_L1_REGION_UNIVERSE,
    )
    f = featurise(state)
    assert f["last_delta_U"] == 2.0
    assert f["mean_recent_delta_U"] == -2.0
    assert f["stall_count"] == 5.0
    assert f["cross_panel_conflict"] == 1.0
    assert f["altitude_rank"] == 3.0  # L1 = highest rank


def test_train_increases_log_likelihood_of_taken_action():
    state = RoutingState(
        last_delta_U=0.4, mean_recent_delta_U=0.3,
        stall_count=0, osc_count=0, cross_panel_conflict=0.0,
        diagnosis_scope="local", current_altitude=ALTITUDE_L3_SLOT_PARAM,
    )
    samples = [(state, ACTION_STAY, 1.0)] * 50
    p = LearnedPolicy.empty()
    p.train(samples, epochs=30, lr=0.5)
    action, margin, scores = p.predict(state)
    assert action == ACTION_STAY
    assert margin > 0


def test_hybrid_decide_uses_policy_when_margin_high():
    policy = LearnedPolicy.empty()
    # Hand-craft weights so stay clearly wins.
    policy.weights[ACTION_STAY]["_bias"] = 1.0
    policy.weights[ACTION_BUBBLE_UP]["_bias"] = 0.0
    policy.weights[ACTION_DEEPER]["_bias"] = 0.0
    policy.weights[ACTION_JUMP_ROOT]["_bias"] = 0.0
    policy.training_samples = 10
    state = RoutingState(
        last_delta_U=0.1, mean_recent_delta_U=0.1,
        stall_count=0, osc_count=0, cross_panel_conflict=0.0,
        diagnosis_scope="local", current_altitude=ALTITUDE_L3_SLOT_PARAM,
    )
    decision = hybrid_decide(state, policy=policy, margin_min=0.1)
    assert decision.action == ACTION_STAY
    assert "Learned policy" in decision.rationale


def test_hybrid_decide_falls_back_to_rule_when_policy_uncertain():
    policy = LearnedPolicy.empty()
    policy.training_samples = 5  # trained but…
    # …all biases equal → margin = 0 → fall back to rule.
    state = RoutingState(
        stall_count=3, current_altitude=ALTITUDE_L4_NUMERIC_TWEAK,
        diagnosis_scope="local",
    )
    decision = hybrid_decide(state, policy=policy, margin_min=0.05)
    assert decision.action == ACTION_BUBBLE_UP  # rule decision


def test_hybrid_decide_forces_rule_path_when_cross_panel_conflict_high():
    policy = LearnedPolicy.empty()
    policy.weights[ACTION_STAY]["_bias"] = 5.0  # would dominate normally
    policy.training_samples = 10
    state = RoutingState(
        cross_panel_conflict=0.9, current_altitude=ALTITUDE_L3_SLOT_PARAM,
        diagnosis_scope="local",
    )
    decision = hybrid_decide(state, policy=policy)
    # Rule path forces jump_root for cross_panel_conflict > τ_root.
    assert decision.action == ACTION_JUMP_ROOT


def test_hybrid_decide_falls_back_when_policy_untrained():
    policy = LearnedPolicy.empty()
    # training_samples == 0 by default.
    state = RoutingState(stall_count=3, current_altitude=ALTITUDE_L3_SLOT_PARAM)
    decision = hybrid_decide(state, policy=policy)
    assert decision.action == ACTION_BUBBLE_UP  # rule path


def test_samples_from_history_infers_stay_when_altitude_unchanged():
    rows = [
        {"ts": 1, "altitude": ALTITUDE_L3_SLOT_PARAM, "delta_U": 0.1},
        {"ts": 2, "altitude": ALTITUDE_L3_SLOT_PARAM, "delta_U": 0.2},
    ]
    samples = samples_from_history(rows)
    assert len(samples) == 1
    state, action, reward = samples[0]
    assert action == ACTION_STAY
    assert reward == pytest.approx(0.2)
    assert state.current_altitude == ALTITUDE_L3_SLOT_PARAM


def test_samples_from_history_infers_bubble_up_and_deeper():
    rows = [
        {"ts": 1, "altitude": ALTITUDE_L4_NUMERIC_TWEAK, "delta_U": 0.0},
        {"ts": 2, "altitude": ALTITUDE_L3_SLOT_PARAM, "delta_U": 0.3},   # bubble up
        {"ts": 3, "altitude": ALTITUDE_L4_NUMERIC_TWEAK, "delta_U": 0.1},  # deeper
    ]
    samples = samples_from_history(rows)
    assert [a for _, a, _ in samples] == [ACTION_BUBBLE_UP, ACTION_DEEPER]


def test_samples_from_history_skips_rows_without_altitude():
    rows = [
        {"ts": 1, "delta_U": 0.1},  # legacy row, no altitude
        {"ts": 2, "altitude": ALTITUDE_L3_SLOT_PARAM, "delta_U": 0.2},
    ]
    samples = samples_from_history(rows)
    assert samples == []


def test_policy_save_and_load_round_trip(tmp_path):
    p = LearnedPolicy.empty()
    p.weights[ACTION_STAY]["_bias"] = 0.123
    p.training_samples = 7
    target = tmp_path / "policy.json"
    p.save(target)
    loaded = LearnedPolicy.load(target)
    assert loaded is not None
    assert loaded.weights[ACTION_STAY]["_bias"] == pytest.approx(0.123)
    assert loaded.training_samples == 7


def test_policy_load_missing_returns_none(tmp_path):
    assert LearnedPolicy.load(tmp_path / "no_such_policy.json") is None


# ── CLI integration ─────────────────────────────────────────────────────


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "_wq_brain_cli_policy", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_train_policy_cli_writes_weights_to_disk(isolated_artifacts):
    from agent_market.wq_brain.colony import colony_run_dir
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.tried_log import append_tried

    # Seed a manifest pointing at one panel.
    (colony_run_dir("policy_train").parent).mkdir(parents=True, exist_ok=True)
    (colony_run_dir("policy_train")).mkdir(parents=True, exist_ok=True)
    (colony_run_dir("policy_train") / "manifest.json").write_text(
        json.dumps({"panels": [{"tag": "policy_panel"}]}), encoding="utf-8"
    )
    # Two enriched rows: STAY transition with reward 0.3.
    append_tried(
        tried_exprs_path("policy_panel"),
        expr="rank(close)", sharpe=1.3, fitness=1.0, turnover=0.20,
        alpha_id="parent", status="COMPLETE",
        region="USA", universe="TOP500", decay=6,
        evidence_type="seed", altitude=ALTITUDE_L3_SLOT_PARAM,
        parent_alpha_id=None, delta_U=0.0,
    )
    append_tried(
        tried_exprs_path("policy_panel"),
        expr="rank(close) + rank(volume)", sharpe=1.6, fitness=1.2, turnover=0.18,
        alpha_id="child", status="COMPLETE",
        region="USA", universe="TOP500", decay=6,
        evidence_type="mutation", altitude=ALTITUDE_L3_SLOT_PARAM,
        parent_alpha_id="parent", delta_U=0.3,
    )

    module = _load_cli()
    parser = module._build_parser()
    args = parser.parse_args([
        "colony", "train-policy",
        "--colony-tag", "policy_train",
        "--epochs", "5", "--lr", "0.2",
    ])
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit) as excinfo:
        args.func(args)
    assert excinfo.value.code == 0
    payload = json.loads(buf.getvalue())
    assert payload["ok"] is True
    assert payload["training_samples"] >= 1
    assert Path(payload["policy_path"]).exists()
    loaded = LearnedPolicy.load(Path(payload["policy_path"]))
    assert loaded is not None
    assert loaded.training_samples >= 1
