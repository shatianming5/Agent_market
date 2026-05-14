"""Tests for the Phase-3 colony orchestrator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_market.wq_brain.colony import (
    ColonyConfig,
    PanelSpec,
    SHARED_ALTITUDES,
    collect_shared_pheromones,
    colony_run_dir,
    inject_shared_pheromones,
    parse_panels,
    run_colony,
)
from agent_market.wq_brain.paths import tried_exprs_path
from agent_market.wq_brain.tried_log import (
    ALTITUDE_L1_REGION_UNIVERSE,
    ALTITUDE_L2_OP_FAMILY,
    ALTITUDE_L3_SLOT_PARAM,
    ALTITUDE_L4_NUMERIC_TWEAK,
    append_tried,
    read_tried,
)


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


def test_parse_panels_accepts_simple_and_trailing_comma():
    out = parse_panels("USA:TOP500,USA:TOP1000,")
    assert out == [("USA", "TOP500"), ("USA", "TOP1000")]


def test_parse_panels_rejects_malformed_entries():
    with pytest.raises(ValueError):
        parse_panels("USA-TOP500")
    with pytest.raises(ValueError):
        parse_panels("USA:")
    with pytest.raises(ValueError):
        parse_panels("")


def test_shared_altitudes_only_contains_l1_and_l2():
    assert ALTITUDE_L1_REGION_UNIVERSE in SHARED_ALTITUDES
    assert ALTITUDE_L2_OP_FAMILY in SHARED_ALTITUDES
    assert ALTITUDE_L3_SLOT_PARAM not in SHARED_ALTITUDES
    assert ALTITUDE_L4_NUMERIC_TWEAK not in SHARED_ALTITUDES


def test_collect_shared_pheromones_keeps_only_high_altitude(isolated_artifacts):
    panel = PanelSpec(tag="src_panel", region="USA", universe="TOP500")
    p = tried_exprs_path("src_panel")
    append_tried(
        p, expr="rank(ts_mean(close,5))", sharpe=1.4, fitness=1.1, turnover=0.2,
        alpha_id="A1", status="COMPLETE", region="USA", universe="TOP500",
        decay=6, evidence_type="op_swap", altitude=ALTITUDE_L2_OP_FAMILY,
        parent_alpha_id=None, delta_U=0.32,
    )
    append_tried(
        p, expr="rank(ts_mean(close,9))", sharpe=1.3, fitness=1.0, turnover=0.21,
        alpha_id="A2", status="COMPLETE", region="USA", universe="TOP500",
        decay=6, evidence_type="numeric_tweak", altitude=ALTITUDE_L4_NUMERIC_TWEAK,
        parent_alpha_id="A1", delta_U=-0.04,
    )
    # legacy row (no altitude)
    append_tried(
        p, expr="rank(close)", sharpe=1.0, fitness=0.9, turnover=0.20,
        alpha_id="legacy", status="COMPLETE", region="USA", universe="TOP500",
        decay=6,
    )

    shared = collect_shared_pheromones([panel])
    exprs = {r["expr"] for r in shared}
    assert exprs == {"rank(ts_mean(close,5))"}  # only the L2 row


def test_inject_shared_pheromones_writes_to_target_and_dedupes(isolated_artifacts):
    source = PanelSpec(tag="src", region="USA", universe="TOP500")
    target = PanelSpec(tag="tgt", region="USA", universe="TOP1000")
    append_tried(
        tried_exprs_path("src"),
        expr="rank(volume) * rank(close)", sharpe=1.6, fitness=1.2, turnover=0.18,
        alpha_id="src1", status="COMPLETE", region="USA", universe="TOP500",
        decay=6, evidence_type="op_swap", altitude=ALTITUDE_L2_OP_FAMILY,
        parent_alpha_id=None, delta_U=0.5,
    )
    # Pre-existing row in target with the SAME expression — must not be re-injected.
    append_tried(
        tried_exprs_path("tgt"),
        expr="rank(volume) * rank(close)", sharpe=1.45, fitness=1.05, turnover=0.20,
        alpha_id="tgt_existing", status="COMPLETE", region="USA", universe="TOP1000",
        decay=6,
    )
    # Plus a different existing row so the panel log is non-empty.
    append_tried(
        tried_exprs_path("tgt"),
        expr="rank(close)", sharpe=1.3, fitness=1.0, turnover=0.22,
        alpha_id="tgt_other", status="COMPLETE", region="USA", universe="TOP1000",
        decay=6,
    )

    shared = collect_shared_pheromones([source])
    n = inject_shared_pheromones(target, shared)
    assert n == 0  # already present, no duplicate

    # Add a fresh L1 row that does NOT exist in target — should be injected.
    append_tried(
        tried_exprs_path("src"),
        expr="rank(returns) - rank(ts_mean(volume, 20))",
        sharpe=1.55, fitness=1.18, turnover=0.19,
        alpha_id="src2", status="COMPLETE", region="USA", universe="TOP500",
        decay=6, evidence_type="region_swap", altitude=ALTITUDE_L1_REGION_UNIVERSE,
        parent_alpha_id=None, delta_U=0.42,
    )
    shared2 = collect_shared_pheromones([source])
    n2 = inject_shared_pheromones(target, shared2)
    assert n2 == 1  # only the new row injected

    rows = read_tried(tried_exprs_path("tgt"))
    injected = [r for r in rows if r.get("evidence_type") == "colony_shared"]
    assert len(injected) == 1
    inj = injected[0]
    assert inj["expr"] == "rank(returns) - rank(ts_mean(volume, 20))"
    assert inj["parent_alpha_id"] == "src2"
    assert inj["altitude"] == ALTITUDE_L1_REGION_UNIVERSE


def test_run_colony_calls_runner_per_panel_and_writes_manifest(isolated_artifacts):
    panels = [
        PanelSpec(tag="colony_p1", region="USA", universe="TOP500", max_turns=1),
        PanelSpec(tag="colony_p2", region="USA", universe="TOP1000", max_turns=1),
    ]
    # Pre-seed panel 1 with one L1 + one L4 row to confirm only L1 propagates.
    append_tried(
        tried_exprs_path("colony_p1"),
        expr="rank(returns)", sharpe=1.5, fitness=1.1, turnover=0.20,
        alpha_id="seed1", status="COMPLETE", region="USA", universe="TOP500",
        decay=6, evidence_type="seed", altitude=ALTITUDE_L1_REGION_UNIVERSE,
        parent_alpha_id=None, delta_U=0.0,
    )
    append_tried(
        tried_exprs_path("colony_p1"),
        expr="rank(ts_mean(close, 5))", sharpe=1.42, fitness=1.05, turnover=0.21,
        alpha_id="seed1b", status="COMPLETE", region="USA", universe="TOP500",
        decay=6, evidence_type="numeric_tweak", altitude=ALTITUDE_L4_NUMERIC_TWEAK,
        parent_alpha_id="seed1", delta_U=0.07,
    )

    cfg = ColonyConfig(
        colony_tag="unit_colony_mvp",
        panels=panels,
        cli="opencode",
        model="MiniMax-test",
        timeout_sec=5.0,
    )
    calls: list[str] = []

    def fake_runner(agent_cfg):
        calls.append(agent_cfg.tag)
        return {"run_id": f"fake_{agent_cfg.tag}", "elapsed_sec": 0.01}

    manifest = run_colony(cfg, runner=fake_runner)
    assert calls == ["colony_p1", "colony_p2"]
    # Manifest written:
    written = colony_run_dir("unit_colony_mvp") / "manifest.json"
    assert written.exists()
    data = json.loads(written.read_text())
    assert data["colony_tag"] == "unit_colony_mvp"
    assert len(data["panel_summaries"]) == 2
    # Panel 2 should have received exactly 1 shared injection (the L1 row).
    assert data["panel_summaries"][0]["shared_pheromones_injected"] == 0
    assert data["panel_summaries"][1]["shared_pheromones_injected"] == 1

    # Verify the injected row really landed:
    rows = read_tried(tried_exprs_path("colony_p2"))
    shared_rows = [r for r in rows if r.get("evidence_type") == "colony_shared"]
    assert len(shared_rows) == 1
    assert shared_rows[0]["expr"] == "rank(returns)"
    assert shared_rows[0]["parent_alpha_id"] == "seed1"


def test_run_colony_rejects_empty_panels_or_missing_model(isolated_artifacts):
    panel = PanelSpec(tag="p", region="USA", universe="TOP500")
    with pytest.raises(ValueError, match="no panels"):
        run_colony(ColonyConfig(colony_tag="bad", panels=[], cli="opencode", model="m"))
    with pytest.raises(ValueError, match="model is empty"):
        run_colony(ColonyConfig(colony_tag="bad", panels=[panel], cli="opencode", model=""))


def test_colony_run_cli_invokes_each_panel(isolated_artifacts, monkeypatch):
    """End-to-end CLI smoke: ``wq_brain colony run`` parses panels + spawns ants."""
    import importlib.util
    import io
    import sys
    from contextlib import redirect_stdout
    from unittest.mock import patch

    REPO_ROOT = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_wq_brain_cli_under_test_colony", REPO_ROOT / "scripts" / "wq_brain.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    seen_tags: list[str] = []

    def fake_run_agent(agent_cfg):
        seen_tags.append(agent_cfg.tag)
        return {"run_id": f"stub_{agent_cfg.tag}", "elapsed_sec": 0.0}

    buf = io.StringIO()
    with patch("agent_market.wq_brain.colony.run_agent", new=fake_run_agent), \
            redirect_stdout(buf), pytest.raises(SystemExit) as excinfo:
        parser = module._build_parser()
        args = parser.parse_args([
            "colony", "run",
            "--colony-tag", "cli_smoke",
            "--panels", "USA:TOP500,USA:TOP1000",
            "--model", "MiniMax-test",
            "--max-turns", "1",
            "--timeout-sec", "5",
        ])
        args.func(args)

    assert excinfo.value.code == 0
    out = json.loads(buf.getvalue())
    assert out["ok"] is True
    assert out["manifest"]["colony_tag"] == "cli_smoke"
    assert len(out["manifest"]["panel_summaries"]) == 2
    assert len(seen_tags) == 2
    # Panel tags derive from colony tag + region + universe (lower-cased).
    assert seen_tags[0].endswith("usa_top500")
    assert seen_tags[1].endswith("usa_top1000")
