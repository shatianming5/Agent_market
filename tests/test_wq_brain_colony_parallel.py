"""Tests for parallel colony workers + telemetry stream."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from agent_market.wq_brain.colony import (
    ColonyConfig,
    PanelSpec,
    colony_run_dir,
    run_colony,
    telemetry_path,
)


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


def _stub_runner_factory(seen, lock):
    def runner(agent_cfg):
        with lock:
            seen.append(agent_cfg.tag)
        # Simulate a small amount of work so parallel runs interleave.
        time.sleep(0.02)
        return {"run_id": f"stub_{agent_cfg.tag}", "elapsed_sec": 0.02}
    return runner


def test_run_colony_parallel_executes_every_panel(isolated_artifacts):
    panels = [
        PanelSpec(tag=f"parpanel_{i}", region="USA",
                  universe=f"TOP{(i + 1) * 100}", max_turns=1)
        for i in range(4)
    ]
    seen: list[str] = []
    lock = threading.Lock()
    cfg = ColonyConfig(
        colony_tag="parallel_colony",
        panels=panels,
        cli="opencode",
        model="stub-model",
        timeout_sec=5.0,
        workers=4,
    )
    manifest = run_colony(cfg, runner=_stub_runner_factory(seen, lock))
    assert set(seen) == {p.tag for p in panels}
    assert manifest["workers"] == 4
    # panel_summaries are sorted by panel_index regardless of completion order.
    indices = [s["panel_index"] for s in manifest["panel_summaries"]]
    assert indices == sorted(indices)


def test_run_colony_emits_telemetry_jsonl(isolated_artifacts):
    panels = [
        PanelSpec(tag="tele_panel_a", region="USA", universe="TOP500", max_turns=1),
        PanelSpec(tag="tele_panel_b", region="USA", universe="TOP1000", max_turns=1),
    ]
    cfg = ColonyConfig(
        colony_tag="tele_colony",
        panels=panels,
        cli="opencode",
        model="stub-model",
        timeout_sec=5.0,
        workers=2,
    )
    seen: list[str] = []
    lock = threading.Lock()
    run_colony(cfg, runner=_stub_runner_factory(seen, lock))

    tpath = telemetry_path("tele_colony")
    assert tpath.exists()
    events = [json.loads(line) for line in tpath.read_text().splitlines() if line]
    event_names = [e["event"] for e in events]
    assert event_names.count("colony_start") == 1
    assert event_names.count("colony_end") == 1
    # Each panel must emit start + end → at least 2 per panel.
    panel_starts = [e for e in events if e["event"] == "panel_start"]
    panel_ends = [e for e in events if e["event"] == "panel_end"]
    assert {e["panel_tag"] for e in panel_starts} == {"tele_panel_a", "tele_panel_b"}
    assert {e["panel_tag"] for e in panel_ends} == {"tele_panel_a", "tele_panel_b"}
    # Routing decision attached to panel_start.
    for e in panel_starts:
        assert e["routing_action"] in {
            "stay", "deeper", "bubble_up", "jump_root",
        }
        assert e["routing_altitude"].startswith("L")


def test_run_colony_workers_one_keeps_sequential_order(isolated_artifacts):
    panels = [
        PanelSpec(tag=f"seq_{i}", region="USA",
                  universe=f"TOP{(i + 1) * 100}", max_turns=1)
        for i in range(3)
    ]
    seen: list[str] = []
    lock = threading.Lock()
    cfg = ColonyConfig(
        colony_tag="seq_colony",
        panels=panels,
        cli="opencode",
        model="stub-model",
        timeout_sec=5.0,
        workers=1,
    )
    run_colony(cfg, runner=_stub_runner_factory(seen, lock))
    # workers=1 must preserve insertion order.
    assert seen == [p.tag for p in panels]
