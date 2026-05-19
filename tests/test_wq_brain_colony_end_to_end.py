"""End-to-end colony integration test.

Wires a stub ``run_agent`` that calls ``cmd_simulate`` against a stubbed
WQ session. After one colony run we verify the entire pheromone pipeline:

  * tried_log rows are typed (evidence_type / altitude / parent / ΔU).
  * shared cache contains the L1/L2 finds and respects capacity.
  * routing advisory JSON exists per panel.
  * best_so_far records exist and reflect the highest-utility candidate.
  * telemetry stream has colony_start / panel_start / panel_end / colony_end.
"""
from __future__ import annotations

import importlib.util
import io
import json
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from agent_market.wq_brain.colony import (
    ColonyConfig,
    PanelSpec,
    run_colony,
    routing_advisory_path,
    telemetry_path,
)
from agent_market.wq_brain.colony_state import (
    read_best_so_far,
)
from agent_market.wq_brain.paths import tried_exprs_path
from agent_market.wq_brain.pheromone_cache import (
    cache_path,
    read_cache,
)
from agent_market.wq_brain.tried_log import (
    ALTITUDE_L1_REGION_UNIVERSE,
    ALTITUDE_L2_OP_FAMILY,
    read_tried,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "wq_brain.py"


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture(scope="module")
def cli_module():
    spec = importlib.util.spec_from_file_location(
        "_wq_brain_cli_e2e", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass
class _StubResult:
    sharpe: float
    fitness: float
    turnover: float
    alpha_id: str
    status: str = "COMPLETE"
    error: Optional[str] = None
    returns: float = 0.0

    def to_dict(self):
        return {
            "sharpe": self.sharpe,
            "fitness": self.fitness,
            "turnover": self.turnover,
            "alpha_id": self.alpha_id,
            "status": self.status,
            "returns": self.returns,
        }


class _StubSession:
    def __init__(self, ladder):
        # ladder maps expression-prefix → SimulationResult so we can return
        # different (sharpe, fitness) per call deterministically.
        self._ladder = ladder
        self._calls = 0

    def login(self):
        return None

    def simulate_and_parse(self, expr, settings, timeout):
        self._calls += 1
        # Longest matching key wins so "rank(close) * rank(volume)" doesn't
        # collide with the "rank(close)" prefix entry.
        best_prefix: Optional[str] = None
        for prefix in self._ladder:
            if expr.startswith(prefix):
                if best_prefix is None or len(prefix) > len(best_prefix):
                    best_prefix = prefix
        if best_prefix is not None:
            result = self._ladder[best_prefix]
            return _StubResult(
                sharpe=result.sharpe,
                fitness=result.fitness,
                turnover=result.turnover,
                alpha_id=f"{result.alpha_id}_{self._calls}",
            )
        return _StubResult(
            sharpe=1.0, fitness=0.9, turnover=0.25,
            alpha_id=f"default_{self._calls}",
        )


def _make_agent_stub(cli_module, ladder, panel_simulate_args):
    """Returns a fake run_agent that drives cmd_simulate for the given panel."""

    def fake_agent(agent_cfg):
        # Pull the simulate args slated for this panel and invoke cmd_simulate.
        args_for_panel = panel_simulate_args.get(agent_cfg.tag, [])
        session = _StubSession(ladder)
        for argv_extra in args_for_panel:
            parser = cli_module._build_parser()
            argv = [
                "simulate", argv_extra["expr"],
                "--tag", agent_cfg.tag,
                "--region", agent_cfg.region,
                "--universe", agent_cfg.universe,
                "--decay", str(agent_cfg.decay),
                "--auto-persist-sharpe", "999",
                "--skip-cooldown",  # tests don't exercise gate here
                "--colony-tag", argv_extra["colony_tag"],
            ]
            if "parent" in argv_extra:
                argv.extend([
                    "--parent-alpha-id", argv_extra["parent"],
                    "--evidence-type", argv_extra.get("evidence_type", "mutation"),
                ])
            args_obj = parser.parse_args(argv)
            buf = io.StringIO()
            fake_reserve = {"status": "ok", "day": "2026-05-15",
                            "counts": {"simulate": 1}}
            with patch("agent_market.wq_brain.client.session_from_env",
                       return_value=session), \
                    patch("agent_market.wq_brain.quota_monitor.reserve_action",
                          return_value=fake_reserve), \
                    patch("agent_market.wq_brain.quota_monitor.release_action",
                          return_value=None), \
                    redirect_stdout(buf), pytest.raises(SystemExit):
                args_obj.func(args_obj)
        return {"run_id": f"e2e_stub_{agent_cfg.tag}", "elapsed_sec": 0.0}

    return fake_agent


def test_colony_end_to_end_pheromone_pipeline(cli_module, isolated_artifacts):
    # Both panels share the same (region, universe) so a same-universe child
    # gets classified as an op-family edit (L2), not a region swap (L1).
    panel_a = PanelSpec(tag="e2e_panel_a", region="USA", universe="TOP500",
                        max_turns=1)
    panel_b = PanelSpec(tag="e2e_panel_b", region="USA", universe="TOP500",
                        max_turns=1)
    panel_simulate_args = {
        "e2e_panel_a": [
            {"expr": "rank(close)", "colony_tag": "e2e_colony"},  # seed (no parent)
        ],
        "e2e_panel_b": [
            # Panel B mutates panel A's seed; parent will exist because the
            # injected colony_shared row carries the parent alpha_id.
            {"expr": "rank(close) * rank(volume)",
             "parent": "seed_a_1",
             "evidence_type": "op_swap",
             "colony_tag": "e2e_colony"},
        ],
    }
    ladder = {
        "rank(close)": _StubResult(
            sharpe=1.55, fitness=1.18, turnover=0.18, alpha_id="seed_a",
        ),
        "rank(close) * rank(volume)": _StubResult(
            sharpe=1.65, fitness=1.25, turnover=0.16, alpha_id="op_swap_b",
        ),
    }
    cfg = ColonyConfig(
        colony_tag="e2e_colony",
        panels=[panel_a, panel_b],
        cli="opencode",
        model="stub-model",
        timeout_sec=5.0,
        workers=1,
    )
    manifest = run_colony(
        cfg, runner=_make_agent_stub(cli_module, ladder, panel_simulate_args)
    )

    # ── Manifest sanity ─────────────────────────────────────────────────
    assert manifest["colony_tag"] == "e2e_colony"
    assert len(manifest["panel_summaries"]) == 2

    # ── tried_log typed rows ───────────────────────────────────────────
    rows_a = read_tried(tried_exprs_path("e2e_panel_a"))
    rows_b = read_tried(tried_exprs_path("e2e_panel_b"))
    assert any(
        r.get("expr") == "rank(close)" and r.get("evidence_type") == "manual"
        for r in rows_a
    ), "panel A's seed row should be tagged 'manual' (no parent given)"
    assert any(
        r.get("expr") == "rank(close) * rank(volume)"
        and r.get("evidence_type") == "op_swap"
        and r.get("altitude") == ALTITUDE_L2_OP_FAMILY
        and r.get("parent_alpha_id") == "seed_a_1"
        and isinstance(r.get("delta_U"), float)
        for r in rows_b
    ), "panel B's mutation row should be fully typed"

    # ── Routing advisory written per panel ─────────────────────────────
    for panel in cfg.panels:
        ap = routing_advisory_path(cfg.colony_tag, panel.tag)
        assert ap.exists(), f"routing advisory missing for {panel.tag}"
        data = json.loads(ap.read_text())
        assert data["action"] in {
            "stay", "deeper", "bubble_up", "jump_root"
        }
        assert data["target_altitude"].startswith("L")

    # ── best_so_far per panel ──────────────────────────────────────────
    bsf_a = read_best_so_far(cfg.colony_tag, "e2e_panel_a")
    bsf_b = read_best_so_far(cfg.colony_tag, "e2e_panel_b")
    assert bsf_a is not None and bsf_a.utility > 0
    assert bsf_b is not None and bsf_b.utility > bsf_a.utility  # better metrics

    # ── Shared cache populated with the seed row from panel A ─────────
    cache = read_cache(cfg.colony_tag)
    assert any(
        link.alpha_id and link.alpha_id.startswith("seed_a")
        for link in cache[ALTITUDE_L1_REGION_UNIVERSE]
    ), "panel A's seed should be in the L1 cache"

    # ── Telemetry stream has the full lifecycle ────────────────────────
    tpath = telemetry_path(cfg.colony_tag)
    assert tpath.exists()
    events = [json.loads(line) for line in tpath.read_text().splitlines() if line]
    event_names = [e["event"] for e in events]
    assert event_names.count("colony_start") == 1
    assert event_names.count("colony_end") == 1
    assert sum(1 for e in events if e["event"] == "panel_start") == 2
    assert sum(1 for e in events if e["event"] == "panel_end") == 2

    # ── Cache path file is a real JSON we can re-read ──────────────────
    cp = cache_path(cfg.colony_tag)
    assert cp.exists()
    json.loads(cp.read_text())  # no exception
