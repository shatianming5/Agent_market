"""Tests for best-so-far tracker, uncertainty tags, and colony CLI surface."""
from __future__ import annotations

import importlib.util
import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_market.wq_brain.colony_state import (
    BestSoFar,
    best_so_far_path,
    list_panel_bests,
    read_best_so_far,
    update_best_so_far,
)
from agent_market.wq_brain.pheromone_cache import (
    PheromoneLink,
    UNCERTAINTY_DISAGREEMENT,
    UNCERTAINTY_EXECUTED,
    UNCERTAINTY_LOW_SUPPORT,
    TTL_SECONDS,
    classify_uncertainty,
    write_links,
)
from agent_market.wq_brain.tried_log import (
    ALTITUDE_L1_REGION_UNIVERSE,
    ALTITUDE_L2_OP_FAMILY,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "wq_brain.py"


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


# ── best-so-far ────────────────────────────────────────────────────────────


def test_update_best_so_far_creates_record_when_missing(isolated_artifacts):
    record, updated = update_best_so_far(
        "bsf_colony", "bsf_panel",
        alpha_id="a1", expr="rank(close)",
        sharpe=1.5, fitness=1.1, turnover=0.20,
        delta_U=0.40,
    )
    assert updated is True
    assert record.utility > 0.0
    assert best_so_far_path("bsf_colony", "bsf_panel").exists()


def test_update_best_so_far_keeps_higher_utility(isolated_artifacts):
    update_best_so_far(
        "bsf_keep", "p1",
        alpha_id="lo", expr="rank(close)",
        sharpe=1.3, fitness=1.0, turnover=0.25,
        delta_U=0.20,
    )
    # New candidate with lower utility shouldn't dethrone.
    record, updated = update_best_so_far(
        "bsf_keep", "p1",
        alpha_id="even_lower", expr="rank(volume)",
        sharpe=1.0, fitness=0.85, turnover=0.30,
        delta_U=0.05,
    )
    assert updated is False
    assert record.alpha_id == "lo"


def test_update_best_so_far_replaces_on_strictly_higher_utility(isolated_artifacts):
    update_best_so_far(
        "bsf_replace", "p1",
        alpha_id="prior", expr="rank(close)",
        sharpe=1.3, fitness=1.0, turnover=0.25,
        delta_U=0.20,
    )
    record, updated = update_best_so_far(
        "bsf_replace", "p1",
        alpha_id="newhigh", expr="rank(close) * rank(volume)",
        sharpe=1.6, fitness=1.2, turnover=0.18,
        delta_U=0.45,
    )
    assert updated is True
    assert record.alpha_id == "newhigh"
    on_disk = read_best_so_far("bsf_replace", "p1")
    assert on_disk is not None and on_disk.alpha_id == "newhigh"


def test_list_panel_bests_returns_every_panel(isolated_artifacts):
    update_best_so_far(
        "many", "panel_a",
        alpha_id="a", expr="rank(close)",
        sharpe=1.5, fitness=1.1, turnover=0.20, delta_U=0.3,
    )
    update_best_so_far(
        "many", "panel_b",
        alpha_id="b", expr="rank(volume)",
        sharpe=1.4, fitness=1.05, turnover=0.21, delta_U=0.25,
    )
    bests = list_panel_bests("many")
    panel_tags = {r.panel_tag for r in bests}
    assert panel_tags == {"panel_a", "panel_b"}


# ── uncertainty tags ───────────────────────────────────────────────────────


def _make(*, support=0, conflicts=0, age_s=0, alt=ALTITUDE_L1_REGION_UNIVERSE):
    now = time.time()
    return PheromoneLink(
        altitude=alt,
        expr="rank(x)",
        alpha_id="u_test",
        region="USA",
        universe="TOP500",
        sharpe=1.5, fitness=1.1, turnover=0.2,
        delta_U=0.3, evidence_type="op_swap",
        parent_alpha_id=None, source_panel_tag="src",
        ts=now - age_s, support=support, conflicts=conflicts,
    )


def test_classify_uncertainty_disagreement_when_conflicts_positive():
    assert classify_uncertainty(_make(conflicts=1)) == UNCERTAINTY_DISAGREEMENT


def test_classify_uncertainty_low_support_when_old_and_unsupported():
    ttl = TTL_SECONDS[ALTITUDE_L1_REGION_UNIVERSE]
    aged = _make(age_s=ttl * 0.75)
    assert classify_uncertainty(aged) == UNCERTAINTY_LOW_SUPPORT


def test_classify_uncertainty_executed_when_fresh_or_supported():
    assert classify_uncertainty(_make(support=2)) == UNCERTAINTY_EXECUTED
    assert classify_uncertainty(_make(age_s=60)) == UNCERTAINTY_EXECUTED


# ── colony CLI ─────────────────────────────────────────────────────────────


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "_wq_brain_cli_state", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_cli(module, argv):
    parser = module._build_parser()
    args = parser.parse_args(argv)
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit) as excinfo:
        args.func(args)
    return excinfo.value.code, buf.getvalue()


def test_colony_status_includes_panel_bests_and_cache_summary(isolated_artifacts):
    update_best_so_far(
        "cli_status", "p1",
        alpha_id="best1", expr="rank(close)",
        sharpe=1.6, fitness=1.20, turnover=0.20, delta_U=0.55,
    )
    write_links("cli_status", [_make()])
    code, out = _run_cli(_load_cli(), ["colony", "status", "--colony-tag", "cli_status"])
    assert code == 0
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["colony_tag"] == "cli_status"
    assert payload["panel_bests"][0]["alpha_id"] == "best1"
    cs = payload["cache_summary"][ALTITUDE_L1_REGION_UNIVERSE]
    assert cs["size"] == 1


def test_colony_pheromones_list_orders_by_score_desc(isolated_artifacts):
    high = PheromoneLink(
        altitude=ALTITUDE_L2_OP_FAMILY, expr="rank(volume)",
        alpha_id="hi", region="USA", universe="TOP500",
        sharpe=1.6, fitness=1.2, turnover=0.18, delta_U=0.55,
        evidence_type="op_swap", parent_alpha_id=None,
        source_panel_tag="src", ts=time.time(),
        support=1, conflicts=0,
    )
    low = PheromoneLink(
        altitude=ALTITUDE_L2_OP_FAMILY, expr="rank(close)",
        alpha_id="lo", region="USA", universe="TOP500",
        sharpe=1.3, fitness=1.0, turnover=0.25, delta_U=0.10,
        evidence_type="op_swap", parent_alpha_id=None,
        source_panel_tag="src", ts=time.time(),
        support=0, conflicts=0,
    )
    write_links("cli_pheromones", [low, high])
    code, out = _run_cli(_load_cli(),
                         ["colony", "pheromones", "list",
                          "--colony-tag", "cli_pheromones"])
    assert code == 0
    payload = json.loads(out)
    aliases = [link["alpha_id"] for link in payload["links"]]
    assert aliases.index("hi") < aliases.index("lo")


def test_colony_reset_wipes_cache_and_advisory_files(isolated_artifacts):
    from agent_market.wq_brain.colony import (
        PanelSpec,
        compute_panel_routing_advisory,
        routing_advisory_path,
        write_panel_routing_advisory,
    )
    from agent_market.wq_brain.colony_state import best_so_far_path
    from agent_market.wq_brain.pheromone_cache import cache_path

    write_links("cli_reset", [_make()])
    update_best_so_far(
        "cli_reset", "p1",
        alpha_id="x", expr="rank(close)",
        sharpe=1.5, fitness=1.1, turnover=0.20, delta_U=0.40,
    )
    panel = PanelSpec(tag="p1", region="USA", universe="TOP500")
    decision = compute_panel_routing_advisory("cli_reset", panel)
    write_panel_routing_advisory("cli_reset", "p1", decision)

    assert cache_path("cli_reset").exists()
    assert best_so_far_path("cli_reset", "p1").exists()
    assert routing_advisory_path("cli_reset", "p1").exists()

    code, out = _run_cli(_load_cli(),
                         ["colony", "reset", "--colony-tag", "cli_reset"])
    assert code == 0
    payload = json.loads(out)
    assert payload["ok"] is True
    assert not cache_path("cli_reset").exists()
    assert not best_so_far_path("cli_reset", "p1").exists()
    assert not routing_advisory_path("cli_reset", "p1").exists()
