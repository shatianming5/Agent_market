"""Tests for cmd_simulate auto-writing pheromone metadata on tried_log."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "wq_brain.py"


@pytest.fixture(scope="module")
def cli_module():
    spec = importlib.util.spec_from_file_location(
        "_wq_brain_cli_for_simulate_pheromone", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


@dataclass
class _StubResult:
    sharpe: float
    fitness: float
    turnover: float
    alpha_id: str
    status: str = "COMPLETE"
    error: Optional[str] = None
    returns: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sharpe": self.sharpe,
            "fitness": self.fitness,
            "turnover": self.turnover,
            "alpha_id": self.alpha_id,
            "status": self.status,
            "returns": self.returns,
        }


class _StubSession:
    def __init__(self, result: _StubResult) -> None:
        self._result = result

    def login(self) -> None:  # not used here
        return None

    def simulate_and_parse(self, expr, settings, timeout):
        return self._result


def _run_simulate(cli_module, argv: list[str], result: _StubResult):
    parser = cli_module._build_parser()
    args = parser.parse_args(argv)
    fake_sess = _StubSession(result)
    fake_reserve = {"status": "ok", "day": "2026-05-15", "counts": {"simulate": 1}}
    buf = io.StringIO()
    with patch(
            "agent_market.wq_brain.client.session_from_env",
            return_value=fake_sess,
        ), patch(
            "agent_market.wq_brain.quota_monitor.reserve_action",
            return_value=fake_reserve,
        ), patch(
            "agent_market.wq_brain.quota_monitor.release_action",
            return_value=None,
        ), redirect_stdout(buf), pytest.raises(SystemExit) as excinfo:
        args.func(args)
    return excinfo.value.code, buf.getvalue()


def test_cmd_simulate_writes_manual_evidence_when_no_parent(
    cli_module, isolated_artifacts
):
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.tried_log import read_tried

    code, out = _run_simulate(
        cli_module,
        ["simulate", "rank(close)", "--tag", "ut_simulate_no_parent",
         "--auto-persist-sharpe", "999"],
        _StubResult(sharpe=1.45, fitness=1.10, turnover=0.20, alpha_id="child1"),
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["ok"] is True
    rows = read_tried(tried_exprs_path("ut_simulate_no_parent"))
    assert len(rows) == 1
    row = rows[0]
    assert row["evidence_type"] == "manual"
    # No parent → no delta_U. Altitude defaults to L1_region_universe so the
    # row is still eligible for cross-panel pheromone propagation.
    assert row.get("parent_alpha_id") is None
    assert row.get("altitude") == "L1_region_universe"
    assert row.get("delta_U") is None


def test_cmd_simulate_classifies_altitude_and_delta_U_against_parent(
    cli_module, isolated_artifacts
):
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.tried_log import (
        ALTITUDE_L2_OP_FAMILY,
        ALTITUDE_L4_NUMERIC_TWEAK,
        append_tried,
        read_tried,
    )

    # Seed parent row.
    append_tried(
        tried_exprs_path("ut_simulate_parent"),
        expr="rank(ts_mean(close, 20))",
        sharpe=1.30, fitness=1.00, turnover=0.25,
        alpha_id="parent_a", status="COMPLETE",
        region="USA", universe="TOP500", decay=6,
        evidence_type="seed", altitude="L1_region_universe",
        parent_alpha_id=None, delta_U=0.0,
    )

    # Child A: same ops, only numeric tweak (window 20 → 30) → L4
    # --skip-cooldown bypasses the slot gate; this test exercises the
    # altitude-classifier wiring, not the cool-down hard gate (covered in
    # test_wq_brain_gates.py).
    code_a, _ = _run_simulate(
        cli_module,
        [
            "simulate", "rank(ts_mean(close, 30))",
            "--tag", "ut_simulate_parent",
            "--parent-alpha-id", "parent_a",
            "--region", "USA", "--universe", "TOP500", "--decay", "6",
            "--auto-persist-sharpe", "999",
            "--skip-cooldown",
        ],
        _StubResult(sharpe=1.45, fitness=1.12, turnover=0.20, alpha_id="child_a"),
    )
    assert code_a == 0

    # Child B: operator-family change (rank(ts_mean…) → rank(signed_power…)) → L2
    code_b, _ = _run_simulate(
        cli_module,
        [
            "simulate", "rank(signed_power(close, 0.3))",
            "--tag", "ut_simulate_parent",
            "--parent-alpha-id", "parent_a",
            "--region", "USA", "--universe", "TOP500", "--decay", "6",
            "--evidence-type", "op_swap",
            "--auto-persist-sharpe", "999",
            "--skip-cooldown",
        ],
        _StubResult(sharpe=1.20, fitness=0.95, turnover=0.18, alpha_id="child_b"),
    )
    assert code_b == 0

    rows = read_tried(tried_exprs_path("ut_simulate_parent"))
    # parent + child_a + child_b
    by_id = {r.get("alpha_id"): r for r in rows}
    assert by_id["parent_a"]["evidence_type"] == "seed"
    a = by_id["child_a"]
    b = by_id["child_b"]

    assert a["parent_alpha_id"] == "parent_a"
    assert a["altitude"] == ALTITUDE_L4_NUMERIC_TWEAK
    assert a["evidence_type"] == "mutation"  # default when parent given
    # ΔU > 0 since sharpe/fitness both went up
    assert isinstance(a["delta_U"], float)
    assert a["delta_U"] > 0

    assert b["parent_alpha_id"] == "parent_a"
    assert b["altitude"] == ALTITUDE_L2_OP_FAMILY
    assert b["evidence_type"] == "op_swap"  # override honoured
    # ΔU < 0 since sharpe dropped
    assert isinstance(b["delta_U"], float)
    assert b["delta_U"] < 0


def test_cmd_simulate_handles_missing_parent_gracefully(cli_module, isolated_artifacts):
    """parent-alpha-id pointing to a non-existent row → row still gets written
    with a sentinel altitude, and delta_U stays None (since no parent metrics)."""
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.tried_log import read_tried

    code, _ = _run_simulate(
        cli_module,
        [
            "simulate", "rank(close)",
            "--tag", "ut_simulate_missing_parent",
            "--parent-alpha-id", "no_such_alpha",
            "--region", "USA", "--universe", "TOP500",
            "--auto-persist-sharpe", "999",
        ],
        _StubResult(sharpe=1.30, fitness=1.05, turnover=0.22, alpha_id="orphan"),
    )
    assert code == 0
    rows = read_tried(tried_exprs_path("ut_simulate_missing_parent"))
    row = rows[-1]
    assert row.get("parent_alpha_id") == "no_such_alpha"
    assert row.get("altitude") == "L1_region_universe"
    assert row.get("delta_U") is None
    assert row["evidence_type"] == "mutation"
