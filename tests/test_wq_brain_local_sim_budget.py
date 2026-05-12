"""Tests for compact-agent local-simulate budgeting."""
from __future__ import annotations

import time

import pytest

from scripts import wq_brain


def test_agent_local_sim_slot_enforces_concurrency_and_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("WQB_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("WQB_AGENT_LOCAL_SIM_LIMIT", "1")
    monkeypatch.setenv("WQB_AGENT_LOCAL_SIM_MAX_CONCURRENT", "1")

    with wq_brain._agent_local_sim_slot("rank(close)"):
        with pytest.raises(RuntimeError, match="concurrency limit"):
            with wq_brain._agent_local_sim_slot("rank(open)"):
                pass

    with pytest.raises(RuntimeError, match="budget exhausted"):
        with wq_brain._agent_local_sim_slot("rank(open)"):
            pass

    with wq_brain._agent_local_sim_slot("rank(close)"):
        pass


def test_agent_local_sim_gate_requires_passed_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("WQB_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("WQB_AGENT_REQUIRE_LOCAL_SIM", "1")

    missing = wq_brain._agent_local_sim_gate_error("rank(close)")
    assert "gate missing" in missing

    monkeypatch.setenv("WQB_AGENT_LOCAL_SIM_LIMIT", "1")
    monkeypatch.setenv("WQB_AGENT_LOCAL_SIM_MAX_CONCURRENT", "1")
    with wq_brain._agent_local_sim_slot("rank(close)"):
        pass
    running = wq_brain._agent_local_sim_gate_error("rank(close)")
    assert "not passed" in running

    with wq_brain._agent_local_sim_slot("rank(close)"):
        wq_brain._mark_agent_local_sim_status(
            "rank(close)", "passed", extra={"passes_local_gate": True}
        )
    assert wq_brain._agent_local_sim_gate_error("rank(close)") == ""


def test_agent_local_sim_time_limit_raises(monkeypatch):
    monkeypatch.setenv("WQB_AGENT_LOCAL_SIM_TIMEOUT_SEC", "0.01")
    with pytest.raises(TimeoutError, match="timed out"):
        with wq_brain._agent_local_sim_time_limit():
            time.sleep(1)
