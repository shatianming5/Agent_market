"""Tests for compact-agent local-simulate budgeting."""
from __future__ import annotations

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
