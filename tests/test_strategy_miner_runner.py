"""Integration tests for strategy_miner runner loop."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_market.strategy_miner.dtypes import MinerConfig, MinerState, Phase, StrategyCandidate
from agent_market.strategy_miner.knowledge_base import KnowledgeBase


# ---------------------------------------------------------------------------
# _should_evolve
# ---------------------------------------------------------------------------


def test_should_evolve_disabled():
    from agent_market.strategy_miner.runner import _should_evolve

    config = MinerConfig(evolve_enabled=False)
    state = MinerState(iteration=2)
    state.best_candidate = StrategyCandidate("S", "code", Path("/tmp/x.py"))
    assert _should_evolve(config, state) is False


def test_should_evolve_no_best():
    from agent_market.strategy_miner.runner import _should_evolve

    config = MinerConfig(evolve_enabled=True)
    state = MinerState(iteration=2)
    assert _should_evolve(config, state) is False


def test_should_evolve_first_iteration():
    from agent_market.strategy_miner.runner import _should_evolve

    config = MinerConfig(evolve_enabled=True, evolve_every_n=2)
    state = MinerState(iteration=0)
    state.best_candidate = StrategyCandidate("S", "code", Path("/tmp/x.py"))
    assert _should_evolve(config, state) is False


def test_should_evolve_every_n():
    from agent_market.strategy_miner.runner import _should_evolve

    config = MinerConfig(evolve_enabled=True, evolve_every_n=2)
    state = MinerState()
    state.best_candidate = StrategyCandidate("S", "code", Path("/tmp/x.py"))

    state.iteration = 1
    assert _should_evolve(config, state) is False
    state.iteration = 2
    assert _should_evolve(config, state) is True
    state.iteration = 3
    assert _should_evolve(config, state) is False
    state.iteration = 4
    assert _should_evolve(config, state) is True


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip():
    from agent_market.strategy_miner.runner import _load_checkpoint, _save_checkpoint

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        state = MinerState()
        state.phase = Phase.EVALUATION
        state.iteration = 3
        state.best_reward = 0.75

        _save_checkpoint(state, run_dir)

        cp = run_dir / "checkpoint.json"
        assert cp.exists()

        loaded = _load_checkpoint(cp)
        assert loaded.run_id == state.run_id
        assert loaded.phase == Phase.EVALUATION
        assert loaded.iteration == 3
        assert loaded.best_reward == 0.75


def test_checkpoint_atomic_write():
    """Verify no .tmp file remains after save."""
    from agent_market.strategy_miner.runner import _save_checkpoint

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        state = MinerState()
        _save_checkpoint(state, run_dir)

        assert not (run_dir / "checkpoint.tmp").exists()
        assert (run_dir / "checkpoint.json").exists()


# ---------------------------------------------------------------------------
# _update_knowledge_base
# ---------------------------------------------------------------------------


def test_update_kb_with_successful_candidate():
    from agent_market.strategy_miner.runner import _update_knowledge_base

    with tempfile.TemporaryDirectory() as td:
        kb = KnowledgeBase(Path(td) / "kb.json")
        state = MinerState()
        c = StrategyCandidate("Good", "code", Path("/tmp/x.py"))
        c.reward = 0.8
        c.backtest_summary = {"profit_total_pct": 10, "trades": 50}
        c.validation_passed = True
        state.candidates.append(c)

        _update_knowledge_base(kb, state)
        assert len(kb.elites) == 1
        assert kb.elites[0]["name"] == "Good"


def test_update_kb_with_failed_candidate():
    from agent_market.strategy_miner.runner import _update_knowledge_base

    with tempfile.TemporaryDirectory() as td:
        kb = KnowledgeBase(Path(td) / "kb.json")
        state = MinerState()
        c = StrategyCandidate("Bad", "code", Path("/tmp/x.py"))
        c.validation_passed = False
        c.diagnosis = "Missing IStrategy"
        state.candidates.append(c)

        _update_knowledge_base(kb, state)
        assert len(kb.failures) == 1
        assert "validation" in kb.failures[0]["failure_type"]


def test_update_kb_no_candidates():
    from agent_market.strategy_miner.runner import _update_knowledge_base

    with tempfile.TemporaryDirectory() as td:
        kb = KnowledgeBase(Path(td) / "kb.json")
        state = MinerState()
        _update_knowledge_base(kb, state)
        assert len(kb.elites) == 0
        assert len(kb.failures) == 0


# ---------------------------------------------------------------------------
# max_retries propagation
# ---------------------------------------------------------------------------


def test_build_strategy_agent_passes_max_retries_to_agent():
    """Agent factory should pass config.max_retries into StrategyAgent."""
    from agent_market.strategy_miner.agent_factory import build_strategy_agent

    with tempfile.TemporaryDirectory() as td:
        with patch("agent_market.strategy_miner.agent_factory.StrategyAgent") as MockAgent:
            mock_instance = MagicMock()
            MockAgent.return_value = mock_instance

            config = MinerConfig(model="test-model", max_retries=7)
            build_strategy_agent(config, Path(td))

            _, kwargs = MockAgent.call_args
            assert kwargs["max_retries"] == 7
