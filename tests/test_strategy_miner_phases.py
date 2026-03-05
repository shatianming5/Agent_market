"""Integration tests for strategy_miner phases."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_market.strategy_miner.dtypes import MinerConfig, MinerState, Phase, StrategyCandidate


_VALID_STRATEGY = """
from freqtrade.strategy import IStrategy

class TestStrategy(IStrategy):
    timeframe = "5m"
    stoploss = -0.10
    minimal_roi = {"0": 0.05}

    def populate_indicators(self, dataframe, metadata):
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe.loc[dataframe['close'] > 0, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe.loc[dataframe['close'] < 0, 'exit_long'] = 1
        return dataframe
"""


# ---------------------------------------------------------------------------
# phase_strategy_gen
# ---------------------------------------------------------------------------


def test_phase_strategy_gen_no_files_produced():
    """If agent produces no strategy files, phase transitions to ANALYSIS."""
    from agent_market.strategy_miner.phases import phase_strategy_gen

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        config = MinerConfig()
        state = MinerState()
        state.phase = Phase.STRATEGY_GEN

        mock_agent = MagicMock()
        mock_agent.run.return_value = "I couldn't generate a strategy."

        phase_strategy_gen(state, config, run_dir, mock_agent)

        assert state.phase == Phase.ANALYSIS
        assert len(state.candidates) == 0
        mock_agent.run.assert_called_once()


def test_phase_strategy_gen_writes_file():
    """If agent writes a file, candidate is created and phase moves to BACKTEST."""
    from agent_market.strategy_miner.phases import phase_strategy_gen
    from agent_market.strategy_miner.sandbox import prepare_sandbox

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        config = MinerConfig()
        state = MinerState()
        state.phase = Phase.STRATEGY_GEN

        # Pre-create sandbox and write a strategy file (simulating agent behavior)
        sandbox = prepare_sandbox(config, run_dir, 0)
        strat_dir = sandbox / "user_data" / "strategies"
        strat_dir.mkdir(parents=True, exist_ok=True)
        strat_file = strat_dir / "TestStrategy.py"
        strat_file.write_text(_VALID_STRATEGY, encoding="utf-8")

        mock_agent = MagicMock()
        mock_agent.run.return_value = "Done"

        phase_strategy_gen(state, config, run_dir, mock_agent)

        assert state.phase == Phase.BACKTEST
        assert len(state.candidates) == 1
        assert state.candidates[0].name == "TestStrategy"


# ---------------------------------------------------------------------------
# phase_backtest
# ---------------------------------------------------------------------------


def test_phase_backtest_validation_failure():
    """Invalid strategy code should fail validation and skip to ANALYSIS."""
    from agent_market.strategy_miner.phases import phase_backtest

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        config = MinerConfig()
        state = MinerState()

        # Create candidate with invalid code (imports os)
        bad_code = """
import os
from freqtrade.strategy import IStrategy
class Bad(IStrategy):
    def populate_indicators(self, df, m): return df
    def populate_entry_trend(self, df, m): return df
    def populate_exit_trend(self, df, m): return df
"""
        sandbox = run_dir / "iter_0" / "sandbox"
        strat_dir = sandbox / "user_data" / "strategies"
        strat_dir.mkdir(parents=True, exist_ok=True)
        strat_path = strat_dir / "Bad.py"
        strat_path.write_text(bad_code)

        candidate = StrategyCandidate(
            name="Bad", code=bad_code, strategy_path=strat_path
        )
        state.candidates.append(candidate)
        state.phase = Phase.BACKTEST

        phase_backtest(state, config, run_dir)

        assert state.phase == Phase.ANALYSIS
        assert not candidate.validation_passed
        assert "Forbidden import" in candidate.diagnosis


def test_phase_backtest_no_candidates():
    """No candidates should transition to ANALYSIS."""
    from agent_market.strategy_miner.phases import phase_backtest

    with tempfile.TemporaryDirectory() as td:
        state = MinerState()
        state.phase = Phase.BACKTEST
        phase_backtest(state, MinerConfig(), Path(td))
        assert state.phase == Phase.ANALYSIS


# ---------------------------------------------------------------------------
# phase_evaluation
# ---------------------------------------------------------------------------


def test_phase_evaluation_computes_reward():
    """Evaluation computes reward and updates best candidate."""
    from agent_market.strategy_miner.phases import phase_evaluation

    state = MinerState()
    state.phase = Phase.EVALUATION
    candidate = StrategyCandidate(
        name="Good", code=_VALID_STRATEGY, strategy_path=Path("/tmp/x.py")
    )
    candidate.backtest_summary = {
        "profit_total_pct": 15.0, "trades": 60,
        "winrate": 0.65, "max_drawdown_abs": -3.0, "avg_profit_pct": 0.8,
    }
    state.candidates.append(candidate)

    config = MinerConfig()
    phase_evaluation(state, config)

    assert state.phase == Phase.ANALYSIS
    assert candidate.reward is not None
    assert -1.0 <= candidate.reward <= 1.0
    assert state.best_candidate is not None
    assert state.best_candidate.name == "Good"
    assert len(state.history) == 1


def test_phase_evaluation_no_summary():
    """Missing backtest summary transitions to ANALYSIS."""
    from agent_market.strategy_miner.phases import phase_evaluation

    state = MinerState()
    candidate = StrategyCandidate("X", "code", Path("/tmp/x.py"))
    candidate.backtest_summary = None
    state.candidates.append(candidate)

    phase_evaluation(state, MinerConfig())
    assert state.phase == Phase.ANALYSIS


# ---------------------------------------------------------------------------
# phase_analysis
# ---------------------------------------------------------------------------


def test_phase_analysis_with_agent():
    """Analysis with agent should produce diagnosis."""
    from agent_market.strategy_miner.phases import phase_analysis

    with tempfile.TemporaryDirectory() as td:
        state = MinerState()
        candidate = StrategyCandidate("Strat", _VALID_STRATEGY, Path("/tmp/x.py"))
        candidate.backtest_summary = {"profit_total_pct": 5.0, "trades": 20}
        candidate.reward = 0.3
        state.candidates.append(candidate)
        state.history.append({"components": {"sharpe": 0.2}})

        mock_agent = MagicMock()
        mock_agent.run.return_value = "Strategy needs better drawdown control."

        config = MinerConfig(max_iterations=5)
        phase_analysis(state, config, Path(td), mock_agent)

        assert "drawdown" in candidate.diagnosis.lower()
        assert state.phase == Phase.STRATEGY_GEN
        assert state.iteration == 1


def test_phase_analysis_last_iteration_completes():
    """Analysis at last iteration should set phase to COMPLETE."""
    from agent_market.strategy_miner.phases import phase_analysis

    with tempfile.TemporaryDirectory() as td:
        state = MinerState()
        state.iteration = 4
        candidate = StrategyCandidate("S", "c", Path("/tmp/x.py"))
        candidate.diagnosis = "test"
        state.candidates.append(candidate)

        config = MinerConfig(max_iterations=5)
        phase_analysis(state, config, Path(td))

        assert state.phase == Phase.COMPLETE


def test_phase_analysis_no_candidates_completes():
    """No candidates should complete."""
    from agent_market.strategy_miner.phases import phase_analysis

    with tempfile.TemporaryDirectory() as td:
        state = MinerState()
        phase_analysis(state, MinerConfig(), Path(td))
        assert state.phase == Phase.COMPLETE


# ---------------------------------------------------------------------------
# phase_evolve
# ---------------------------------------------------------------------------


def test_phase_evolve_no_best_candidate():
    """No best candidate returns None."""
    from agent_market.strategy_miner.phases import phase_evolve

    with tempfile.TemporaryDirectory() as td:
        state = MinerState()
        result = phase_evolve(state, MinerConfig(), Path(td))
        assert result is None


def test_phase_evolve_with_valid_code():
    """Evolving valid code should produce a candidate."""
    from agent_market.strategy_miner.phases import phase_evolve

    with tempfile.TemporaryDirectory() as td:
        state = MinerState()
        state.best_candidate = StrategyCandidate(
            "BestStrat", _VALID_STRATEGY, Path("/tmp/x.py")
        )

        result = phase_evolve(state, MinerConfig(), Path(td))
        # May or may not produce a change depending on random mutations
        # But should not raise
        assert result is None or isinstance(result, StrategyCandidate)
