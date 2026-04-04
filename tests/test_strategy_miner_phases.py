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
        mock_agent.generate_strategy.return_value = None

        phase_strategy_gen(state, config, run_dir, mock_agent)

        assert state.phase == Phase.ANALYSIS
        assert len(state.candidates) == 0
        mock_agent.generate_strategy.assert_called_once()


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
        sandbox = prepare_sandbox(config, run_dir, 0, variant="cand_00/coder")
        strat_dir = sandbox / "user_data" / "strategies"
        strat_dir.mkdir(parents=True, exist_ok=True)
        strat_file = strat_dir / "TestStrategy.py"
        strat_file.write_text(_VALID_STRATEGY, encoding="utf-8")

        mock_agent = MagicMock()
        mock_agent.generate_strategy.return_value = None

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


def test_phase_evaluation_prefers_realistic_sharpe():
    """Evaluation should prefer audited realistic Sharpe over inflated native Sharpe."""
    from agent_market.strategy_miner.phases import phase_evaluation

    state = MinerState()
    state.phase = Phase.EVALUATION
    candidate = StrategyCandidate(
        name="Good", code=_VALID_STRATEGY, strategy_path=Path("/tmp/x.py")
    )
    candidate.backtest_summary = {
        "profit_total_pct": 15.0, "trades": 60,
        "winrate": 0.65, "max_drawdown_abs": -3.0, "avg_profit_pct": 0.8,
        "realistic_sharpe": 1.5,
        "realistic_sortino": 2.0,
        "realistic_calmar": 0.8,
        "sharpe": 99.0,
        "native_sharpe": 99.0,
        "sortino": 88.0,
        "native_sortino": 88.0,
        "calmar": 77.0,
        "native_calmar": 77.0,
        "profit_factor": 1.8, "sqn": 2.1,
        "max_drawdown_pct": 3.0,
        "positive_days_ratio": 0.7,
        "return_over_drawdown": 5.0,
        "observation_days": 29,
        "metric_flags": ["native_sharpe_inflated"],
    }
    state.candidates.append(candidate)

    config = MinerConfig()
    phase_evaluation(state, config)

    assert state.phase == Phase.ANALYSIS
    assert candidate.reward < 99.0
    assert state.best_candidate is not None
    assert state.best_candidate.name == "Good"
    assert state.best_score == candidate.reward
    assert len(state.history) == 1
    assert state.history[0]["sharpe"] == 1.5
    assert state.history[0]["sortino"] == 2.0
    assert state.history[0]["native_sharpe"] == 99.0


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
        state.history.append({"sharpe": 0.2, "sortino": 0.3, "calmar": 0.1})

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


def test_phase_backtest_repairs_validation_failure(monkeypatch):
    """When validation fails, agent repair should be attempted and backtest rerun."""
    from types import SimpleNamespace

    from agent_market.strategy_miner.phases import phase_backtest

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        config = MinerConfig(repair_attempts=1)
        state = MinerState()

        bad_code = """
import os
from freqtrade.strategy import IStrategy
class Bad(IStrategy):
    timeframe = '5m'
    def populate_indicators(self, df, m): return df
    def populate_entry_trend(self, df, m): return df
    def populate_exit_trend(self, df, m): return df
"""

        sandbox = run_dir / "iter_0" / "sandbox"
        strat_dir = sandbox / "user_data" / "strategies"
        strat_dir.mkdir(parents=True, exist_ok=True)
        strat_path = strat_dir / "Bad.py"
        strat_path.write_text(bad_code, encoding="utf-8")

        candidate = StrategyCandidate(name="Bad", code=bad_code, strategy_path=strat_path)
        state.candidates.append(candidate)
        state.phase = Phase.BACKTEST

        class FakeAgent:
            def generate_strategy(self, prompt, *, on_turn=None, filename_hint=None):
                _ = prompt
                _ = on_turn
                _ = filename_hint
                strat_path.write_text(_VALID_STRATEGY, encoding="utf-8")
                return strat_path

        monkeypatch.setattr(
            "agent_market.strategy_miner._backtest.subprocess.run",
            lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""),
        )
        monkeypatch.setattr(
            "agent_market.strategy_miner._backtest.find_latest_backtest_zip",
            lambda *_a, **_kw: Path(td) / "bt.zip",
        )
        monkeypatch.setattr(
            "agent_market.strategy_miner._backtest.build_backtest_summary",
            lambda *_a, **_kw: {"profit_total_pct": 1.0, "trades": 20, "winrate": 0.6, "max_drawdown_abs": -1.0},
        )

        phase_backtest(state, config, run_dir, agent=FakeAgent())

        assert state.phase == Phase.EVALUATION
        assert candidate.validation_passed is True
        assert candidate.backtest_summary is not None



def test_phase_evaluation_applies_risk_constraints():
    """Candidates violating min_trades/winrate/drawdown are marked ineligible."""
    from agent_market.strategy_miner.phases import phase_evaluation

    state = MinerState()
    state.phase = Phase.EVALUATION
    candidate = StrategyCandidate(
        name="C",
        code=_VALID_STRATEGY,
        strategy_path=Path("/tmp/c.py"),
    )
    candidate.backtest_summary = {
        "profit_total_pct": 1.0,
        "trades": 3,
        "winrate": 0.6,
        "max_drawdown_abs": -1.0,
        "avg_profit_pct": 0.1,
        "sharpe": 0.5,
    }
    state.candidates.append(candidate)

    config = MinerConfig(min_trades=10)
    phase_evaluation(state, config)

    assert candidate.constraints_ok is False
    assert any("min_trades" in v for v in candidate.constraint_violations)
    assert state.best_candidate is None


def test_phase_evaluation_trade_count_penalty():
    """Lower-trade candidate should get a lower effective score than a similar higher-trade candidate."""
    from agent_market.strategy_miner.phases import phase_evaluation

    state_a = MinerState()
    state_a.phase = Phase.EVALUATION
    candidate_a = StrategyCandidate(
        name="FewTrades",
        code=_VALID_STRATEGY,
        strategy_path=Path("/tmp/x.py"),
    )
    candidate_a.backtest_summary = {
        "profit_total_pct": 10.0, "trades": 15,
        "winrate": 0.6, "max_drawdown_abs": -2.0,
        "realistic_sharpe": 1.0,
        "profit_factor": 1.4,
        "max_drawdown_pct": 4.0,
        "positive_days_ratio": 0.6,
        "return_over_drawdown": 2.5,
        "observation_days": 29,
    }
    state_a.candidates.append(candidate_a)

    state_b = MinerState()
    state_b.phase = Phase.EVALUATION
    candidate_b = StrategyCandidate(
        name="ManyTrades",
        code=_VALID_STRATEGY,
        strategy_path=Path("/tmp/y.py"),
    )
    candidate_b.backtest_summary = {
        "profit_total_pct": 10.0, "trades": 60,
        "winrate": 0.6, "max_drawdown_abs": -2.0,
        "realistic_sharpe": 1.0,
        "profit_factor": 1.4,
        "max_drawdown_pct": 4.0,
        "positive_days_ratio": 0.6,
        "return_over_drawdown": 2.5,
        "observation_days": 29,
    }
    state_b.candidates.append(candidate_b)

    config = MinerConfig(min_trades=0, target_trades=60, min_acceptable_trades=30)
    phase_evaluation(state_a, config)
    phase_evaluation(state_b, config)

    assert candidate_b.reward > candidate_a.reward


def test_phase_evaluation_no_penalty_above_30_trades():
    """Positive return and PF should contribute to the effective score."""
    from agent_market.strategy_miner.phases import phase_evaluation

    state = MinerState()
    state.phase = Phase.EVALUATION
    candidate = StrategyCandidate(
        name="ManyTrades",
        code=_VALID_STRATEGY,
        strategy_path=Path("/tmp/x.py"),
    )
    candidate.backtest_summary = {
        "profit_total_pct": 10.0, "trades": 60,
        "winrate": 0.6, "max_drawdown_abs": -2.0,
        "realistic_sharpe": 1.5,
        "profit_factor": 1.4,
        "max_drawdown_pct": 4.0,
        "positive_days_ratio": 0.6,
        "return_over_drawdown": 2.5,
        "observation_days": 29,
    }
    state.candidates.append(candidate)

    config = MinerConfig(min_trades=0)
    phase_evaluation(state, config)

    assert candidate.reward > 1.0


def test_phase_evaluation_kb_not_called_for_constraint_violated():
    """KB.add_elite should NOT be called for constraint-violated candidates."""
    from agent_market.strategy_miner.knowledge_base import KnowledgeBase
    from agent_market.strategy_miner.phases import phase_evaluation

    with tempfile.TemporaryDirectory() as td:
        kb = KnowledgeBase(Path(td) / "kb.json")
        state = MinerState()
        state.phase = Phase.EVALUATION
        candidate = StrategyCandidate(
            name="Bad", code=_VALID_STRATEGY, strategy_path=Path("/tmp/x.py"),
        )
        candidate.backtest_summary = {
            "profit_total_pct": 1.0, "trades": 3,
            "winrate": 0.6, "max_drawdown_abs": -1.0,
            "sharpe": 0.5,
        }
        state.candidates.append(candidate)

        config = MinerConfig(min_trades=10)
        phase_evaluation(state, config, kb=kb)

        assert len(kb.elites) == 0


def test_phase_evaluation_drawdown_pct_constraint():
    """max_drawdown_pct constraint should block candidates."""
    from agent_market.strategy_miner.phases import phase_evaluation

    state = MinerState()
    state.phase = Phase.EVALUATION
    candidate = StrategyCandidate(
        name="HighDD", code=_VALID_STRATEGY, strategy_path=Path("/tmp/x.py"),
    )
    candidate.backtest_summary = {
        "profit_total_pct": 10.0, "trades": 50,
        "winrate": 0.6, "max_drawdown_abs": -20.0,
        "max_drawdown_pct": 35.0,
        "realistic_sharpe": 1.5,
        "profit_factor": 1.4,
        "positive_days_ratio": 0.6,
        "return_over_drawdown": 0.2,
        "observation_days": 29,
    }
    state.candidates.append(candidate)

    config = MinerConfig(min_trades=0, max_drawdown_pct=30.0)
    phase_evaluation(state, config)

    assert candidate.constraints_ok is False
    assert any("max_drawdown_pct" in v for v in candidate.constraint_violations)


def test_phase_evaluation_enforces_realistic_profit_robustness_gates():
    from agent_market.strategy_miner.phases import phase_evaluation

    state = MinerState()
    state.phase = Phase.EVALUATION
    candidate = StrategyCandidate(
        name="Weak",
        code=_VALID_STRATEGY,
        strategy_path=Path("/tmp/weak.py"),
    )
    candidate.backtest_summary = {
        "profit_total_pct": 0.5,
        "trades": 200,
        "winrate": 0.58,
        "profit_factor": 1.2,
        "realistic_sharpe": 0.8,
        "max_drawdown_pct": 8.0,
        "positive_days_ratio": 0.48,
        "return_over_drawdown": 0.06,
        "observation_days": 29,
    }
    state.candidates.append(candidate)

    config = MinerConfig(
        min_trades=150,
        min_profit_pct=2.0,
        min_positive_days_ratio=0.55,
        min_return_over_drawdown=1.25,
    )
    phase_evaluation(state, config)

    assert candidate.constraints_ok is False
    assert any("min_profit_pct" in v for v in candidate.constraint_violations)
    assert any("min_positive_days_ratio" in v for v in candidate.constraint_violations)
    assert any("min_return_over_drawdown" in v for v in candidate.constraint_violations)


def test_phase_evaluation_uses_configurable_pair_loss_floor():
    from agent_market.strategy_miner.phases import phase_evaluation

    state = MinerState()
    state.phase = Phase.EVALUATION
    candidate = StrategyCandidate(
        name="PairFloor",
        code=_VALID_STRATEGY,
        strategy_path=Path("/tmp/pairfloor.py"),
    )
    candidate.backtest_summary = {
        "profit_total_pct": 8.0,
        "trades": 220,
        "winrate": 0.6,
        "profit_factor": 1.4,
        "realistic_sharpe": 0.9,
        "max_drawdown_pct": 6.0,
        "positive_days_ratio": 0.62,
        "return_over_drawdown": 1.33,
        "observation_days": 29,
        "results_per_pair": [
            {"key": "BTC/USDT", "profit_total_pct": 3.0},
            {"key": "XRP/USDT", "profit_total_pct": -0.75},
        ],
    }
    state.candidates.append(candidate)

    config = MinerConfig(
        min_trades=150,
        min_winrate=0.55,
        min_profit_factor=1.15,
        min_profit_pct=2.0,
        min_positive_days_ratio=0.55,
        min_return_over_drawdown=1.25,
        min_pair_profit_pct=-1.0,
    )
    phase_evaluation(state, config)

    assert candidate.constraints_ok is True


def test_phase_analysis_no_increment_on_infra_failure():
    """Iteration should not increment when no candidates have backtest results."""
    from agent_market.strategy_miner.phases import phase_analysis

    with tempfile.TemporaryDirectory() as td:
        state = MinerState()
        state.iteration = 2
        candidate = StrategyCandidate("FailedStrat", "code", Path("/tmp/x.py"))
        candidate.iteration = 2
        candidate.backtest_summary = None  # infra failure
        candidate.diagnosis = "Agent unavailable"
        state.candidates.append(candidate)

        config = MinerConfig(max_iterations=10)
        phase_analysis(state, config, Path(td))

        # Should retry without incrementing
        assert state.phase == Phase.STRATEGY_GEN
        assert state.iteration == 2  # not incremented
