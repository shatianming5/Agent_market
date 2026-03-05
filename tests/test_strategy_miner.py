"""Smoke tests for strategy_miner module."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# dtypes
# ---------------------------------------------------------------------------


def test_miner_config_from_dict():
    from agent_market.strategy_miner.dtypes import MinerConfig

    cfg = MinerConfig.from_dict({"model": "gpt-4o", "max_iterations": 3, "unknown_key": 99})
    assert cfg.model == "gpt-4o"
    assert cfg.max_iterations == 3
    assert cfg.max_turns == 30  # default


def test_miner_config_max_retries_default():
    from agent_market.strategy_miner.dtypes import MinerConfig

    cfg = MinerConfig()
    assert cfg.max_retries == 2


def test_miner_config_max_retries_override():
    from agent_market.strategy_miner.dtypes import MinerConfig

    cfg = MinerConfig.from_dict({"max_retries": 5})
    assert cfg.max_retries == 5


def test_miner_state_roundtrip():
    from agent_market.strategy_miner.dtypes import MinerState, Phase, StrategyCandidate

    state = MinerState()
    state.phase = Phase.BACKTEST
    state.iteration = 2
    candidate = StrategyCandidate(
        name="TestStrat", code="class X: pass", strategy_path=Path("/tmp/t.py")
    )
    state.candidates.append(candidate)
    state.best_candidate = candidate
    state.best_reward = 0.42

    d = state.to_dict()
    j = json.dumps(d)
    state2 = MinerState.from_dict(json.loads(j))
    assert state2.run_id == state.run_id
    assert state2.phase == Phase.BACKTEST
    assert state2.iteration == 2
    assert state2.best_reward == 0.42
    assert state2.best_candidate.name == "TestStrat"
    assert len(state2.candidates) == 1


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------


def test_compute_reward_basic():
    from agent_market.strategy_miner.grading import compute_reward

    summary = {
        "profit_total_pct": 10.0,
        "trades": 50,
        "winrate": 0.6,
        "max_drawdown_abs": -5.0,
        "avg_profit_pct": 0.5,
    }
    weights = {
        "sharpe": 0.3, "profit_pct": 0.2, "max_drawdown": 0.15,
        "winrate": 0.1, "trade_count": 0.05, "stability": 0.1,
    }
    reward, comps = compute_reward(summary, weights)
    assert -1.0 <= reward <= 1.0
    assert "sharpe" in comps
    assert "profit_pct" in comps


def test_compute_reward_edge_cases():
    from agent_market.strategy_miner.grading import compute_reward

    # Zero trades
    summary = {"profit_total_pct": 0, "trades": 0, "winrate": 0, "max_drawdown_abs": 0, "avg_profit_pct": 0}
    weights = {"sharpe": 1.0}
    reward, _ = compute_reward(summary, weights)
    assert -1.0 <= reward <= 1.0

    # None values
    summary = {}
    reward, _ = compute_reward(summary, {"profit_pct": 1.0})
    assert reward == 0.0


def test_enhanced_reward_without_factors():
    from agent_market.strategy_miner.grading import compute_enhanced_reward

    summary = {"profit_total_pct": 10.0, "trades": 50, "winrate": 0.6, "max_drawdown_abs": -5.0, "avg_profit_pct": 0.5}
    weights = {"sharpe": 0.3, "profit_pct": 0.2}
    reward, comps = compute_enhanced_reward(summary, weights)
    assert -1.0 <= reward <= 1.0
    assert "factor_quality" not in comps  # no factor scores


def test_enhanced_reward_with_factors():
    from agent_market.strategy_miner.grading import compute_enhanced_reward

    summary = {"profit_total_pct": 10.0, "trades": 50, "winrate": 0.6, "max_drawdown_abs": -5.0, "avg_profit_pct": 0.5}
    weights = {"sharpe": 0.3, "profit_pct": 0.2}
    factor_scores = {"best_ic": 0.05, "best_sharpe": 1.5}
    reward, comps = compute_enhanced_reward(summary, weights, factor_scores=factor_scores)
    assert -1.0 <= reward <= 1.0
    assert "factor_quality" in comps


# ---------------------------------------------------------------------------
# sandbox validation
# ---------------------------------------------------------------------------


def test_validate_strategy_code_pass():
    from agent_market.strategy_miner.sandbox import validate_strategy_code

    code = """
from freqtrade.strategy import IStrategy
import talib.abstract as ta

class MyStrategy(IStrategy):
    timeframe = "5m"
    def populate_indicators(self, dataframe, metadata):
        return dataframe
    def populate_entry_trend(self, dataframe, metadata):
        return dataframe
    def populate_exit_trend(self, dataframe, metadata):
        return dataframe
"""
    ok, msg = validate_strategy_code(code)
    assert ok, f"Expected pass: {msg}"


def test_validate_strategy_code_forbidden_import():
    from agent_market.strategy_miner.sandbox import validate_strategy_code

    code = """
import os
from freqtrade.strategy import IStrategy
class Bad(IStrategy):
    def populate_indicators(self, df, m): return df
    def populate_entry_trend(self, df, m): return df
    def populate_exit_trend(self, df, m): return df
"""
    ok, msg = validate_strategy_code(code)
    assert not ok
    assert "os" in msg


def test_validate_strategy_code_missing_method():
    from agent_market.strategy_miner.sandbox import validate_strategy_code

    code = """
from freqtrade.strategy import IStrategy
class NoMethods(IStrategy):
    pass
"""
    ok, msg = validate_strategy_code(code)
    assert not ok
    assert "Missing" in msg


def test_validate_strategy_code_no_istrategy():
    from agent_market.strategy_miner.sandbox import validate_strategy_code

    code = """
class NotAStrategy:
    def populate_indicators(self, df, m): return df
    def populate_entry_trend(self, df, m): return df
    def populate_exit_trend(self, df, m): return df
"""
    ok, msg = validate_strategy_code(code)
    assert not ok
    assert "IStrategy" in msg


def test_validate_strategy_code_forbidden_call():
    from agent_market.strategy_miner.sandbox import validate_strategy_code

    code = """
from freqtrade.strategy import IStrategy
class Bad(IStrategy):
    def populate_indicators(self, df, m):
        eval("1+1")
        return df
    def populate_entry_trend(self, df, m): return df
    def populate_exit_trend(self, df, m): return df
"""
    ok, msg = validate_strategy_code(code)
    assert not ok
    assert "eval" in msg


# ---------------------------------------------------------------------------
# knowledge base
# ---------------------------------------------------------------------------


def test_knowledge_base_roundtrip():
    from agent_market.strategy_miner.knowledge_base import KnowledgeBase

    with tempfile.TemporaryDirectory() as td:
        kb = KnowledgeBase(Path(td) / "kb.json")
        kb.add_elite("S1", "code1", 0.8, {"profit_total_pct": 10, "trades": 50}, 0)
        kb.add_elite("S2", "code2", 0.5, {"profit_total_pct": 5, "trades": 30}, 1)
        kb.add_failure("F1", 0, "validation", "Bad import")

        assert len(kb.elites) == 2
        assert kb.elites[0]["reward"] == 0.8  # sorted desc
        assert len(kb.failures) == 1

        # Reload
        kb2 = KnowledgeBase(Path(td) / "kb.json")
        assert len(kb2.elites) == 2
        assert kb2.to_dict()["top_reward"] == 0.8


# ---------------------------------------------------------------------------
# evolution
# ---------------------------------------------------------------------------


def test_mutate_parameters():
    from agent_market.strategy_miner.evolution import mutate_parameters

    code = """
class MyStrategy(IStrategy):
    stoploss = -0.10
    minimal_roi = 0.05
    buy_rsi_threshold = 30
"""
    mutated = mutate_parameters(code, intensity=1.0)
    # At least some parameter should change
    assert isinstance(mutated, str)
    assert "class MyStrategy" in mutated


def test_mutate_indicators():
    from agent_market.strategy_miner.evolution import mutate_indicators

    code = """
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=20)
"""
    mutated = mutate_indicators(code, n_swaps=2)
    assert isinstance(mutated, str)


def test_evolve_strategy():
    from agent_market.strategy_miner.evolution import evolve_strategy

    code = """
from freqtrade.strategy import IStrategy
import talib.abstract as ta

class TestStrat(IStrategy):
    stoploss = -0.10
    minimal_roi = 0.05
    timeframe = "5m"

    def populate_indicators(self, dataframe, metadata):
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe.loc[dataframe['rsi'] < 30, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe.loc[dataframe['rsi'] > 70, 'exit_long'] = 1
        return dataframe
"""
    evolved, ops = evolve_strategy(code, mutation_intensity=1.0, indicator_swaps=1)
    assert isinstance(evolved, str)
    assert isinstance(ops, list)
    assert len(ops) >= 1


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def test_build_strategy_gen_prompt():
    from agent_market.strategy_miner.prompts import build_strategy_gen_prompt

    p = build_strategy_gen_prompt(
        iteration=0,
        sandbox_path="/tmp/sandbox",
        freqtrade_config="config.json",
        timerange="20250101-20260101",
        history=[],
        best_reward=float("-inf"),
    )
    assert "IStrategy" in p
    assert "Iteration 0" in p


def test_build_analysis_prompt():
    from agent_market.strategy_miner.prompts import build_analysis_prompt

    p = build_analysis_prompt(
        strategy_code="class X: pass",
        backtest_summary={"profit_total_pct": 5.0, "trades": 30},
        reward=0.5,
        reward_components={"sharpe": 0.3},
    )
    assert "diagnosis" in p.lower()


# ---------------------------------------------------------------------------
# Phase.EVOLVE in dtypes
# ---------------------------------------------------------------------------


def test_phase_evolve_enum():
    from agent_market.strategy_miner.dtypes import Phase

    assert Phase.EVOLVE.value == "evolve"
    # Roundtrip
    assert Phase("evolve") == Phase.EVOLVE


def test_miner_config_evolve_fields():
    from agent_market.strategy_miner.dtypes import MinerConfig

    cfg = MinerConfig.from_dict({"evolve_enabled": False, "evolve_every_n": 3})
    assert cfg.evolve_enabled is False
    assert cfg.evolve_every_n == 3
    # Defaults
    cfg2 = MinerConfig()
    assert cfg2.evolve_enabled is True
    assert cfg2.mutation_intensity == 0.3


def test_miner_state_evolve_roundtrip():
    """Verify EVOLVE phase survives serialization."""
    from agent_market.strategy_miner.dtypes import MinerState, Phase

    state = MinerState()
    state.phase = Phase.EVOLVE
    d = state.to_dict()
    state2 = MinerState.from_dict(d)
    assert state2.phase == Phase.EVOLVE


# ---------------------------------------------------------------------------
# prompts with KB context
# ---------------------------------------------------------------------------


def test_prompt_with_kb_context():
    from agent_market.strategy_miner.prompts import build_strategy_gen_prompt

    elites = [
        {"name": "EliteA", "reward": 0.8, "profit_pct": 15, "trades": 80, "winrate": 0.65},
    ]
    failure_summary = "  - iter0 [validation]: Forbidden import: os"
    p = build_strategy_gen_prompt(
        iteration=2,
        sandbox_path="/tmp/sandbox",
        freqtrade_config="config.json",
        timerange="20250101-20260101",
        history=[],
        best_reward=0.8,
        elite_summaries=elites,
        failure_summary=failure_summary,
    )
    assert "Elite Strategy Archive" in p
    assert "EliteA" in p
    assert "Failure Patterns" in p
    assert "Forbidden import" in p


def test_prompt_without_kb_context():
    from agent_market.strategy_miner.prompts import build_strategy_gen_prompt

    p = build_strategy_gen_prompt(
        iteration=0,
        sandbox_path="/tmp/sandbox",
        freqtrade_config="config.json",
        timerange="20250101-20260101",
        history=[],
        best_reward=float("-inf"),
        elite_summaries=None,
        failure_summary=None,
    )
    assert "Elite Strategy Archive" not in p
    assert "Failure Patterns" not in p


# ---------------------------------------------------------------------------
# KB integration with prompts end-to-end
# ---------------------------------------------------------------------------


def test_kb_feeds_into_prompt():
    """Knowledge base data flows correctly into prompt generation."""
    from agent_market.strategy_miner.knowledge_base import KnowledgeBase
    from agent_market.strategy_miner.prompts import build_strategy_gen_prompt

    with tempfile.TemporaryDirectory() as td:
        kb = KnowledgeBase(Path(td) / "kb.json")
        kb.add_elite("TopStrat", "code...", 0.9, {"profit_total_pct": 20, "trades": 100, "winrate": 0.7, "max_drawdown_abs": -3}, 0)
        kb.add_failure("BadStrat", 1, "backtest", "Timeout after 300s")

        p = build_strategy_gen_prompt(
            iteration=2,
            sandbox_path="/tmp/sandbox",
            freqtrade_config="cfg.json",
            timerange="20250101-20260101",
            history=[],
            best_reward=0.9,
            elite_summaries=kb.elites[:3],
            failure_summary=kb.failure_summary(5),
        )
        assert "TopStrat" in p
        assert "Timeout" in p


# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------


def test_all_imports():
    """Ensure the full module tree is importable."""
    from agent_market.strategy_miner import KnowledgeBase, MinerConfig, MinerState, Phase, StrategyCandidate, run_strategy_miner
    from agent_market.strategy_miner.agent_adapter import StrategyAgent
    from agent_market.strategy_miner.evolution import evolve_strategy, mutate_parameters
    from agent_market.strategy_miner.grading import compute_enhanced_reward, compute_factor_score
    from agent_market.strategy_miner.phases import phase_evolve
    from agent_market.strategy_miner.sandbox import validate_strategy_code
    assert Phase.EVOLVE.value == "evolve"


# ---------------------------------------------------------------------------
# agent_adapter: max_retries & error handling
# ---------------------------------------------------------------------------


def test_agent_adapter_passes_max_retries():
    """StrategyAgent should forward max_retries to OpenCodeExecutor."""
    from unittest.mock import MagicMock, patch

    with patch("agent_market.strategy_miner.agent_adapter.OpenCodeExecutor") as MockExec:
        mock_instance = MagicMock()
        mock_instance.close.return_value = None
        MockExec.return_value = mock_instance

        from agent_market.strategy_miner.agent_adapter import StrategyAgent

        agent = StrategyAgent(
            workspace=Path("/tmp/test_ws"),
            model="test-model",
            max_retries=5,
        )
        _, kwargs = MockExec.call_args
        assert kwargs["max_retries"] == 5
        agent.close()


def test_agent_adapter_default_max_retries():
    """Default max_retries should be 2."""
    from unittest.mock import MagicMock, patch

    with patch("agent_market.strategy_miner.agent_adapter.OpenCodeExecutor") as MockExec:
        mock_instance = MagicMock()
        mock_instance.close.return_value = None
        MockExec.return_value = mock_instance

        from agent_market.strategy_miner.agent_adapter import StrategyAgent

        agent = StrategyAgent(
            workspace=Path("/tmp/test_ws"),
            model="test-model",
        )
        _, kwargs = MockExec.call_args
        assert kwargs["max_retries"] == 2
        agent.close()


def test_agent_adapter_close_is_idempotent():
    """Calling close() multiple times should not raise."""
    from agent_market.strategy_miner.agent_adapter import StrategyAgent

    agent = StrategyAgent(workspace=Path("/tmp/ws"), provider="template")
    agent.close()
    agent.close()  # should not raise


def test_agent_adapter_run_after_close_raises():
    """Calling run() after close() should raise RuntimeError."""
    from agent_market.strategy_miner.agent_adapter import StrategyAgent

    agent = StrategyAgent(workspace=Path("/tmp/ws"), provider="template")
    agent.close()
    with pytest.raises(RuntimeError, match="already closed"):
        agent.run("test prompt")


def test_agent_adapter_no_model_falls_back_to_template():
    """StrategyAgent without any provider config should gracefully fall back."""
    import os
    from unittest.mock import patch

    from agent_market.strategy_miner.agent_adapter import StrategyAgent

    with patch.dict(os.environ, {}, clear=True):
        with tempfile.TemporaryDirectory() as td:
            agent = StrategyAgent(workspace=Path(td))
            out = agent.generate_strategy("dummy prompt")
            assert out is not None
            assert out.exists()


def test_agent_adapter_opencode_without_model_raises():
    """If provider=opencode is forced, missing model should raise."""
    import os
    from unittest.mock import patch

    from agent_market.strategy_miner.agent_adapter import StrategyAgent

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="model"):
            StrategyAgent(workspace=Path("/tmp/ws"), provider="opencode")
