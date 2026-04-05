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


def test_miner_config_nested_sections():
    from agent_market.strategy_miner.dtypes import MinerConfig

    cfg = MinerConfig.from_dict(
        {
            "budget": {
                "provider": "opencode",
                "max_iterations": 3,
                "max_turns": 9,
                "max_retries": 4,
                "repair_attempts": 2,
            },
            "tools": {
                "tool_allowlist": ["file"],
                "bash_allow": False,
                "bash_timeout": 12,
                "bash_allowlist": ["echo ", "python3 "],
            },
            "evaluation": {
                "min_trades": 25,
                "max_abs_drawdown": 12.5,
                "min_winrate": 0.55,
                "benchmark_suite": "benchmark_pack/default",
            },
            "portfolio": {
                "portfolio_enabled": True,
                "portfolio_top_k": 4,
                "portfolio_min_candidates": 2,
                "portfolio_correlation_threshold": 0.8,
                "portfolio_max_weight": 0.5,
            },
        }
    )

    assert cfg.provider == "opencode"
    assert cfg.max_iterations == 3
    assert cfg.max_turns == 9
    assert cfg.max_retries == 4
    assert cfg.repair_attempts == 2

    assert cfg.tool_allowlist == ["file"]
    assert cfg.bash_allow is False
    assert cfg.bash_timeout == 12
    assert cfg.bash_allowlist == ["echo ", "python3 "]

    assert cfg.min_trades == 25
    assert cfg.max_abs_drawdown == 12.5
    assert cfg.min_winrate == 0.55
    assert cfg.benchmark_suite == "benchmark_pack/default"
    assert cfg.portfolio_enabled is True
    assert cfg.portfolio_top_k == 4
    assert cfg.portfolio_correlation_threshold == 0.8
    assert cfg.portfolio_max_weight == 0.5



def test_miner_config_defaults():
    from agent_market.strategy_miner.dtypes import MinerConfig

    cfg = MinerConfig()
    assert cfg.max_retries == 2
    assert cfg.max_parallel_roles == 1
    assert cfg.max_drawdown_pct == 0.0


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
    state.best_score = 0.42

    d = state.to_dict()
    j = json.dumps(d)
    state2 = MinerState.from_dict(json.loads(j))
    assert state2.run_id == state.run_id
    assert state2.phase == Phase.BACKTEST
    assert state2.iteration == 2
    assert state2.best_score == 0.42
    assert state2.best_candidate.name == "TestStrat"
    assert len(state2.candidates) == 1


def test_miner_state_backward_compat_best_reward():
    """Old checkpoints with 'best_reward' should load into best_score."""
    from agent_market.strategy_miner.dtypes import MinerState, Phase

    old_data = {
        "run_id": "abc123",
        "phase": "strategy_gen",
        "iteration": 1,
        "best_reward": 0.75,
        "candidates": [],
        "history": [],
    }
    state = MinerState.from_dict(old_data)
    assert state.best_score == 0.75


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------


def test_compute_factor_score_none_inputs():
    from agent_market.strategy_miner.grading import compute_factor_score

    assert compute_factor_score() is None
    assert compute_factor_score(features_parquet=None, expression="x") is None


# ---------------------------------------------------------------------------
# sandbox validation
# ---------------------------------------------------------------------------


def test_validate_strategy_code_pass():
    from agent_market.strategy_miner.sandbox import validate_strategy_code

    code = """
from freqtrade.strategy import IStrategy
import pandas_ta as ta

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


def test_validate_strategy_code_blocks_negative_shift():
    from agent_market.strategy_miner.sandbox import validate_strategy_code

    code = """
from freqtrade.strategy import IStrategy
class Leaky(IStrategy):
    def populate_indicators(self, df, m):
        df['future'] = df['close'].shift(-1)
        return df
    def populate_entry_trend(self, df, m): return df
    def populate_exit_trend(self, df, m): return df
"""
    ok, msg = validate_strategy_code(code)
    assert not ok
    assert "look-ahead" in msg


def test_validate_strategy_code_blocks_centered_rolling():
    from agent_market.strategy_miner.sandbox import validate_strategy_code

    code = """
from freqtrade.strategy import IStrategy
class LeakyRolling(IStrategy):
    def populate_indicators(self, df, m):
        df['ma'] = df['close'].rolling(10, center=True).mean()
        return df
    def populate_entry_trend(self, df, m): return df
    def populate_exit_trend(self, df, m): return df
"""
    ok, msg = validate_strategy_code(code)
    assert not ok
    assert "look-ahead" in msg


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
        best_score=float("-inf"),
    )
    assert "IStrategy" in p
    assert "Iteration 0" in p


def test_build_analysis_prompt():
    from agent_market.strategy_miner.prompts import build_analysis_prompt

    p = build_analysis_prompt(
        strategy_code="class X: pass",
        backtest_summary={"profit_total_pct": 5.0, "trades": 30},
        metrics={"sharpe": 1.2, "sortino": 1.5},
    )
    assert "JSON" in p
    assert "strengths" in p
    assert "verdict" in p


def test_strategy_gen_prompt_openai_compatible_no_tool_tags():
    from agent_market.strategy_miner.prompts import build_strategy_gen_prompt

    p = build_strategy_gen_prompt(
        iteration=0,
        sandbox_path="/tmp/sandbox",
        freqtrade_config="config.json",
        timerange="20250101-20260101",
        history=[],
        best_score=float("-inf"),
        provider="openai_compatible",
    )
    assert "You MAY use tool-call tags" not in p
    assert "single Python code block" in p


def test_strategy_gen_prompt_opencode_has_tool_tags():
    from agent_market.strategy_miner.prompts import build_strategy_gen_prompt

    p = build_strategy_gen_prompt(
        iteration=0,
        sandbox_path="/tmp/sandbox",
        freqtrade_config="config.json",
        timerange="20250101-20260101",
        history=[],
        best_score=float("-inf"),
        provider="opencode",
    )
    assert "<write" in p or "tool-call tags" in p


def test_strategy_gen_prompt_market_orders_advice():
    from agent_market.strategy_miner.prompts import build_strategy_gen_prompt

    p = build_strategy_gen_prompt(
        iteration=0,
        sandbox_path="/tmp/sandbox",
        freqtrade_config="config.json",
        timerange="20250101-20260101",
        history=[],
        best_score=float("-inf"),
    )
    assert "market" in p.lower()


def test_strategy_gen_prompt_market_profile():
    from agent_market.strategy_miner.prompts import build_strategy_gen_prompt

    p = build_strategy_gen_prompt(
        iteration=0,
        sandbox_path="/tmp/sandbox",
        freqtrade_config="config.json",
        timerange="20250101-20260101",
        history=[],
        best_score=float("-inf"),
        market_profile="- Trading pairs: BTC/USDT\n- Stake currency: USDT",
    )
    assert "Market Profile" in p
    assert "BTC/USDT" in p


def test_repair_prompt_openai_compatible_no_tool_tags():
    from agent_market.strategy_miner.prompts import build_repair_prompt

    p = build_repair_prompt(
        sandbox_path="/tmp/sandbox",
        strategy_rel_path="user_data/strategies/Foo.py",
        freqtrade_config="config.json",
        timerange="20250101-20260101",
        failure="Syntax error",
        attempt=1,
        max_attempts=3,
        provider="openai_compatible",
    )
    assert "Start by reading" not in p
    assert "single Python code block" in p


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
        best_score=0.8,
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
        best_score=float("-inf"),
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
            best_score=0.9,
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
    from agent_market.strategy_miner.grading import compute_factor_score
    from agent_market.strategy_miner.sandbox import validate_strategy_code


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
    from unittest.mock import MagicMock, patch

    with patch("agent_market.strategy_miner.agent_adapter.OpenCodeExecutor") as MockExec:
        mock_instance = MagicMock()
        mock_instance.close.return_value = None
        MockExec.return_value = mock_instance

        from agent_market.strategy_miner.agent_adapter import StrategyAgent

        agent = StrategyAgent(workspace=Path("/tmp/ws"), provider="opencode", model="test-model")
        agent.close()
        agent.close()  # should not raise


def test_agent_adapter_run_after_close_raises():
    """Calling run() after close() should raise RuntimeError."""
    from unittest.mock import MagicMock, patch

    with patch("agent_market.strategy_miner.agent_adapter.OpenCodeExecutor") as MockExec:
        mock_instance = MagicMock()
        mock_instance.close.return_value = None
        MockExec.return_value = mock_instance

        from agent_market.strategy_miner.agent_adapter import StrategyAgent

        agent = StrategyAgent(workspace=Path("/tmp/ws"), provider="opencode", model="test-model")
        agent.close()
        with pytest.raises(RuntimeError, match="already closed"):
            agent.run("test prompt")


def test_agent_adapter_no_model_errors_in_no_template_mode():
    """Without a configured model/credentials, StrategyAgent should error (no-template enforced)."""
    import os
    from unittest.mock import patch

    from agent_market.strategy_miner.agent_adapter import StrategyAgent

    with patch.dict(os.environ, {}, clear=True):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(ValueError, match="model"):
                StrategyAgent(workspace=Path(td))


def test_agent_adapter_opencode_without_model_raises():
    """If provider=opencode is forced, missing model should raise."""
    import os
    from unittest.mock import patch

    from agent_market.strategy_miner.agent_adapter import StrategyAgent

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="model"):
            StrategyAgent(workspace=Path("/tmp/ws"), provider="opencode")
