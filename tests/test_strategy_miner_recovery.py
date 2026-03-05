"""Targeted tests for strategy miner recovery hardening."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


def _valid_strategy(class_name: str) -> str:
    return f"""
from __future__ import annotations

from freqtrade.strategy import IStrategy


class {class_name}(IStrategy):
    timeframe = "1h"
    minimal_roi = {{"0": 0.01}}
    stoploss = -0.10

    def populate_indicators(self, dataframe, metadata):
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        return dataframe
""".lstrip()


# ---------------------------------------------------------------------------
# Multi-candidate generation
# ---------------------------------------------------------------------------


def test_phase_strategy_gen_multi_candidate_parallel(monkeypatch):
    from agent_market.strategy_miner.dtypes import MinerConfig, MinerState, Phase
    from agent_market.strategy_miner.phases import phase_strategy_gen

    class FakeAgent:
        def __init__(self, sandbox: Path):
            self.sandbox = sandbox

        def generate_strategy(self, prompt: str, *, on_turn=None, filename_hint=None):
            _ = prompt
            _ = on_turn
            stem = Path(filename_hint or "Mined.py").stem
            out = self.sandbox / "user_data" / "strategies" / f"{stem}.py"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(_valid_strategy(stem), encoding="utf-8")
            return out

        def close(self) -> None:
            return

    monkeypatch.setattr(
        "agent_market.strategy_miner.phases.build_strategy_agent",
        lambda _cfg, sandbox: FakeAgent(sandbox),
    )

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        config = MinerConfig(candidates_per_iteration=3)
        state = MinerState()
        state.phase = Phase.STRATEGY_GEN

        phase_strategy_gen(state, config, run_dir)

        assert state.phase == Phase.BACKTEST
        assert len(state.candidates) == 3
        assert len(state.pending_candidate_idxs) == 3
        assert state.active_candidate_idx == state.pending_candidate_idxs[0]
        # Multi-candidate should isolate sandboxes.
        assert all("cand_" in str(c.strategy_path) for c in state.candidates)


def test_phase_strategy_gen_multi_candidate_partial_failure(monkeypatch):
    from agent_market.strategy_miner.dtypes import MinerConfig, MinerState, Phase
    from agent_market.strategy_miner.phases import phase_strategy_gen

    class FakeAgent:
        def __init__(self, sandbox: Path):
            self.sandbox = sandbox

        def generate_strategy(self, prompt: str, *, on_turn=None, filename_hint=None):
            _ = prompt
            _ = on_turn
            # Fail candidate idx=1 deterministically.
            stem = Path(filename_hint or "Mined.py").stem
            if stem.endswith("_1"):
                return None
            out = self.sandbox / "user_data" / "strategies" / f"{stem}.py"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(_valid_strategy(stem), encoding="utf-8")
            return out

        def close(self) -> None:
            return

    monkeypatch.setattr(
        "agent_market.strategy_miner.phases.build_strategy_agent",
        lambda _cfg, sandbox: FakeAgent(sandbox),
    )

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        config = MinerConfig(candidates_per_iteration=3)
        state = MinerState()
        state.phase = Phase.STRATEGY_GEN

        phase_strategy_gen(state, config, run_dir)

        assert state.phase == Phase.BACKTEST
        assert len(state.candidates) == 2
        assert len(state.pending_candidate_idxs) == 2


# ---------------------------------------------------------------------------
# Provider fallback chain
# ---------------------------------------------------------------------------


def test_strategy_agent_fallback_opencode_to_openai(monkeypatch, tmp_path: Path):
    from agent_market.agents.executor import AgentRunResult
    from agent_market.strategy_miner.agent_adapter import StrategyAgent

    seen: dict = {}

    class BoomOpenCode:
        def __init__(self, **_kwargs):
            raise RuntimeError("opencode_down")

    class FakeOpenAI:
        def __init__(self, *, model: str, **kwargs):
            seen["model"] = model
            seen["kwargs"] = dict(kwargs)

        def run(self, prompt: str, *, on_turn=None):
            _ = prompt
            _ = on_turn
            code = _valid_strategy("OpenAIFallbackStrategy")
            return AgentRunResult(
                assistant_text=f"```python\n{code}\n```",
                provider="openai_compatible",
                model=seen.get("model"),
            )

        def close(self) -> None:
            return

    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    monkeypatch.setattr("agent_market.strategy_miner.agent_adapter.OpenCodeExecutor", BoomOpenCode)
    monkeypatch.setattr("agent_market.strategy_miner.agent_adapter.OpenAIChatExecutor", FakeOpenAI)

    agent = StrategyAgent(workspace=tmp_path, provider="auto")
    out = agent.generate_strategy("dummy prompt")
    assert out is not None and out.exists()
    assert "class OpenAIFallbackStrategy" in out.read_text(encoding="utf-8")
    assert seen["model"] == "glm-4-flash"


def test_strategy_agent_fallback_to_template_when_openai_unavailable(monkeypatch, tmp_path: Path):
    from agent_market.strategy_miner.agent_adapter import StrategyAgent

    class BoomOpenCode:
        def __init__(self, **_kwargs):
            raise RuntimeError("opencode_down")

    class BoomOpenAI:
        def __init__(self, **_kwargs):
            raise RuntimeError("openai_down")

    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    monkeypatch.setattr("agent_market.strategy_miner.agent_adapter.OpenCodeExecutor", BoomOpenCode)
    monkeypatch.setattr("agent_market.strategy_miner.agent_adapter.OpenAIChatExecutor", BoomOpenAI)

    agent = StrategyAgent(workspace=tmp_path, provider="auto")
    out = agent.generate_strategy("dummy prompt")
    assert out is not None and out.exists()
    assert "TemplateRsiStrategy" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Look-ahead: new rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "df['x'] = df['close'].diff(-1)",
        "df['x'] = df['close'].bfill()",
        "df['x'] = df['close'].fillna(method='bfill')",
        "import numpy as np\ndf['x'] = np.roll(df['close'], -1)",
    ],
)
def test_validate_strategy_code_blocks_common_lookahead(expr: str):
    from agent_market.strategy_miner.sandbox import validate_strategy_code

    body = "\n".join("        " + line for line in expr.splitlines())
    code = f"""
from freqtrade.strategy import IStrategy

class Leaky(IStrategy):
    def populate_indicators(self, df, m):
{body}
        return df
    def populate_entry_trend(self, df, m):
        return df
    def populate_exit_trend(self, df, m):
        return df
"""

    ok, msg = validate_strategy_code(code)
    assert not ok
    assert "look-ahead" in msg


# ---------------------------------------------------------------------------
# Duplicate class name normalization
# ---------------------------------------------------------------------------


def test_phase_strategy_gen_renames_duplicate_class_safely(monkeypatch):
    from agent_market.strategy_miner.dtypes import MinerConfig, MinerState, Phase
    from agent_market.strategy_miner.phases import phase_strategy_gen
    from agent_market.strategy_miner.sandbox import validate_strategy_code

    class FakeAgent:
        def __init__(self, sandbox: Path):
            self.sandbox = sandbox

        def generate_strategy(self, prompt: str, *, on_turn=None, filename_hint=None):
            _ = prompt
            _ = on_turn
            _ = filename_hint
            out = self.sandbox / "user_data" / "strategies" / "DupStrategy.py"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(_valid_strategy("DupStrategy"), encoding="utf-8")
            return out

        def close(self) -> None:
            return

    monkeypatch.setattr(
        "agent_market.strategy_miner.phases.build_strategy_agent",
        lambda _cfg, sandbox: FakeAgent(sandbox),
    )

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        config = MinerConfig(candidates_per_iteration=3)
        state = MinerState()
        state.phase = Phase.STRATEGY_GEN

        phase_strategy_gen(state, config, run_dir)

        names = [c.name for c in state.candidates]
        assert len(names) == len(set(names))
        for c in state.candidates:
            ok, msg = validate_strategy_code(c.code)
            assert ok, msg
            assert f"class {c.name}" in c.code
