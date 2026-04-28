from __future__ import annotations

from types import SimpleNamespace

import agent_market.freqai.llm as llm_mod
from agent_market.agents.executor import AgentRunResult
from agent_market.freqai.llm import LLMConfig, request_completion


def test_llm_config_from_args_supports_opencode_fields() -> None:
    args = SimpleNamespace(
        llm_base_url="http://example.test/v1",
        llm_agent_url="http://agent.test",
        llm_api_key="dummy",
        llm_model="gpt-test",
        llm_provider="opencode",
        llm_workspace=".",
        llm_temperature=0.3,
        llm_max_tokens=256,
        llm_count=12,
        llm_retries=4,
        llm_timeout=30,
        llm_max_turns=9,
        llm_stale_timeout=120,
    )

    cfg = LLMConfig.from_args(args)

    assert cfg.provider == "opencode"
    assert cfg.agent_url == "http://agent.test"
    assert cfg.workspace == "."
    assert cfg.max_turns == 9
    assert cfg.stale_timeout == 120.0


def test_request_completion_uses_opencode_provider(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeOpenCodeExecutor:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            captured["init"] = kwargs

        def run(self, prompt: str, *, on_turn=None):  # noqa: ANN001, ARG002
            captured["prompt"] = prompt
            return AgentRunResult(
                assistant_text='{"expressions":[{"name":"alpha","expression":"close","description":"d","reason":"r","category":"trend"}]}',
                provider="opencode",
            )

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr("agent_market.agents.executor.OpenCodeExecutor", _FakeOpenCodeExecutor)

    cfg = LLMConfig(
        provider="opencode",
        model="gpt-test",
        workspace=".",
        agent_url="http://agent.test",
        retries=2,
        max_turns=7,
        stale_timeout=90,
    )

    content, usage = request_completion("return one expression", cfg)

    assert '"expressions"' in content
    assert usage is None
    assert captured["closed"] is True
    init = captured["init"]
    assert init["model"] == "gpt-test"
    assert init["base_url"] == "http://agent.test"
    assert init["max_turns"] == 7
    assert init["stale_timeout"] == 90.0

