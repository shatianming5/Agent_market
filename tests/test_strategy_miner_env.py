from __future__ import annotations

from pathlib import Path


def test_load_dotenv_fallback_sets_missing_env(monkeypatch, tmp_path: Path) -> None:
    from agent_market.strategy_miner import agent_adapter

    env_path = tmp_path / ".env"
    env_path.write_text("LLM_API_KEY=_\nOPENAI_MODEL=gpt-5.2\n", encoding="utf-8")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    agent_adapter._load_dotenv_fallback(env_path)

    assert agent_adapter.os.environ["LLM_API_KEY"] == "_"
    assert agent_adapter.os.environ["OPENAI_MODEL"] == "gpt-5.2"


def test_load_dotenv_fallback_preserves_existing_env(monkeypatch, tmp_path: Path) -> None:
    from agent_market.strategy_miner import agent_adapter

    env_path = tmp_path / ".env"
    env_path.write_text("LLM_API_KEY=file-value\n", encoding="utf-8")
    monkeypatch.setenv("LLM_API_KEY", "env-value")

    agent_adapter._load_dotenv_fallback(env_path)

    assert agent_adapter.os.environ["LLM_API_KEY"] == "env-value"
