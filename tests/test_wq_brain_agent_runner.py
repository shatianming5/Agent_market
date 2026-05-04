"""agent_runner tests with mocked subprocess."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_market.wq_brain.agent_runner import (
    AgentConfig,
    _build_hermes_cmd,
    _build_opencode_cmd,
    _build_system_prompt,
    _resolve_cli,
    run_agent,
)


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


def test_build_system_prompt_substitutes_all_placeholders(isolated_artifacts, tmp_path):
    config = AgentConfig(
        tag="testtag",
        region="USA",
        universe="TOP3000",
        decay=6,
        neutralization="SUBINDUSTRY",
        max_turns=42,
        auto_submit=True,
        quality_sharpe_min=1.30,
        quality_fitness_min=1.10,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prompt = _build_system_prompt(config, run_dir)
    import re
    leftover = re.findall(r"\{[A-Z_]+\}", prompt)
    assert leftover == [], f"un-substituted placeholders: {leftover}"
    assert "testtag" in prompt
    assert "USA" in prompt
    assert "TOP3000" in prompt
    assert "42" in prompt
    assert "yes" in prompt


def test_resolve_cli_explicit_returns_input():
    assert _resolve_cli("hermes") == "hermes"
    assert _resolve_cli("opencode") == "opencode"


def test_resolve_cli_auto_prefers_opencode_when_available():
    with patch("agent_market.wq_brain.agent_runner.shutil.which") as mock_which:
        mock_which.side_effect = lambda c, **kw: "/usr/bin/opencode" if c == "opencode" else None
        assert _resolve_cli("auto") == "opencode"


def test_resolve_cli_auto_falls_back_to_hermes():
    with patch("agent_market.wq_brain.agent_runner.shutil.which") as mock_which:
        mock_which.side_effect = lambda c, **kw: "/usr/bin/hermes" if c == "hermes" else None
        assert _resolve_cli("auto") == "hermes"


def test_resolve_cli_auto_raises_when_none_available():
    with patch("agent_market.wq_brain.agent_runner.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="No agentic CLI found"):
            _resolve_cli("auto")


def test_build_opencode_cmd_includes_model_and_prompt():
    config = AgentConfig(tag="t1", model="my-model")
    cmd = _build_opencode_cmd(config, "do the thing")
    assert cmd == ["opencode", "run", "-m", "my-model", "do the thing"]


def test_build_opencode_cmd_requires_model():
    config = AgentConfig(tag="t1", model="")
    with pytest.raises(RuntimeError, match="opencode requires --model"):
        _build_opencode_cmd(config, "prompt")


def test_build_hermes_cmd_assembles_full_command(tmp_path):
    config = AgentConfig(
        tag="t1", max_turns=20, model="MiniMax", provider="openrouter",
        yolo=True, toolsets="terminal,file",
    )
    env: dict[str, str] = {}
    with patch("agent_market.wq_brain.agent_runner.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        cmd = _build_hermes_cmd(config, "prompt-text", tmp_path, env)
    assert cmd[:3] == ["hermes", "chat", "-Q"]
    assert "--toolsets" in cmd and "terminal,file" in cmd
    assert "--max-turns" in cmd and "20" in cmd
    assert "--source" in cmd and "wq-brain-agent" in cmd
    assert "-m" in cmd and "MiniMax" in cmd
    assert "--provider" in cmd and "openrouter" in cmd
    assert "--yolo" in cmd
    assert "-q" in cmd and "prompt-text" in cmd
    # HERMES_HOME should be set in env
    assert "HERMES_HOME" in env
    assert (tmp_path / "hermes_home").exists()


def test_run_agent_creates_run_dir_and_writes_artifacts(isolated_artifacts, tmp_path):
    config = AgentConfig(tag="t1", max_turns=5, timeout_sec=10.0,
                         cli="opencode", model="m1")

    with patch("agent_market.wq_brain.agent_runner.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        summary = run_agent(config)

    run_dir = Path(summary["run_dir"])
    assert run_dir.exists()
    assert (run_dir / "system_prompt.md").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "summary.json").exists()
    assert summary["agent_returncode"] == 0
    assert summary["cli"] == "opencode"


def test_run_agent_passes_correct_hermes_flags(isolated_artifacts):
    config = AgentConfig(
        tag="t1", max_turns=20, model="MiniMax", cli="hermes",
        provider="openrouter", yolo=True, toolsets="terminal,file",
        timeout_sec=10.0,
    )
    with patch("agent_market.wq_brain.agent_runner.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        run_agent(config)

    # Inspect the LAST subprocess.run call (the actual chat invocation;
    # there's no reasoning_effort config call when reasoning_effort is empty)
    cmd = mock_run.call_args_list[-1][0][0]
    assert cmd[0] == "hermes"
    assert cmd[1] == "chat"
    assert "--toolsets" in cmd and "terminal,file" in cmd
    assert "--max-turns" in cmd and "20" in cmd
    assert "-m" in cmd and "MiniMax" in cmd
    assert "--provider" in cmd and "openrouter" in cmd
    assert "--yolo" in cmd


def test_run_agent_handles_timeout(isolated_artifacts):
    import subprocess
    config = AgentConfig(tag="t1", max_turns=5, timeout_sec=1.0,
                         cli="opencode", model="m1")
    with patch("agent_market.wq_brain.agent_runner.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired("opencode", 1.0)
        summary = run_agent(config)
    assert summary["agent_returncode"] == -1
