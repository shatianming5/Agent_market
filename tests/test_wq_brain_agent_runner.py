"""agent_runner tests with mocked subprocess."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_market.wq_brain.agent_runner import AgentConfig, _build_system_prompt, run_agent


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
    # All placeholders should be substituted (no remaining {WORD})
    import re
    leftover = re.findall(r"\{[A-Z_]+\}", prompt)
    assert leftover == [], f"un-substituted placeholders: {leftover}"
    assert "testtag" in prompt
    assert "USA" in prompt
    assert "TOP3000" in prompt
    assert "1.3" in prompt or "1.30" in prompt  # sharpe min
    assert "1.1" in prompt or "1.10" in prompt  # fitness min
    assert "42" in prompt  # max_turns
    assert "yes" in prompt  # auto_submit


def test_run_agent_creates_run_dir_and_writes_artifacts(isolated_artifacts, tmp_path):
    config = AgentConfig(tag="t1", max_turns=5, timeout_sec=10.0)

    with patch("agent_market.wq_brain.agent_runner.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        summary = run_agent(config)

    run_dir = Path(summary["run_dir"])
    assert run_dir.exists()
    assert (run_dir / "system_prompt.md").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "summary.json").exists()
    assert summary["hermes_returncode"] == 0


def test_run_agent_passes_correct_hermes_flags(isolated_artifacts):
    config = AgentConfig(
        tag="t1",
        max_turns=20,
        model="MiniMax-M2.7",
        hermes_provider="openrouter",
        hermes_yolo=True,
        hermes_toolsets="terminal,file",
        hermes_reasoning_effort="high",
        timeout_sec=10.0,
    )

    with patch("agent_market.wq_brain.agent_runner.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        run_agent(config)

    call_args = mock_run.call_args
    cmd = call_args[0][0]
    assert cmd[0] == "hermes"
    assert "--max-turns" in cmd and "20" in cmd
    assert "--model" in cmd and "MiniMax-M2.7" in cmd
    assert "--provider" in cmd and "openrouter" in cmd
    assert "--toolsets" in cmd and "terminal,file" in cmd
    assert "--reasoning-effort" in cmd and "high" in cmd
    assert "--yolo" in cmd


def test_run_agent_handles_timeout(isolated_artifacts):
    import subprocess
    config = AgentConfig(tag="t1", max_turns=5, timeout_sec=1.0)

    with patch("agent_market.wq_brain.agent_runner.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired("hermes", 1.0)
        summary = run_agent(config)

    assert summary["hermes_returncode"] == -1
