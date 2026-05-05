"""agent_runner tests with mocked subprocess."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_market.wq_brain.agent_runner import (
    AgentConfig,
    _build_hermes_cmd,
    _build_opencode_cmd,
    _build_prior_knowledge_block,
    _build_system_prompt,
    _family_diversity_hint,
    _resolve_cli,
    _tried_family_concentration_hint,
    run_agent,
)
from agent_market.wq_brain.dtypes import AlphaPoolEntry
from agent_market.wq_brain.paths import alpha_pool_path
from agent_market.wq_brain.pool import AlphaPool


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


def test_build_opencode_cmd_auto_prefixes_custom_provider():
    config = AgentConfig(tag="t1", model="my-model")
    cmd = _build_opencode_cmd(config, "do the thing")
    assert cmd == ["opencode", "run", "-m", "custom/my-model", "do the thing"]


def test_build_opencode_cmd_keeps_explicit_provider():
    config = AgentConfig(tag="t1", model="anthropic/claude-sonnet-4")
    cmd = _build_opencode_cmd(config, "x")
    assert cmd == ["opencode", "run", "-m", "anthropic/claude-sonnet-4", "x"]


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


def test_family_diversity_hint_calls_out_missing_families():
    # Two ACTIVE alphas, both decay_linear → hint must list 8 missing families
    hint = _family_diversity_hint(["decay_linear", "decay_linear"])
    assert "MUST" in hint
    assert "decay_linear" in hint  # dominant family is shown
    # canonical missing families should appear
    for fam in ("ts_corr_pv", "intraday_range", "vwap_dev"):
        assert fam in hint


def test_family_diversity_hint_empty_pool_returns_empty():
    assert _family_diversity_hint([]) == ""


def test_family_diversity_hint_full_coverage_picks_lowest():
    """When all 8 families present, hint asks for lowest-count next."""
    full = list({"ts_corr_pv", "intraday_range", "vwap_dev", "volume_rank",
                 "open_gap", "humped", "multi_signal", "sector_relative"})
    hint = _family_diversity_hint(full)
    assert "LOWEST count" in hint


def test_tried_family_concentration_hint_silent_below_threshold():
    """7/10 of one family — exactly at threshold (default 0.7), should fire."""
    records = [
        {"ts": i, "expr": "rank(group_zscore(close, sector))"} for i in range(7)
    ] + [
        {"ts": 100 + i, "expr": "rank(close)"} for i in range(3)
    ]
    hint = _tried_family_concentration_hint(records)
    assert "STUCK IN" in hint
    assert "sector_relative" in hint
    assert "7/10" in hint


def test_tried_family_concentration_hint_quiet_when_diverse():
    """No single family >=70% → no hint."""
    records = [
        {"ts": 1, "expr": "rank(close)"},  # other
        {"ts": 2, "expr": "rank(ts_corr(close, volume, 20))"},  # ts_corr_pv
        {"ts": 3, "expr": "rank(close/vwap)"},  # vwap_dev
        {"ts": 4, "expr": "rank((high-low)/close)"},  # intraday_range
        {"ts": 5, "expr": "rank(ts_rank(volume, 20))"},  # volume_rank
        {"ts": 6, "expr": "hump(rank(close))"},  # humped
        {"ts": 7, "expr": "rank(open - ts_delay(close, 1))"},  # open_gap
        {"ts": 8, "expr": "rank(group_zscore(close, sector))"},  # sector_relative
        {"ts": 9, "expr": "rank(close) + 0.5 * rank(volume)"},  # multi_signal
        {"ts": 10, "expr": "rank(close)"},
    ]
    assert _tried_family_concentration_hint(records) == ""


def test_tried_family_concentration_hint_too_few_records():
    records = [{"ts": i, "expr": "rank(close)"} for i in range(5)]
    assert _tried_family_concentration_hint(records) == ""


def test_tried_family_concentration_hint_uses_recency():
    """Older records shouldn't pollute the recent-window analysis."""
    # 10 sector_relative early, then 10 fresh ts_corr_pv → window=10 should
    # see only ts_corr_pv (concentrated)
    records = [
        {"ts": i, "expr": "rank(group_zscore(close, sector))"} for i in range(10)
    ] + [
        {"ts": 100 + i, "expr": "rank(ts_corr(close, volume, 20))"} for i in range(10)
    ]
    hint = _tried_family_concentration_hint(records, window=10)
    assert "ts_corr_pv" in hint
    assert "sector_relative" not in hint  # the recent 10 are all ts_corr_pv


def test_tried_family_concentration_hint_includes_anti_examples():
    """Hint should mention concrete alternative expressions to try."""
    records = [
        {"ts": i, "expr": "rank(group_zscore(close, sector))"} for i in range(8)
    ] + [
        {"ts": 100 + i, "expr": "rank(close)"} for i in range(2)
    ]
    hint = _tried_family_concentration_hint(records)
    # Should suggest something concrete for sector_relative
    assert any(token in hint for token in ("hump", "high - low", "vwap"))


def test_build_prior_knowledge_block_renders_family_column(isolated_artifacts, tmp_path):
    """ACTIVE table must contain family column populated by infer_family."""
    pool = AlphaPool(alpha_pool_path("difamily"))
    pool.add(AlphaPoolEntry(
        alpha_id="a1",
        expr="rank(ts_corr(close, volume, 20))",  # ts_corr_pv
        sharpe=1.30, fitness=1.10, returns=0.20, turnover=0.45,
        settings_dict={}, tag="difamily", source="test", verified_status="ACTIVE",
    ))
    pool.add(AlphaPoolEntry(
        alpha_id="a2",
        expr="rank(close / vwap)",  # vwap_dev
        sharpe=1.40, fitness=1.20, returns=0.22, turnover=0.40,
        settings_dict={}, tag="difamily", source="test", verified_status="ACTIVE",
    ))

    block = _build_prior_knowledge_block("difamily", max_pool=10)
    # family column header present
    assert "| alpha_id | family | expr | sh | fi | to |" in block
    assert "`ts_corr_pv`" in block
    assert "`vwap_dev`" in block
    # diversity hint must call out the missing 6 canonical families
    assert "MUST" in block
    for fam in ("intraday_range", "volume_rank", "open_gap", "humped",
                "multi_signal", "sector_relative"):
        assert fam in block
