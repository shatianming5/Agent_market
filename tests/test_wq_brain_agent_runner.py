"""agent_runner tests with mocked subprocess."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_market.wq_brain.agent_runner import (
    AgentConfig,
    _build_hermes_cmd,
    _build_opencode_cmd,
    _build_prior_knowledge_block,
    _build_system_prompt,
    _family_diversity_hint,
    _operator_skeleton,
    _resolve_cli,
    _tried_family_concentration_hint,
    _tried_skeleton_concentration_hint,
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
    assert "Compact Loop Mode" in prompt
    assert "Remote `simulate` at most 2 candidates" in prompt
    assert "never in parallel" in prompt
    assert "At least 1 `math` command is mandatory" in prompt
    assert "local-simulate-supported fields" in prompt
    assert "identifier '<field>' not available locally" in prompt


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
    assert cmd == [
        "opencode", "run",
        "--print-logs", "--log-level", "ERROR",
        "-m", "custom/my-model",
        "do the thing",
    ]


def test_build_opencode_cmd_keeps_explicit_provider():
    config = AgentConfig(tag="t1", model="anthropic/claude-sonnet-4")
    cmd = _build_opencode_cmd(config, "x")
    assert cmd == [
        "opencode", "run",
        "--print-logs", "--log-level", "ERROR",
        "-m", "anthropic/claude-sonnet-4",
        "x",
    ]


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

    with patch("agent_market.wq_brain.agent_runner._run_cli_with_group_timeout") as mock_run:
        mock_run.return_value = 0
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
    with patch("agent_market.wq_brain.agent_runner._run_cli_with_group_timeout") as mock_run:
        mock_run.return_value = 0
        run_agent(config)

    # Inspect the CLI invocation passed to the group-timeout runner.
    cmd = mock_run.call_args_list[-1][0][0]
    assert cmd[0] == "hermes"
    assert cmd[1] == "chat"
    assert "--toolsets" in cmd and "terminal,file" in cmd
    assert "--max-turns" in cmd and "20" in cmd
    assert "-m" in cmd and "MiniMax" in cmd
    assert "--provider" in cmd and "openrouter" in cmd
    assert "--yolo" in cmd


def test_run_agent_sets_compact_local_sim_limits(isolated_artifacts):
    config = AgentConfig(
        tag="t1", max_turns=12, model="MiniMax", cli="opencode",
        timeout_sec=10.0,
    )
    with patch("agent_market.wq_brain.agent_runner._run_cli_with_group_timeout") as mock_run:
        mock_run.return_value = 0
        run_agent(config)

    env = mock_run.call_args.kwargs["env"]
    assert env["WQB_AGENT_LOCAL_SIM_LIMIT"] == "1"
    assert env["WQB_AGENT_LOCAL_SIM_MAX_CONCURRENT"] == "1"
    assert env["WQB_AGENT_LOCAL_SIM_TIMEOUT_SEC"] == "360"
    assert env["WQB_AGENT_REQUIRE_LOCAL_SIM"] == "1"


def test_run_agent_handles_timeout(isolated_artifacts):
    config = AgentConfig(tag="t1", max_turns=5, timeout_sec=1.0,
                         cli="opencode", model="m1")
    with patch("agent_market.wq_brain.agent_runner._run_cli_with_group_timeout") as mock_run:
        mock_run.return_value = -1
        summary = run_agent(config)
    assert summary["agent_returncode"] == -1


def test_run_cli_timeout_writes_marker(tmp_path):
    from agent_market.wq_brain.agent_runner import _run_cli_with_group_timeout
    log_path = tmp_path / "agent.log"
    rc = _run_cli_with_group_timeout(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=tmp_path,
        env={},
        log_path=log_path,
        timeout_sec=0.1,
    )
    assert rc == -1
    assert "terminating process group" in log_path.read_text(encoding="utf-8")


def test_llm_cli_env_accepts_openai_api_base_alias(monkeypatch):
    """README §195/209 documents OPENAI_API_BASE — agent_runner must
    honor it (was previously only OPENAI_BASE_URL / LLM_BASE_URL)."""
    from agent_market.wq_brain import agent_runner as ar
    monkeypatch.setattr(ar, "_load_dotenv_into", lambda *a, **kw: None)
    for k in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "LLM_BASE_URL",
              "OPENAI_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "http://my-llm-host:38889")
    env = ar._llm_cli_env()
    assert env["OPENAI_BASE_URL"].endswith("/v1")
    assert env["OPENAI_API_BASE"].endswith("/v1")
    assert env["LLM_BASE_URL"].endswith("/v1")


def test_llm_cli_env_idempotent_when_already_v1(monkeypatch):
    from agent_market.wq_brain import agent_runner as ar
    monkeypatch.setattr(ar, "_load_dotenv_into", lambda *a, **kw: None)
    for k in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "LLM_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    env = ar._llm_cli_env()
    assert env["OPENAI_BASE_URL"] == "https://api.openai.com/v1"
    assert not env["OPENAI_BASE_URL"].endswith("/v1/v1")


def test_run_agent_resolves_openai_model_alias(isolated_artifacts, monkeypatch):
    """OPENAI_MODEL (README §48) should be honored when config.model empty."""
    from agent_market.wq_brain import agent_runner as ar
    monkeypatch.setattr(ar, "_load_dotenv_into", lambda *a, **kw: None)
    for k in ("LLM_MODEL", "OPENAI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "custom/gpt-5.2")

    config = AgentConfig(tag="modeltag", max_turns=2, timeout_sec=5.0,
                         cli="opencode", model="")

    captured = {}
    def _fake(cmd, **kwargs):
        captured["cmd"] = cmd
        return 0
    with patch("agent_market.wq_brain.agent_runner._run_cli_with_group_timeout",
               side_effect=_fake):
        run_agent(config)
    assert "custom/gpt-5.2" in captured["cmd"]


def test_run_agent_resolves_opencode_model_alias(isolated_artifacts, monkeypatch):
    """OPENCODE_MODEL — referenced in _build_opencode_cmd error message —
    must actually be honored by the resolution path."""
    from agent_market.wq_brain import agent_runner as ar
    monkeypatch.setattr(ar, "_load_dotenv_into", lambda *a, **kw: None)
    for k in ("LLM_MODEL", "OPENAI_MODEL", "OPENCODE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENCODE_MODEL", "custom/oc-model")

    config = AgentConfig(tag="octag", max_turns=2, timeout_sec=5.0,
                         cli="opencode", model="")

    captured = {}
    def _fake(cmd, **kwargs):
        captured["cmd"] = cmd
        return 0
    with patch("agent_market.wq_brain.agent_runner._run_cli_with_group_timeout",
               side_effect=_fake):
        run_agent(config)
    assert "custom/oc-model" in captured["cmd"]


def test_run_agent_persists_resolved_model_in_config_json(
    isolated_artifacts, monkeypatch,
):
    """config.json should reflect the model that ACTUALLY ran, not the
    pre-resolution empty-string default."""
    from agent_market.wq_brain import agent_runner as ar
    monkeypatch.setattr(ar, "_load_dotenv_into", lambda *a, **kw: None)
    for k in ("LLM_MODEL", "OPENAI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "custom/gpt-5.2")

    config = AgentConfig(tag="cfgtag", max_turns=2, timeout_sec=5.0,
                         cli="opencode", model="")

    with patch("agent_market.wq_brain.agent_runner._run_cli_with_group_timeout",
               return_value=0):
        summary = run_agent(config)
    cfg_path = Path(summary["run_dir"]) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    assert cfg["model"] == "custom/gpt-5.2"


def test_classify_agent_failure_recognises_llm_quota(tmp_path):
    from agent_market.wq_brain.agent_runner import _classify_agent_failure
    log = tmp_path / "agent.log"
    log.write_text(
        "starting agent…\nERROR: 用户额度不足, 剩余额度: $0.0000\n",
        encoding="utf-8",
    )
    cls = _classify_agent_failure(log)
    assert cls["kind"] == "llm_quota"
    assert "credit" in cls["hint"].lower() or "quota" in cls["hint"].lower()


def test_classify_agent_failure_recognises_wq_auth(tmp_path):
    from agent_market.wq_brain.agent_runner import _classify_agent_failure
    log = tmp_path / "agent.log"
    log.write_text("HTTP 401 Unauthorized\n", encoding="utf-8")
    cls = _classify_agent_failure(log)
    assert cls["kind"] == "wq_auth"


def test_classify_agent_failure_recognises_network(tmp_path):
    from agent_market.wq_brain.agent_runner import _classify_agent_failure
    log = tmp_path / "agent.log"
    log.write_text("urllib3: connection refused (err 111)\n", encoding="utf-8")
    cls = _classify_agent_failure(log)
    assert cls["kind"] == "network"


def test_classify_agent_failure_recognises_runner_timeout(tmp_path):
    from agent_market.wq_brain.agent_runner import _classify_agent_failure
    log = tmp_path / "agent.log"
    log.write_text(
        "[agent_runner] timeout after 1800s; terminating process group\n",
        encoding="utf-8",
    )
    cls = _classify_agent_failure(log)
    assert cls["kind"] == "timeout"


def test_classify_agent_failure_unknown_when_no_pattern(tmp_path):
    from agent_market.wq_brain.agent_runner import _classify_agent_failure
    log = tmp_path / "agent.log"
    log.write_text("everything completed normally — exit 1 anyway\n",
                   encoding="utf-8")
    cls = _classify_agent_failure(log)
    assert cls["kind"] == "unknown"


def test_classify_agent_failure_missing_log(tmp_path):
    from agent_market.wq_brain.agent_runner import _classify_agent_failure
    cls = _classify_agent_failure(tmp_path / "no_such_log.log")
    assert cls["kind"] == "unknown"
    assert "no log" in cls["hint"]


def test_run_agent_writes_checkpoint_sidecar(isolated_artifacts):
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.tried_log import read_checkpoint

    config = AgentConfig(tag="ckpt_tag", max_turns=2, timeout_sec=5.0,
                         cli="opencode", model="m1")

    with patch("agent_market.wq_brain.agent_runner._run_cli_with_group_timeout", return_value=0):
        run_agent(config)

    ck = read_checkpoint(tried_exprs_path("ckpt_tag"))
    assert ck is not None
    assert ck["session_id"].startswith("wqbrain_agent_ckpt_tag_")
    assert ck["extra"]["tag"] == "ckpt_tag"
    assert ck["extra"]["rc"] == 0
    assert ck["extra"]["failure_kind"] is None


def test_run_agent_failure_kind_recorded_on_nonzero_exit(isolated_artifacts):
    """Non-zero rc + log containing a quota pattern should land in summary['failure']."""
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.tried_log import read_checkpoint

    config = AgentConfig(tag="fail_tag", max_turns=2, timeout_sec=5.0,
                         cli="opencode", model="m1")

    def _fake_run(*args, **kwargs):
        log_path = kwargs.get("log_path")
        if log_path is not None:
            Path(log_path).write_text("FATAL: 用户额度不足\n", encoding="utf-8")
        return 5

    with patch("agent_market.wq_brain.agent_runner._run_cli_with_group_timeout",
               side_effect=_fake_run):
        summary = run_agent(config)

    assert summary["agent_returncode"] == 5
    assert summary["failure"]["kind"] == "llm_quota"
    ck = read_checkpoint(tried_exprs_path("fail_tag"))
    assert ck["extra"]["failure_kind"] == "llm_quota"


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
    """When all 9 canonical families present, hint asks for lowest-count next."""
    full = list({"ts_corr_pv", "intraday_range", "vwap_dev", "volume_rank",
                 "open_gap", "humped", "multi_signal", "sector_relative",
                 "fundamental_ratio"})
    hint = _family_diversity_hint(full)
    assert "LOWEST count" in hint


def test_tried_family_concentration_hint_fires_at_default_threshold():
    """6/10 of one family — exactly at default threshold (0.6), should fire."""
    records = [
        {"ts": i, "expr": "rank(group_zscore(close, sector))"} for i in range(6)
    ] + [
        {"ts": 100 + i, "expr": "rank(close)"} for i in range(4)
    ]
    hint = _tried_family_concentration_hint(records)
    assert "STUCK IN" in hint
    assert "sector_relative" in hint
    assert "6/10" in hint


def test_tried_family_concentration_hint_below_threshold_silent():
    """5/10 of one family — under default threshold, should NOT fire."""
    records = [
        {"ts": i, "expr": "rank(group_zscore(close, sector))"} for i in range(5)
    ] + [
        {"ts": 100 + i, "expr": "rank(close)"} for i in range(5)
    ]
    assert _tried_family_concentration_hint(records) == ""


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


def test_operator_skeleton_drops_fields_and_numbers():
    """Same operator multiset → same skeleton, regardless of fields/numbers."""
    sk1 = _operator_skeleton("rank(sales/assets) * ts_decay_linear(rank(-ts_delta(close, 3)/close), 20)")
    sk2 = _operator_skeleton("rank(debt/equity) * ts_decay_linear(rank(-ts_delta(close, 7)/close), 30)")
    sk3 = _operator_skeleton("rank(revenue/assets) * ts_decay_linear(rank(-ts_delta(vwap, 5)/vwap), 10)")
    assert sk1 == sk2 == sk3
    assert "rank" in sk1 and "ts_decay_linear" in sk1 and "ts_delta" in sk1


def test_operator_skeleton_distinguishes_different_stacks():
    """Adding/dropping an operator → different skeleton."""
    sk1 = _operator_skeleton("rank(sales/assets)")
    sk2 = _operator_skeleton("rank(group_zscore(sales/assets, sector))")
    assert sk1 != sk2


def test_operator_skeleton_empty_input():
    assert _operator_skeleton("") == ""
    assert _operator_skeleton("close") == ""  # no operators, just a field


def test_tried_skeleton_hint_fires_at_threshold():
    """5/10 attempts share `rank+ts_decay_linear+ts_delta` skeleton → fires at 0.5."""
    same_skel = [
        {"ts": i, "expr": f"rank({fld}/assets) * ts_decay_linear(rank(-ts_delta(close, 3)/close), 20)"}
        for i, fld in enumerate(["sales", "revenue", "ebit", "debt", "fcf"])
    ]
    diverse = [
        {"ts": 100 + i, "expr": e} for i, e in enumerate([
            "rank(close)",
            "rank(group_zscore(volume, sector))",
            "hump(rank(close))",
            "rank((high - low)/close)",
            "rank(open - ts_delay(close, 1))",
        ])
    ]
    hint = _tried_skeleton_concentration_hint(same_skel + diverse)
    assert "STUCK IN OPERATOR SKELETON" in hint
    assert "5/10" in hint or "5" in hint  # show concentration ratio


def test_tried_skeleton_hint_silent_when_diverse():
    records = [
        {"ts": i, "expr": e} for i, e in enumerate([
            "rank(close)",
            "rank(group_zscore(volume, sector))",
            "hump(rank(close))",
            "rank((high - low)/close)",
            "rank(open - ts_delay(close, 1))",
            "rank(sales/assets)",
            "ts_decay_linear(rank(close), 5)",
            "rank(ts_corr(close, volume, 20))",
            "rank(vwap/close)",
            "rank(ts_rank(close, 252))",
        ])
    ]
    assert _tried_skeleton_concentration_hint(records) == ""


def test_tried_skeleton_hint_too_few_records():
    records = [{"ts": i, "expr": "rank(close)"} for i in range(5)]
    assert _tried_skeleton_concentration_hint(records) == ""


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
