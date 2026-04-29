from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent_market import paths as repo_paths
from agent_market.factor_lab.strategy_loop import (
    PHASE_BACKTEST,
    PHASE_COMPLETE,
    LOOP_COMPLETED,
    VERIFICATION_FAILED,
    VERIFICATION_INCONCLUSIVE,
    VERIFICATION_PASSED,
    StrategyLoopConfig,
    StrategyLoopRunner,
    StrategyLoopState,
    build_iteration_manifest,
    build_pareto_pool,
    doctor_strategy_loop_run,
    _hermes_cli_env,
    _hermes_model,
    _opencode_cli_env,
    _rank_kwargs,
    iteration_dir,
    leaderboard_path,
    load_checkpoint,
    parse_lookahead_csv,
    parse_recursive_output,
    rank_profile_signature,
    prepare_context,
    promote_candidate,
    render_agent_prompt,
    save_checkpoint,
    score_backtest_result,
    score_strategy_loop_backtest,
    score_triple_holdout_backtest,
    scaled_gate_values,
    strategy_loop_registry_path,
    strategy_loop_retention_tier,
    validate_candidate,
    write_strategy_loop_registry_entry,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_strategy_loop_checkpoint_roundtrip() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit", run_id="unit_checkpoint", max_iterations=2)
    state = StrategyLoopState(run_id=cfg.run_id, iteration=2, phase=PHASE_BACKTEST)
    state.best_score = 12.5
    state.candidate_paths.append("artifacts/factor_strategy_loop/unit/iter_01/candidate.json")

    path = save_checkpoint(cfg, state)
    loaded_cfg, loaded_state = load_checkpoint("unit_checkpoint")

    assert path.exists()
    assert loaded_cfg.tag == "unit"
    assert loaded_state.iteration == 2
    assert loaded_state.phase == PHASE_BACKTEST
    assert loaded_state.best_score == 12.5
    assert loaded_state.candidate_paths == state.candidate_paths


def test_candidate_schema_rejects_out_of_bounds_leverage(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    _write_json(
        candidate,
        {
            "candidate_type": "rank_profile",
            "name": "too_hot",
            "params": {"top_k": 2, "leverage_cap": 25.0},
        },
    )

    with pytest.raises(ValueError, match="leverage_cap"):
        validate_candidate(candidate)


def test_candidate_schema_rejects_strategy_path_escape(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    _write_json(
        candidate,
        {
            "candidate_type": "freqtrade_strategy",
            "name": "escape",
            "strategy_path": "../strategy.py",
        },
    )

    with pytest.raises(ValueError, match="escapes"):
        validate_candidate(candidate)


def test_candidate_schema_accepts_freqtrade_strategy(tmp_path: Path) -> None:
    (tmp_path / "strategy.py").write_text(
        "\n".join(
            [
                "import os",
                "import pandas as pd",
                "from freqtrade.strategy import IStrategy",
                "",
                "class UnitRankStrategy(IStrategy):",
                "    def _signals(self):",
                "        path = os.environ.get('RP_SIGNAL_DIR') or os.environ.get('RP_TAG', 'tag')",
                "        df = pd.read_feather(path)",
                "        return df[['rp_target_weight', 'rp_side']]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.json"
    _write_json(
        candidate,
        {
            "candidate_type": "freqtrade_strategy",
            "name": "unit_strategy",
            "strategy_path": "strategy.py",
            "rank_profile": {"top_k": 2, "gross_cap": 2.0, "leverage_cap": 5.0},
        },
    )

    normalized = validate_candidate(candidate)

    assert normalized["candidate_type"] == "freqtrade_strategy"
    assert normalized["strategy_validation"]["istrategy_classes"] == ["UnitRankStrategy"]
    assert normalized["rank_profile"]["leverage_cap"] == 5.0


def test_candidate_schema_rejects_hardcoded_strategy_paths(tmp_path: Path) -> None:
    (tmp_path / "strategy.py").write_text(
        "\n".join(
            [
                "import pandas as pd",
                "from freqtrade.strategy import IStrategy",
                "",
                "class BadPathStrategy(IStrategy):",
                "    def _signals(self):",
                "        df = pd.read_feather('/Users/shatianming/Downloads/Agent_market/artifacts/x.feather')",
                "        return df[['rp_target_weight', 'rp_side']]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.json"
    _write_json(
        candidate,
        {
            "candidate_type": "freqtrade_strategy",
            "name": "bad_path",
            "strategy_path": "strategy.py",
        },
    )

    with pytest.raises(ValueError, match="hard-code absolute"):
        validate_candidate(candidate)


def test_prompt_can_force_freqtrade_candidate_type(tmp_path: Path) -> None:
    prompt = render_agent_prompt(tmp_path / "context" / "prepare.json", candidate_type="freqtrade_strategy")

    assert "must create a `freqtrade_strategy` candidate" in prompt
    assert "Do not put signal-column mappings" in prompt
    assert "RP_SIGNAL_DIR" in prompt
    assert "loop_memory.best_candidate" in prompt
    assert "avoid_repeating_rank_profiles" in prompt
    assert "rp_target_weight` used in stake/position sizing logic" in prompt


def test_config_accepts_cli_opencode_mode() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit", agent="opencode", opencode_mode="cli", candidate_type="freqtrade_strategy")

    assert cfg.agent == "opencode"
    assert cfg.opencode_mode == "cli"
    assert cfg.candidate_type == "freqtrade_strategy"


def test_config_defaults_to_rank_profile_two_stage() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit")

    assert cfg.agent == "hermes"
    assert cfg.candidate_type == "rank_profile"
    assert cfg.eval_mode == "two_stage"
    assert cfg.hermes_toolsets == "terminal,file"
    assert cfg.hermes_reasoning_effort == ""


def test_config_accepts_hermes_reasoning_effort() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit", agent="hermes", hermes_reasoning_effort="xhigh")

    assert cfg.hermes_reasoning_effort == "xhigh"


def test_strategy_loop_doctor_accepts_complete_formal_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    run_id = "doctor_formal_ok"
    root = repo_paths.artifacts_root() / "factor_strategy_loop" / run_id
    root.mkdir(parents=True)
    cfg = StrategyLoopConfig.from_args(
        tag="unit",
        run_id=run_id,
        validation_protocol="triple_holdout",
        verify_policy="pareto",
        promote_policy="final",
    )
    _write_json(
        root / "checkpoint.json",
        {
            "config": cfg.__dict__,
            "state": {"run_id": run_id, "iteration": 1},
        },
    )
    _write_json(root / "manifest.json", {"cli_args": cfg.__dict__})
    _write_json(root / "leaderboard.json", {"rows": [{"iteration": 1, "promotion_eligible": False}]})
    _write_json(root / "pareto_pool.json", {"finalists": []})
    _write_json(root / "final_promotion.json", {"promoted": False})

    deep_root = repo_paths.artifacts_root() / "strategy_deepresearch" / run_id
    _write_json(deep_root / "context.json", {"run_id": run_id})
    _write_json(deep_root / "sources.json", {"sources": []})
    selected = {
        "blind_final": True,
        "promotion_eligible": True,
        "verification_status": VERIFICATION_PASSED,
    }
    _write_json(
        root / "final_blind_status.json",
        {
            "selected": selected,
            "promotion": {"promoted": False},
            "deepresearch": {
                "artifacts": {
                    "context": f"artifacts/strategy_deepresearch/{run_id}/context.json",
                    "sources": f"artifacts/strategy_deepresearch/{run_id}/sources.json",
                }
            },
        },
    )
    blind_dir = root / "blind_1"
    _write_json(blind_dir / "verification.json", {"status": VERIFICATION_PASSED})
    _write_json(
        blind_dir / "manifest.json",
        {"artifact_refs": {"verification.json": {"path": "x", "sha256": "abc", "bytes": 1}}},
    )

    result = doctor_strategy_loop_run(run_id)

    assert result["ok"] is True
    assert result["summary"]["verification_counts"][VERIFICATION_PASSED] == 1
    assert result["findings"] == []


def test_strategy_loop_registry_records_retention_tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    run_id = "registry_unit"
    cfg = StrategyLoopConfig.from_args(
        tag="unit",
        run_id=run_id,
        validation_protocol="triple_holdout",
        verify_policy="pareto",
        promote_policy="final",
    )
    state = StrategyLoopState(run_id=run_id, status=LOOP_COMPLETED)
    root = repo_paths.artifacts_root() / "factor_strategy_loop" / run_id
    _write_json(root / "manifest.json", {"run_id": run_id})
    _write_json(root / "checkpoint.json", {"run_id": run_id})
    _write_json(
        root / "final_blind_status.json",
        {"selected": {"promotion_eligible": True, "verification_status": VERIFICATION_PASSED}},
    )

    path = write_strategy_loop_registry_entry(
        cfg,
        state,
        final_promotion={"promoted": False, "reason": "unit"},
    )

    assert path == strategy_loop_registry_path()
    assert strategy_loop_retention_tier(run_id) == "keep_blind_passed_not_promoted"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["run_id"] == run_id
    assert row["retention_tier"] == "keep_blind_passed_not_promoted"


def test_candidate_schema_accepts_baseline_rank_profile_controls(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    _write_json(
        candidate,
        {
            "candidate_type": "rank_profile",
            "name": "baseline_replay",
            "rank_profile": {
                "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
                "recompute_corr": False,
                "short_max_mom_24h": 0.04,
                "short_max_mom_72h": 0.10,
                "max_entry_atr_pct": 0.05,
                "pair_edge_leverage": "false",
            },
        },
    )

    normalized = validate_candidate(candidate)

    profile = normalized["rank_profile"]
    assert profile["candidate_state"].endswith("state_0149.json")
    assert profile["recompute_corr"] is False
    assert profile["short_max_mom_24h"] == 0.04
    assert profile["pair_edge_leverage"] is False


def test_rank_kwargs_inherits_optimized_baseline_controls() -> None:
    tag = "unit_rank_kwargs_baseline"
    baseline_path = repo_paths.artifacts_root() / "rank_portfolio" / tag / "optimized_profile.json"
    _write_json(
        baseline_path,
        {
            "tag": tag,
            "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
            "selection": {"recompute_corr": False, "n": 50},
            "risk": {
                "top_k": 2,
                "gross_cap": 2.0,
                "net_cap": 2.0,
                "side_mode": "short",
                "score_threshold": 1.5,
                "rebalance_hours": 8,
                "short_max_mom_24h": 0.04,
                "short_max_mom_72h": 0.10,
                "max_entry_atr_pct": 0.05,
            },
        },
    )
    cfg = StrategyLoopConfig.from_args(tag=tag, baseline_profile=str(baseline_path))

    kwargs = _rank_kwargs({"rebalance_hours": 12}, cfg, candidate_state=None, tag="effective")

    assert kwargs["tag"] == "effective"
    assert kwargs["candidate_state"].endswith("state_0149.json")
    assert kwargs["recompute_corr"] is False
    assert kwargs["rebalance_hours"] == 12
    assert kwargs["short_max_mom_24h"] == 0.04
    assert kwargs["max_entry_atr_pct"] == 0.05


def test_runner_seeds_initial_optimized_baseline_candidate(tmp_path: Path) -> None:
    baseline_path = tmp_path / "optimized_profile.json"
    _write_json(
        baseline_path,
        {
            "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
            "selection": {"recompute_corr": False, "n": 50},
            "risk": {
                "top_k": 2,
                "gross_cap": 2.0,
                "net_cap": 2.0,
                "side_mode": "short",
                "score_threshold": 1.5,
                "short_max_mom_24h": 0.04,
            },
        },
    )
    cfg = StrategyLoopConfig.from_args(
        tag="unit_seed_baseline",
        run_id="unit_seed_baseline_run",
        baseline_profile=str(baseline_path),
        candidate_type="rank_profile",
    )
    runner = StrategyLoopRunner(cfg)
    idir = tmp_path / "iter_01"
    idir.mkdir()

    runner._code_gen(idir)

    candidate = validate_candidate(idir / "candidate.json")
    assert candidate["name"] == "optimized_baseline_replay"
    assert candidate["rank_profile"]["candidate_state"].endswith("state_0149.json")
    assert candidate["rank_profile"]["recompute_corr"] is False
    assert candidate["rank_profile"]["min_abs_score_z"] == 1.5
    assert runner.state.candidate_paths


def test_opencode_cli_env_uses_project_config_and_bypasses_local_proxy() -> None:
    env = _opencode_cli_env(
        {
            "http_proxy": "http://127.0.0.1:1097",
            "https_proxy": "http://127.0.0.1:1097",
            "OPENAI_API_BASE": "http://127.0.0.1:8317",
            "OPENAI_API_KEY": "_",
            "NO_PROXY": "example.com",
        },
        load_dotenv=False,
    )

    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8317/v1"
    assert env["OPENCODE_CONFIG"].endswith(".opencode.json")
    assert "example.com" in env["NO_PROXY"]
    assert "127.0.0.1" in env["NO_PROXY"]
    assert "localhost" in env["no_proxy"]


def test_hermes_cli_env_uses_openai_compatible_settings_and_bypasses_local_proxy() -> None:
    env = _hermes_cli_env(
        {
            "http_proxy": "http://127.0.0.1:1097",
            "OPENAI_API_BASE": "http://127.0.0.1:8317",
            "LLM_API_KEY": "local-key",
            "OPENAI_MODEL": "gpt-5.5",
        },
        load_dotenv=False,
    )

    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8317/v1"
    assert env["OPENAI_API_KEY"] == "local-key"
    assert "127.0.0.1" in env["NO_PROXY"]
    assert _hermes_model("", env) == "gpt-5.5"
    assert _hermes_model("custom/gpt-5.2", env) == "gpt-5.2"


def test_runner_uses_hermes_cli_for_codegen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = StrategyLoopConfig.from_args(
        tag="unit_hermes",
        run_id="unit_hermes_run",
        agent="hermes",
        model="gpt-5.5",
        hermes_provider="openai-codex",
        hermes_toolsets="terminal,file",
        max_turns=7,
    )
    runner = StrategyLoopRunner(cfg)
    runner.state.iteration = 2
    captured: dict[str, object] = {}

    monkeypatch.setattr("agent_market.factor_lab.strategy_loop.shutil.which", lambda name: "/usr/bin/hermes" if name == "hermes" else None)

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        cwd = Path(str(kwargs["cwd"]))
        _write_json(
            cwd / "candidate.json",
            {
                "candidate_type": "rank_profile",
                "name": "hermes_candidate",
                "rank_profile": {"top_k": 2, "rebalance_hours": 8},
            },
        )
        (cwd / "analysis.md").write_text("Hermes generated candidate.\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n")

    monkeypatch.setattr("agent_market.factor_lab.strategy_loop.subprocess.run", fake_run)

    runner._code_gen(tmp_path)

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:3] == ["hermes", "chat", "-Q"]
    assert ["-m", "gpt-5.5"] == cmd[cmd.index("-m"):cmd.index("-m") + 2]
    assert ["--provider", "openai-codex"] == cmd[cmd.index("--provider"):cmd.index("--provider") + 2]
    assert ["--max-turns", "7"] == cmd[cmd.index("--max-turns"):cmd.index("--max-turns") + 2]
    assert validate_candidate(tmp_path / "candidate.json")["name"] == "hermes_candidate"
    assert runner.state.candidate_paths


def test_runner_applies_hermes_reasoning_effort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = StrategyLoopConfig.from_args(
        tag="unit_hermes_effort",
        run_id="unit_hermes_effort_run",
        agent="hermes",
        model="gpt-5.5",
        hermes_reasoning_effort="xhigh",
    )
    runner = StrategyLoopRunner(cfg)
    runner.state.iteration = 2
    commands: list[list[str]] = []
    envs: list[dict[str, str]] = []

    monkeypatch.setattr("agent_market.factor_lab.strategy_loop.shutil.which", lambda name: "/usr/bin/hermes" if name == "hermes" else None)

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        envs.append(dict(kwargs["env"]))
        if cmd[:3] == ["hermes", "chat", "-Q"]:
            cwd = Path(str(kwargs["cwd"]))
            _write_json(
                cwd / "candidate.json",
                {
                    "candidate_type": "rank_profile",
                    "name": "hermes_effort_candidate",
                    "rank_profile": {"top_k": 2, "rebalance_hours": 8},
                },
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n")

    monkeypatch.setattr("agent_market.factor_lab.strategy_loop.subprocess.run", fake_run)

    runner._code_gen(tmp_path)

    assert commands[0] == ["hermes", "config", "set", "agent.reasoning_effort", "xhigh"]
    assert commands[1][:3] == ["hermes", "chat", "-Q"]
    assert envs[0]["HERMES_HOME"].endswith("factor_strategy_loop/unit_hermes_effort_run/hermes_home")
    assert envs[1]["HERMES_HOME"] == envs[0]["HERMES_HOME"]
    response = (tmp_path / "agent_response.txt").read_text(encoding="utf-8")
    assert "Hermes HERMES_HOME=" in response
    assert "Hermes reasoning_effort=xhigh" in response


def test_runner_records_iteration_failure_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit_failure", run_id="unit_failure_run", max_iterations=2)
    runner = StrategyLoopRunner(cfg)
    codegen_calls: list[int] = []

    def fake_prepare(self: StrategyLoopRunner, idir: Path) -> None:
        return None

    def fake_code_gen(self: StrategyLoopRunner, idir: Path) -> None:
        codegen_calls.append(self.state.iteration)
        if self.state.iteration == 1:
            raise ValueError("invalid generated strategy")
        self.state.phase = PHASE_COMPLETE

    monkeypatch.setattr(StrategyLoopRunner, "_prepare", fake_prepare)
    monkeypatch.setattr(StrategyLoopRunner, "_code_gen", fake_code_gen)

    runner.run()

    assert codegen_calls == [1, 2]
    payload = json.loads(leaderboard_path(cfg.run_id).read_text(encoding="utf-8"))
    failed = payload["rows"][0]
    assert failed["iteration"] == 1
    assert failed["constraints_ok"] is False
    assert "invalid generated strategy" in failed["violations"][0]


def test_prepare_context_includes_loop_memory() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit_memory", run_id="unit_memory_run", max_iterations=3)
    row = {
        "run_id": cfg.run_id,
        "iteration": 1,
        "candidate": {
            "candidate_type": "freqtrade_strategy",
            "name": "memory_candidate",
            "rank_profile": {"top_k": 2, "rebalance_hours": 4},
        },
        "score": -12.5,
        "constraints_ok": False,
        "metrics": {"profit_pct": -1.0, "trades": 100, "profit_over_max_drawdown": -0.2},
        "violations": ["profit_over_max_drawdown=-0.2 < 1.2"],
    }
    state = StrategyLoopState(run_id=cfg.run_id, iteration=2, best_candidate=row, best_score=-12.5, score_history=[row])
    save_checkpoint(cfg, state)
    err_path = repo_paths.artifacts_root() / "factor_strategy_loop" / cfg.run_id / "iter_01" / "error.json"
    _write_json(err_path, {"phase": "CODE_GEN", "error_type": "ValueError", "message": "bad strategy"})

    context = prepare_context(cfg, cfg.run_id, 2)

    memory = context["loop_memory"]
    assert memory["best_candidate"]["name"] == "memory_candidate"
    assert memory["recent_score_history"][0]["metrics"]["trades"] == 100
    assert memory["avoid_repeating_rank_profiles"] == [{"top_k": 2, "rebalance_hours": 4}]
    assert memory["previous_failure"]["message"] == "bad strategy"


def test_prepare_context_promotes_optimized_profile_to_first_class_baseline() -> None:
    tag = "unit_context_baseline"
    baseline_path = repo_paths.artifacts_root() / "rank_portfolio" / tag / "optimized_profile.json"
    _write_json(
        baseline_path,
        {
            "tag": tag,
            "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
            "selection": {"recompute_corr": False, "n": 50},
            "risk": {
                "top_k": 2,
                "score_threshold": 1.5,
                "short_max_mom_24h": 0.04,
                "short_max_mom_72h": 0.10,
                "max_entry_atr_pct": 0.05,
            },
            "research_backtest": {
                "total_return_pct": 35.26,
                "max_drawdown_pct": 14.67,
                "profit_over_max_drawdown": 2.40,
                "trades": 119,
            },
            "freqtrade_backtest": {
                "total_profit_pct": 39.78,
                "max_account_underwater_pct": 16.39,
                "trades": 139,
            },
        },
    )
    _write_json(
        repo_paths.artifacts_root() / "rank_portfolio" / tag / "backtest.json",
        {"total_return_pct": -12.0, "candidate_source": "artifacts/factor_lab/mining/unit/state_0241.json"},
    )
    cfg = StrategyLoopConfig.from_args(tag=tag, run_id="unit_context_baseline_run", baseline_profile=str(baseline_path))

    context = prepare_context(cfg, cfg.run_id, 1)

    baseline = context["optimized_baseline"]
    assert baseline["available"] is True
    assert baseline["rank_profile"]["recompute_corr"] is False
    assert baseline["rank_profile"]["candidate_state"].endswith("state_0149.json")
    assert baseline["latest_delta"]["research_metrics"]["total_return_pct"]["baseline"] == 35.26
    assert context["baseline_search_policy"]["first_iteration"].startswith("reproduce")


def test_scorer_enforces_hard_gates_and_orders_candidates() -> None:
    strong = score_backtest_result(
        {
            "total_return_pct": 38.0,
            "max_drawdown_pct": 12.0,
            "profit_over_max_drawdown": 3.17,
            "trades": 140,
            "simulated_liquidations": 0,
            "liquidation_rejects": 0,
            "avg_turnover": 0.8,
        }
    )
    weak = score_backtest_result(
        {
            "total_return_pct": 30.0,
            "max_drawdown_pct": 31.0,
            "profit_over_max_drawdown": 0.97,
            "trades": 40,
            "simulated_liquidations": 1,
            "liquidation_rejects": 2,
            "avg_turnover": 4.0,
        }
    )

    assert strong["constraints_ok"] is True
    assert weak["constraints_ok"] is False
    assert "simulated_liquidations" in ";".join(weak["violations"])
    assert strong["score"] > weak["score"]


def test_scorer_prefers_positive_profit_high_drawdown_over_low_drawdown_loss() -> None:
    positive = score_backtest_result(
        {
            "total_return_pct": 12.0,
            "max_drawdown_pct": 24.0,
            "profit_over_max_drawdown": 0.5,
            "trades": 120,
        }
    )
    loss = score_backtest_result(
        {
            "total_return_pct": -3.0,
            "max_drawdown_pct": 8.0,
            "profit_over_max_drawdown": -0.375,
            "trades": 120,
        }
    )

    assert positive["constraints_ok"] is False
    assert loss["constraints_ok"] is False
    assert positive["score"] > loss["score"]


def test_composite_score_uses_freqtrade_as_primary_sort_key() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit_composite", eval_mode="freqtrade", score_mode="composite")
    research_pass = {
        "total_return_pct": 30.0,
        "max_drawdown_pct": 15.0,
        "profit_over_max_drawdown": 2.0,
        "trades": 140,
        "simulated_liquidations": 0,
        "liquidation_rejects": 0,
    }
    lower_research_higher_ft = {
        **research_pass,
        "total_return_pct": 25.0,
        "profit_over_max_drawdown": 1.67,
        "freqtrade_backtest": {
            "ok": True,
            "metrics": {"profit_pct": 45.0, "max_drawdown_pct": 12.0, "profit_over_max_drawdown": 3.75, "trades": 140},
        },
    }
    higher_research_lower_ft = {
        **research_pass,
        "total_return_pct": 50.0,
        "profit_over_max_drawdown": 3.33,
        "freqtrade_backtest": {
            "ok": True,
            "metrics": {"profit_pct": 25.0, "max_drawdown_pct": 12.0, "profit_over_max_drawdown": 2.08, "trades": 140},
        },
    }

    high_ft = score_strategy_loop_backtest(lower_research_higher_ft, cfg)
    low_ft = score_strategy_loop_backtest(higher_research_lower_ft, cfg)

    assert high_ft["constraints_ok"] is True
    assert high_ft["score_components"]["freqtrade_score"] > low_ft["score_components"]["freqtrade_score"]
    assert high_ft["score"] > low_ft["score"]
    assert "Freqtrade quality is primary" in high_ft["score_components"]["selection_reason"]


def test_rank_profile_signature_quantizes_and_sorts_pairs() -> None:
    first = {
        "top_k": 2,
        "risk_per_trade": 0.01782417761,
        "short_max_mom_24h": 0.04049,
        "max_entry_atr_pct": 0.05049,
        "exclude_pairs": ["ETH/USDT:USDT", "btc-usdt"],
    }
    second = {
        "top_k": 2,
        "risk_per_trade": 0.01782417762,
        "short_max_mom_24h": 0.0404,
        "max_entry_atr_pct": 0.0504,
        "exclude_pairs": ["BTC/USDT", "ETH/USDT"],
    }

    assert rank_profile_signature(first) == rank_profile_signature(second)


def test_runner_rejects_near_duplicate_rank_profile() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit_dup", run_id="unit_dup_run")
    runner = StrategyLoopRunner(cfg)
    profile = {
        "top_k": 2,
        "risk_per_trade": 0.01782417761,
        "short_max_mom_24h": 0.04049,
        "exclude_pairs": ["ETH/USDT:USDT", "btc-usdt"],
    }
    runner.state.score_history = [
        {
            "run_id": cfg.run_id,
            "iteration": 1,
            "candidate": {"candidate_type": "rank_profile", "name": "prior", "rank_profile": profile},
            "parameter_signature": rank_profile_signature(profile),
        }
    ]
    duplicate = {
        "candidate_type": "rank_profile",
        "name": "tiny_precision_nudge",
        "rank_profile": {
            "top_k": 2,
            "risk_per_trade": 0.01782417762,
            "short_max_mom_24h": 0.0404,
            "exclude_pairs": ["BTC/USDT", "ETH/USDT"],
        },
    }

    with pytest.raises(ValueError, match="near-duplicate rank_profile signature"):
        runner._validate_unique_candidate(duplicate)


def test_iteration_failure_violations_are_truncated() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit_failure_truncate", run_id="unit_failure_truncate_run")
    runner = StrategyLoopRunner(cfg)
    idir = iteration_dir(cfg.run_id, 1)
    long_message = "Hermes prompt: " + ("x" * 5000)

    runner._record_iteration_failure(idir, "CODE_GEN", RuntimeError(long_message))

    payload = json.loads(leaderboard_path(cfg.run_id).read_text(encoding="utf-8"))
    violation = payload["rows"][0]["violations"][0]
    error = json.loads((idir / "error.json").read_text(encoding="utf-8"))
    assert len(violation) < 700
    assert violation.endswith("...[truncated]")
    assert len(error["traceback"]) <= 4015


def test_prepare_context_includes_pareto_memory() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit_pareto", run_id="unit_pareto_run", max_iterations=4)
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 1,
            "candidate": {"candidate_type": "rank_profile", "name": "best_composite", "rank_profile": {"top_k": 2}},
            "score_components": {"composite_score": 10.0, "freqtrade_score": 5.0, "research_score": 100.0},
            "research_metrics": {"profit_over_max_drawdown": 1.5},
            "freqtrade_metrics": {"profit_pct": 10.0, "profit_over_max_drawdown": 1.0},
        },
        {
            "run_id": cfg.run_id,
            "iteration": 2,
            "candidate": {"candidate_type": "rank_profile", "name": "best_ft_profit", "rank_profile": {"top_k": 3}},
            "score_components": {"composite_score": 8.0, "freqtrade_score": 12.0, "research_score": 90.0},
            "research_metrics": {"profit_over_max_drawdown": 1.2},
            "freqtrade_metrics": {"profit_pct": 25.0, "profit_over_max_drawdown": 0.9},
        },
        {
            "run_id": cfg.run_id,
            "iteration": 3,
            "candidate": {"candidate_type": "rank_profile", "name": "best_research_pdd", "rank_profile": {"top_k": 4}},
            "score_components": {"composite_score": 7.0, "freqtrade_score": 10.0, "research_score": 150.0},
            "research_metrics": {"profit_over_max_drawdown": 3.0},
            "freqtrade_metrics": {"profit_pct": 15.0, "profit_over_max_drawdown": 2.5},
        },
    ]
    state = StrategyLoopState(run_id=cfg.run_id, iteration=4, best_candidate=rows[0], best_score=10.0, score_history=rows)
    save_checkpoint(cfg, state)

    context = prepare_context(cfg, cfg.run_id, 4)

    pareto = context["loop_memory"]["pareto_memory"]
    assert pareto["best_composite"]["name"] == "best_composite"
    assert pareto["best_freqtrade_profit"]["name"] == "best_ft_profit"
    assert pareto["best_freqtrade_profit_over_drawdown"]["name"] == "best_research_pdd"
    assert pareto["best_research_profit_over_drawdown"]["name"] == "best_research_pdd"


def test_promotion_requires_passing_constraints(tmp_path: Path) -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit_promote", run_id="unit_promote_run", promote=True)
    iter_dir = tmp_path / "iter_01"
    iter_dir.mkdir()
    _write_json(iter_dir / "candidate.json", {"candidate_type": "rank_profile", "rank_profile": {"top_k": 2}})
    candidate = validate_candidate(iter_dir / "candidate.json")

    failed = promote_candidate(
        candidate,
        {"constraints_ok": False, "promotion_reason": "trades=0 < 80"},
        cfg,
        iter_dir=iter_dir,
    )
    out = repo_paths.artifacts_root() / "rank_portfolio" / "unit_promote" / "optimized_profile.json"

    assert failed["promoted"] is False
    assert not out.exists()

    passed = promote_candidate(
        candidate,
        {"constraints_ok": True, "score": 123.0, "promotion_reason": "passes hard gates"},
        cfg,
        iter_dir=iter_dir,
    )

    assert passed["promoted"] is True
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["rank_profile"]["top_k"] == 2


def test_promote_policy_final_defers_global_optimized_profile(tmp_path: Path) -> None:
    tag = "unit_promote_final"
    out = repo_paths.artifacts_root() / "rank_portfolio" / tag / "optimized_profile.json"
    _write_json(out, {"sentinel": "unchanged"})
    cfg = StrategyLoopConfig.from_args(tag=tag, run_id="unit_promote_final_run", promote=True, promote_policy="final")
    iter_dir = tmp_path / "iter_01"
    iter_dir.mkdir()
    _write_json(iter_dir / "candidate.json", {"candidate_type": "rank_profile", "rank_profile": {"top_k": 2}})
    candidate = validate_candidate(iter_dir / "candidate.json")

    deferred = promote_candidate(
        candidate,
        {"constraints_ok": True, "score": 123.0, "promotion_reason": "passes hard gates", "iteration": 1},
        cfg,
        iter_dir=iter_dir,
    )

    assert deferred["promoted"] is False
    assert "deferred" in deferred["reason"]
    assert json.loads(out.read_text(encoding="utf-8")) == {"sentinel": "unchanged"}


def test_triple_holdout_defaults_and_scaled_gates() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit_triple", validation_protocol="triple_holdout")

    assert cfg.start == "2025-12-01"
    assert cfg.end == "2026-04-12"
    assert cfg.search_timerange == "20251201-20260228"
    validation_gates = scaled_gate_values(cfg, cfg.validation_timerange)
    blind_gates = scaled_gate_values(cfg, cfg.blind_timerange)
    assert validation_gates["min_trades"] >= 5
    assert blind_gates["min_trades"] >= 5
    assert validation_gates["max_drawdown_pct"] == cfg.max_drawdown_pct
    assert validation_gates["min_profit_over_dd"] == cfg.min_profit_over_dd
    assert validation_gates["target_profit_pct"] < cfg.target_profit_pct


def test_triple_holdout_score_uses_validation_not_search() -> None:
    cfg = StrategyLoopConfig.from_args(
        tag="unit_triple_score",
        validation_protocol="triple_holdout",
        eval_mode="freqtrade",
        score_mode="composite",
    )
    search = {
        "timerange": cfg.search_timerange,
        "total_return_pct": 200.0,
        "max_drawdown_pct": 5.0,
        "profit_over_max_drawdown": 40.0,
        "trades": 500,
        "freqtrade_backtest": {"ok": True, "metrics": {"profit_pct": 200.0, "max_drawdown_pct": 5.0, "profit_over_max_drawdown": 40.0, "trades": 500}},
    }
    validation = {
        "timerange": cfg.validation_timerange,
        "total_return_pct": 8.0,
        "max_drawdown_pct": 10.0,
        "profit_over_max_drawdown": 1.5,
        "trades": 40,
        "freqtrade_backtest": {"ok": True, "metrics": {"profit_pct": 9.0, "max_drawdown_pct": 10.0, "profit_over_max_drawdown": 1.4, "trades": 40}},
    }

    evaluation = score_triple_holdout_backtest({"stages": {"search": search, "validation": validation}}, cfg)
    expected_validation = score_strategy_loop_backtest(validation, cfg, gates=scaled_gate_values(cfg, cfg.validation_timerange))

    assert evaluation["selected_window"] == "validation"
    assert evaluation["score"] == expected_validation["score"]
    assert evaluation["window_metrics"]["search"]["metrics"]["profit_pct"] == 200.0
    assert evaluation["window_metrics"]["validation"]["metrics"]["profit_pct"] == 9.0


def test_triple_holdout_promotion_requires_blind_and_passed_verification(tmp_path: Path) -> None:
    cfg = StrategyLoopConfig.from_args(
        tag="unit_triple_promote",
        run_id="unit_triple_promote_run",
        promote=True,
        promote_policy="final",
        validation_protocol="triple_holdout",
    )
    candidate = {"candidate_type": "rank_profile", "rank_profile": {"top_k": 2}}
    validation_eval = {"constraints_ok": True, "score": 10.0, "promotion_reason": "validation passed"}
    blocked_validation = promote_candidate(candidate, validation_eval, cfg, iter_dir=tmp_path)
    blocked_pending = promote_candidate(
        candidate,
        {**validation_eval, "blind_final": True, "verification_status": "pending", "promotion_eligible": False},
        cfg,
        iter_dir=tmp_path,
        final=True,
    )

    assert blocked_validation["promoted"] is False
    assert "blind" in blocked_validation["reason"]
    assert blocked_pending["promoted"] is False
    assert "verification_status" in blocked_pending["reason"] or "promotion_eligible" in blocked_pending["reason"]


def test_lookahead_and_recursive_parsers_block_bias(tmp_path: Path) -> None:
    ok_csv = tmp_path / "lookahead_ok.csv"
    ok_csv.write_text(
        "strategy,has_bias,biased_entry_signals,biased_exit_signals,biased_indicators,total_signals\n"
        "ELRankPortfolioLeverageStrategy,false,0,0,,12\n",
        encoding="utf-8",
    )
    bad_csv = tmp_path / "lookahead_bad.csv"
    bad_csv.write_text(
        "strategy,has_bias,biased_entry_signals,biased_exit_signals,biased_indicators,total_signals\n"
        "ELRankPortfolioLeverageStrategy,true,1,0,rp_future,12\n",
        encoding="utf-8",
    )
    recursive_ok = tmp_path / "recursive_ok.csv"
    recursive_ok.write_text("indicator,diff\nrsi,0\n", encoding="utf-8")
    recursive_bad = tmp_path / "recursive_bad.csv"
    recursive_bad.write_text("indicator,diff\nrsi,0.001\n", encoding="utf-8")

    assert parse_lookahead_csv(ok_csv, min_trades=5)["status"] == VERIFICATION_PASSED
    assert parse_lookahead_csv(bad_csv, min_trades=5)["status"] == VERIFICATION_FAILED
    assert parse_lookahead_csv(ok_csv, min_trades=50)["status"] == VERIFICATION_FAILED
    assert parse_recursive_output(recursive_ok)["status"] == VERIFICATION_PASSED
    assert parse_recursive_output(recursive_bad)["status"] == VERIFICATION_FAILED
    assert parse_recursive_output(tmp_path / "missing.csv")["status"] == VERIFICATION_INCONCLUSIVE


def test_pareto_pool_six_axes_dedup_and_caps() -> None:
    rows = []
    for idx in range(18):
        rows.append(
            {
                "run_id": "unit_pareto_pool",
                "iteration": idx + 1,
                "candidate_path": f"artifacts/factor_strategy_loop/unit/iter_{idx:02d}/candidate.json",
                "candidate": {"candidate_type": "rank_profile", "name": f"cand_{idx}", "rank_profile": {"top_k": (idx % 5) + 1}},
                "parameter_signature": f"sig_{idx}",
                "score_components": {
                    "composite_score": float(idx),
                    "research_robustness_score": float(100 - idx),
                    "regime_stability_score": float(idx % 7),
                },
                "research_metrics": {"profit_pct": 5.0 + idx, "profit_over_max_drawdown": 1.2 + idx / 100.0, "trades": 20 + idx},
                "freqtrade_metrics": {
                    "profit_pct": float(30 - idx),
                    "profit_over_max_drawdown": float(idx / 2 + 1),
                    "max_drawdown_pct": float(20 - idx / 2),
                    "trades": 20 + idx,
                },
                "constraints_ok": True,
            }
        )

    pool = build_pareto_pool(rows, size_per_axis=3, max_total=12)

    assert set(pool["axes"]) == {
        "best_validation_composite",
        "best_validation_freqtrade_profit",
        "best_validation_freqtrade_profit_over_drawdown",
        "lowest_validation_drawdown_positive_profit",
        "best_research_robustness",
        "best_regime_stability",
    }
    assert all(len(axis_rows) <= 3 for axis_rows in pool["axes"].values())
    assert len(pool["finalists"]) <= 12
    assert all(finalist["pareto_axes"] for finalist in pool["finalists"])


def test_iteration_manifest_uses_artifact_refs_without_embedding_payload(tmp_path: Path) -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit_manifest", run_id="unit_manifest_run")
    idir = tmp_path / "iter_01"
    _write_json(idir / "candidate.json", {"candidate_type": "rank_profile", "rank_profile": {"top_k": 2}})
    _write_json(idir / "backtest.json", {"signals": str(idir / "signals" / "all.feather")})
    (idir / "signals").mkdir()
    (idir / "signals" / "all.feather").write_bytes(b"fake feather")
    candidate = validate_candidate(idir / "candidate.json")
    evaluation = {"iteration": 1, "parameter_signature": "abc", "window_metrics": {"single": {"score": 1.0}}}

    manifest = build_iteration_manifest(idir, cfg, candidate, evaluation)

    assert manifest["candidate_signature"] == "abc"
    assert manifest["artifact_refs"]["candidate.json"]["path"].endswith("candidate.json")
    assert "sha256" in manifest["artifact_refs"]["single_signals"]
    assert "fake feather" not in json.dumps(manifest)


def test_structured_mode_requires_structural_change() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit_structured", run_id="unit_structured_run")
    runner = StrategyLoopRunner(cfg)
    runner.state.exploration_mode = "structured"
    runner.state.best_candidate = {
        "candidate": {"candidate_type": "rank_profile", "rank_profile": {"top_k": 2, "risk_per_trade": 0.04, "side_mode": "short"}}
    }
    local_only = {
        "candidate_type": "rank_profile",
        "metadata": {"search_mode": "structured_explore"},
        "rank_profile": {"top_k": 2, "risk_per_trade": 0.041, "side_mode": "short"},
    }
    structural = {
        "candidate_type": "rank_profile",
        "metadata": {"search_mode": "structured_explore"},
        "rank_profile": {"top_k": 3, "risk_per_trade": 0.041, "side_mode": "short"},
    }

    with pytest.raises(ValueError, match="insufficient structural change"):
        runner._validate_unique_candidate(local_only)
    runner._validate_unique_candidate(structural)
