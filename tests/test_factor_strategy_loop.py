from __future__ import annotations

import json
import subprocess
import types
from pathlib import Path

import pytest

from agent_market import paths as repo_paths
from agent_market.factor_lab import strategy_loop as strategy_loop_mod
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
    build_rank_profile_repair_queue,
    doctor_strategy_loop_run,
    _hermes_cli_env,
    _hermes_model,
    _opencode_cli_env,
    _rank_kwargs,
    iteration_dir,
    leaderboard_path,
    load_checkpoint,
    normalize_rank_profile,
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


def test_fixed_freqtrade_rank_config_supports_market_order_analysis() -> None:
    config_path = repo_paths.REPO_ROOT / strategy_loop_mod.FIXED_FREQTRADE_CONFIG
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert payload["entry_pricing"]["price_side"] == "other"
    assert payload["exit_pricing"]["price_side"] == "other"


def _install_fake_lean_gate(
    monkeypatch: pytest.MonkeyPatch,
    *,
    comparison_status: str = "ok",
    lean_overrides: dict | None = None,
) -> None:
    def fake_export_project(*, rank_artifact, output, timeframe=None, data_root=None):
        project = Path(output)
        (project / "data").mkdir(parents=True, exist_ok=True)
        (project / "data" / "signals.csv").write_text(
            "time,pair,symbol,lean_target_weight\n"
            "2026-04-12 00:00:00,BTC/USDT,BTCUSDT,0\n",
            encoding="utf-8",
        )
        _write_json(project / "manifest.json", {"local_only": True, "timeframe": timeframe, "rank_artifact": str(rank_artifact)})
        (project / "main.py").write_text("# unit\n", encoding="utf-8")
        return {"local_only": True, "timeframe": timeframe, "rank_artifact": str(rank_artifact), "data_root": data_root}

    def fake_run_lean_backtest(*, lean_project, lean_bin="lean", timeout=None):
        result = Path(lean_project) / "results.json"
        _write_json(result, {"statistics": {"unit": True}})
        _write_json(Path(lean_project) / "lean_backtest_run.json", {"command": [lean_bin, "backtest", str(lean_project)], "timeout": timeout})
        return {"command": [lean_bin, "backtest", str(lean_project)], "returncode": 0, "result_path": str(result)}

    def fake_compare_results(*, rank_artifact, lean_result, output=None, timeframe=None, skip_signal_load=False):
        lean = {
            "final_equity": 1.12,
            "max_drawdown": 0.08,
            "trades": 120.0,
            "orders": 12.0,
            "turnover": 1.0,
            "max_gross": 1.0,
            "fee_cost": 0.001,
            "ending_open_positions": 0.0,
        }
        if lean_overrides:
            lean.update(lean_overrides)
        metrics = {
            field: {"status": "ok", "research": lean.get(field), "lean": lean.get(field), "threshold": 0.05}
            for field in ("final_equity", "max_drawdown", "trades", "orders", "turnover")
        }
        report = {
            "status": comparison_status,
            "metrics": metrics,
            "lean": lean,
            "research": dict(lean),
            "rank_artifact": str(rank_artifact),
            "lean_result": str(lean_result),
            "output": str(output) if output else "",
        }
        if output:
            _write_json(Path(output), report)
        return report

    monkeypatch.setattr(strategy_loop_mod.lean_bridge, "export_project", fake_export_project)
    monkeypatch.setattr(strategy_loop_mod.lean_bridge, "run_lean_backtest", fake_run_lean_backtest)
    monkeypatch.setattr(strategy_loop_mod.lean_bridge, "compare_results", fake_compare_results)


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


def test_resume_extended_stagnated_run_advances_with_grace() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit", run_id="unit_resume_stagnated", max_iterations=3)
    state = StrategyLoopState(run_id=cfg.run_id, iteration=3, phase=PHASE_COMPLETE)
    state.status = strategy_loop_mod.LOOP_STOPPED_STAGNATED
    state.stopped_reason = "30 valid candidates without composite improvement"
    state.no_composite_improvement_count = strategy_loop_mod.STAGNATION_STOP_AFTER
    state.exploration_mode = "structured"
    save_checkpoint(cfg, state)

    resume_cfg = StrategyLoopConfig.from_args(
        tag="unit",
        run_id=cfg.run_id,
        max_iterations=5,
        resume=True,
    )
    runner = StrategyLoopRunner(resume_cfg)

    assert runner.state.iteration == 4
    assert runner.state.phase == strategy_loop_mod.PHASE_PREPARE
    assert runner.state.status == strategy_loop_mod.LOOP_RUNNING
    assert runner.state.stopped_reason == ""
    assert runner.state.no_composite_improvement_count == strategy_loop_mod._stagnation_grace_count()
    assert runner.state.exploration_mode == "structured"


def test_resume_extended_completed_run_marks_running() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit", run_id="unit_resume_completed", max_iterations=3)
    state = StrategyLoopState(run_id=cfg.run_id, iteration=4, phase=strategy_loop_mod.PHASE_PREPARE)
    state.status = LOOP_COMPLETED
    state.final_promotion = {"promoted": False, "reason": "old final"}
    save_checkpoint(cfg, state)

    resume_cfg = StrategyLoopConfig.from_args(
        tag="unit",
        run_id=cfg.run_id,
        max_iterations=5,
        resume=True,
    )
    runner = StrategyLoopRunner(resume_cfg)

    assert runner.state.iteration == 4
    assert runner.state.status == strategy_loop_mod.LOOP_RUNNING
    assert runner.state.final_promotion is None


def test_resume_strict_formal_rejects_stale_controller_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    cfg = StrategyLoopConfig.from_args(
        tag="unit",
        run_id="unit_resume_stale_formal",
        max_iterations=3,
        validation_protocol="triple_holdout",
        verify_policy="pareto",
        promote_policy="final",
    )
    state = StrategyLoopState(run_id=cfg.run_id, iteration=2, phase=PHASE_COMPLETE, status=LOOP_COMPLETED)
    save_checkpoint(cfg, state)
    _write_json(
        repo_paths.artifacts_root() / "factor_strategy_loop" / cfg.run_id / "manifest.json",
        {"cli_args": cfg.__dict__, "git": {"commit": "old-controller-sha", "dirty_files": []}},
    )
    resume_cfg = StrategyLoopConfig.from_args(
        tag="unit",
        run_id=cfg.run_id,
        max_iterations=4,
        resume=True,
        validation_protocol="triple_holdout",
        verify_policy="pareto",
        promote_policy="final",
    )

    with pytest.raises(ValueError, match="refusing to resume strict formal"):
        StrategyLoopRunner(resume_cfg)

    monkeypatch.setenv("AGENT_MARKET_ALLOW_STALE_FORMAL_RESUME", "1")
    runner = StrategyLoopRunner(resume_cfg)
    assert runner.state.status == strategy_loop_mod.LOOP_RUNNING


def test_stagnation_recovery_candidate_gets_grace_at_stop_threshold() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit", run_id="unit_recovery_stagnation", max_iterations=3)
    runner = StrategyLoopRunner(cfg)
    runner.state.best_composite_score = 10.0
    runner.state.no_composite_improvement_count = strategy_loop_mod.STAGNATION_STOP_AFTER - 1

    runner._update_stagnation(
        {
            "score_components": {"composite_score": 1.0},
            "candidate": {
                "metadata": {
                    "source": "controller_rank_profile_search_quality_repair",
                    "hypothesis_family": "search_quality_entry_repair_after_duplicate_paths",
                    "behavior_feedback": ["validation-pass repairs duplicated the baseline path"],
                }
            },
        }
    )

    assert runner.state.status == strategy_loop_mod.LOOP_RUNNING
    assert runner.state.no_composite_improvement_count == strategy_loop_mod._stagnation_grace_count() + 1
    assert runner.state.exploration_mode == "structured"


def test_positive_validation_recovery_candidate_gets_stagnation_grace() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit", run_id="unit_positive_validation_recovery", max_iterations=3)
    runner = StrategyLoopRunner(cfg)
    runner.state.best_composite_score = 10.0
    runner.state.no_composite_improvement_count = strategy_loop_mod.STAGNATION_STOP_AFTER - 1

    runner._update_stagnation(
        {
            "score_components": {"composite_score": 1.0},
            "candidate": {
                "metadata": {
                    "source": "controller_rank_profile_positive_validation_trade_repair",
                    "hypothesis_family": "validation_trade_regime_coverage_repair",
                }
            },
        }
    )

    assert runner.state.status == strategy_loop_mod.LOOP_RUNNING
    assert runner.state.no_composite_improvement_count == strategy_loop_mod._stagnation_grace_count() + 1


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
                f"        df = pd.read_feather('{repo_paths.REPO_ROOT / 'artifacts' / 'x.feather'}')",
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
    assert "final_blind_feedback" not in prompt
    assert "do not tune from blind holdout results" in prompt
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


def test_rank_profile_normalizes_common_enum_aliases() -> None:
    profile = normalize_rank_profile(
        {
            "edge_mode": "on",
            "regime_mode": "on",
            "side_mode": "short",
        }
    )

    assert profile["edge_mode"] == "rolling_ic"
    assert profile["regime_mode"] == "hq"
    assert profile["side_mode"] == "short"


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
        lean_gate_mode="final",
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
        "lean_gate": {"status": VERIFICATION_PASSED, "comparison_status": "ok"},
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
    _write_json(blind_dir / "lean_gate.json", {"status": VERIFICATION_PASSED, "comparison_status": "ok"})
    _write_json(
        blind_dir / "manifest.json",
        {"artifact_refs": {"verification.json": {"path": "x", "sha256": "abc", "bytes": 1}}},
    )

    result = doctor_strategy_loop_run(run_id)

    assert result["ok"] is True
    assert result["summary"]["verification_counts"][VERIFICATION_PASSED] == 1
    assert result["findings"] == []
    persisted = json.loads((root / "doctor_latest.json").read_text(encoding="utf-8"))
    assert persisted["ok"] is True
    assert persisted["artifacts"]["doctor_latest.json"].endswith("/doctor_latest.json")


def test_strategy_loop_doctor_flags_stale_run_git_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    run_id = "doctor_formal_stale_git"
    root = repo_paths.artifacts_root() / "factor_strategy_loop" / run_id
    root.mkdir(parents=True)
    cfg = StrategyLoopConfig.from_args(
        tag="unit",
        run_id=run_id,
        validation_protocol="triple_holdout",
        verify_policy="pareto",
        promote_policy="final",
        lean_gate_mode="final",
    )
    _write_json(root / "checkpoint.json", {"config": cfg.__dict__, "state": {"run_id": run_id, "iteration": 1}})
    _write_json(root / "manifest.json", {"cli_args": cfg.__dict__, "git": {"commit": "old-controller-sha", "dirty_files": []}})
    _write_json(root / "leaderboard.json", {"rows": [{"iteration": 1, "promotion_eligible": False}]})
    _write_json(root / "pareto_pool.json", {"finalists": []})
    _write_json(root / "final_promotion.json", {"promoted": False})

    deep_root = repo_paths.artifacts_root() / "strategy_deepresearch" / run_id
    _write_json(deep_root / "context.json", {"run_id": run_id})
    _write_json(deep_root / "sources.json", {"sources": []})
    _write_json(
        root / "final_blind_status.json",
        {
            "selected": {
                "blind_final": True,
                "promotion_eligible": True,
                "verification_status": VERIFICATION_PASSED,
                "lean_gate": {"status": VERIFICATION_PASSED, "comparison_status": "ok"},
            },
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
    _write_json(blind_dir / "lean_gate.json", {"status": VERIFICATION_PASSED, "comparison_status": "ok"})
    _write_json(blind_dir / "manifest.json", {"artifact_refs": {"verification.json": {"path": "x", "sha256": "abc", "bytes": 1}}})

    result = doctor_strategy_loop_run(run_id, write=False)

    assert result["ok"] is False
    assert result["summary"]["run_manifest_commit"] == "old-controller-sha"
    assert any("git commit differs" in item["message"] for item in result["findings"])


def test_strategy_loop_doctor_no_finalists_does_not_require_final_sidecars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    run_id = "doctor_no_finalists"
    root = repo_paths.artifacts_root() / "factor_strategy_loop" / run_id
    root.mkdir(parents=True)
    cfg = StrategyLoopConfig.from_args(
        tag="unit",
        run_id=run_id,
        validation_protocol="triple_holdout",
        verify_policy="pareto",
        promote_policy="final",
        lean_gate_mode="final",
    )
    _write_json(root / "checkpoint.json", {"config": cfg.__dict__, "state": {"run_id": run_id, "iteration": 1}})
    _write_json(root / "manifest.json", {"cli_args": cfg.__dict__, "git": strategy_loop_mod._git_provenance()})
    _write_json(root / "leaderboard.json", {"rows": [{"iteration": 1, "promotion_eligible": False, "constraints_ok": False}]})
    _write_json(root / "pareto_pool.json", {"finalists": []})
    _write_json(root / "final_blind_status.json", {"selected": None, "finalists": [], "promotion": {"promoted": False}})
    _write_json(root / "final_promotion.json", {"promoted": False})
    _write_json(root / "iter_01" / "manifest.json", {"artifact_refs": {"evaluation.json": {"path": "x", "sha256": "abc", "bytes": 1}}})

    result = doctor_strategy_loop_run(run_id, write=False)
    messages = [item["message"] for item in result["findings"]]

    assert "no selected blind finalist" in messages
    assert not any("verification.json files" in message for message in messages)
    assert not any("deepresearch artifact missing" in message for message in messages)


def test_strategy_loop_doctor_persists_and_cli_fails_on_blockers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    run_id = "doctor_formal_blocked"
    root = repo_paths.artifacts_root() / "factor_strategy_loop" / run_id
    root.mkdir(parents=True)
    cfg = StrategyLoopConfig.from_args(tag="unit", run_id=run_id, promote_policy="final")
    _write_json(root / "checkpoint.json", {"config": cfg.__dict__, "state": {"run_id": run_id}})
    _write_json(root / "leaderboard.json", {"rows": []})

    result = doctor_strategy_loop_run(run_id)
    assert result["ok"] is False
    assert (root / "doctor_latest.json").exists()
    persisted = json.loads((root / "doctor_latest.json").read_text(encoding="utf-8"))
    assert persisted["ok"] is False
    assert any(item["severity"] == "BLOCKER" for item in persisted["findings"])

    from scripts.factor_lab import cmd_strategy_loop_doctor

    args = types.SimpleNamespace(run_id=run_id, no_strict_formal=False, no_write=False, no_fail=False)
    with pytest.raises(SystemExit) as exc:
        cmd_strategy_loop_doctor(args)
    assert exc.value.code == 1


def test_strategy_loop_formal_cli_preset_forces_strict_promotion_gates() -> None:
    from scripts.factor_lab import _apply_strategy_loop_formal_preset

    args = types.SimpleNamespace(
        formal=True,
        eval_mode="two_stage",
        score_mode="research",
        promote_policy="immediate",
        validation_protocol="single",
        verify_policy="none",
        lean_gate_mode="off",
    )

    _apply_strategy_loop_formal_preset(args)

    assert args.eval_mode == "two_stage"
    assert args.score_mode == "composite"
    assert args.promote_policy == "final"
    assert args.validation_protocol == "triple_holdout"
    assert args.verify_policy == "pareto"
    assert args.lean_gate_mode == "final"


def test_strategy_loop_formal_cli_preset_preserves_stricter_gates() -> None:
    from scripts.factor_lab import _apply_strategy_loop_formal_preset

    args = types.SimpleNamespace(
        formal=True,
        eval_mode="two_stage",
        score_mode="research",
        promote_policy="immediate",
        validation_protocol="single",
        verify_policy="all",
        lean_gate_mode="all",
    )

    _apply_strategy_loop_formal_preset(args)

    assert args.validation_protocol == "triple_holdout"
    assert args.verify_policy == "all"
    assert args.lean_gate_mode == "all"


def test_strategy_loop_formal_cli_preset_leaves_manual_mode_unchanged() -> None:
    from scripts.factor_lab import _apply_strategy_loop_formal_preset

    args = types.SimpleNamespace(
        formal=False,
        eval_mode="two_stage",
        score_mode="research",
        promote_policy="immediate",
        validation_protocol="single",
        verify_policy="none",
        lean_gate_mode="off",
    )

    _apply_strategy_loop_formal_preset(args)

    assert args.eval_mode == "two_stage"
    assert args.score_mode == "research"
    assert args.promote_policy == "immediate"
    assert args.validation_protocol == "single"
    assert args.verify_policy == "none"
    assert args.lean_gate_mode == "off"


def test_factor_lab_cli_default_lean_bin_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.factor_lab import _default_lean_bin

    monkeypatch.setenv("LEAN_BIN", "/opt/lean/bin/lean")

    assert _default_lean_bin() == "/opt/lean/bin/lean"


def test_factor_lab_cli_default_lean_bin_finds_user_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.factor_lab import _default_lean_bin

    user_lean = tmp_path / ".local" / "bin" / "lean"
    user_lean.parent.mkdir(parents=True)
    user_lean.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.delenv("LEAN_BIN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert _default_lean_bin() == str(user_lean)


def test_triple_holdout_without_finalists_writes_final_promotion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    run_id = "unit_empty_finalists"
    cfg = StrategyLoopConfig.from_args(
        tag="unit_empty",
        run_id=run_id,
        promote=True,
        promote_policy="final",
        validation_protocol="triple_holdout",
        max_iterations=0,
    )

    result = StrategyLoopRunner(cfg).run()
    root = repo_paths.artifacts_root() / "factor_strategy_loop" / run_id
    final_status = json.loads((root / "final_blind_status.json").read_text(encoding="utf-8"))
    final_promotion = json.loads((root / "final_promotion.json").read_text(encoding="utf-8"))

    assert final_promotion == {"promoted": False, "artifacts": {}, "reason": "no Pareto finalists available"}
    assert final_status["promotion"] == final_promotion
    assert result["final_promotion"] == final_promotion


def test_triple_holdout_skips_invalid_pareto_finalist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    run_id = "unit_invalid_finalist"
    cfg = StrategyLoopConfig.from_args(
        tag="unit_invalid_finalist",
        run_id=run_id,
        promote=True,
        promote_policy="final",
        validation_protocol="triple_holdout",
    )
    invalid_candidate = tmp_path / "candidate.json"
    _write_json(
        invalid_candidate,
        {
            "candidate_type": "rank_profile",
            "rank_profile": {"top_k": 2, "regime_mode": "unsupported"},
        },
    )
    runner = StrategyLoopRunner(cfg)
    monkeypatch.setattr(
        runner,
        "_refresh_pareto_pool",
        lambda: {"finalists": [{"iteration": 1, "candidate_path": str(invalid_candidate)}]},
    )
    monkeypatch.setattr(runner, "_deepresearch_sidecar", lambda final_status: {"status": VERIFICATION_PASSED})

    promotion = runner._finalize_triple_holdout()
    final_status = json.loads((repo_paths.artifacts_root() / "factor_strategy_loop" / run_id / "final_blind_status.json").read_text(encoding="utf-8"))

    assert promotion["promoted"] is False
    assert "no Pareto finalist passed" in promotion["reason"]
    assert final_status["finalists"][0]["reason"].startswith("candidate invalid:")


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
                "pair_edge_min_entry_ic": 0.03,
                "pair_edge_min_hold_ic": 0.02,
                "min_pairs_for_top_k": 6,
                "low_pair_top_k": 2,
            },
        },
    )

    normalized = validate_candidate(candidate)

    profile = normalized["rank_profile"]
    assert profile["candidate_state"].endswith("state_0149.json")
    assert profile["recompute_corr"] is False
    assert profile["short_max_mom_24h"] == 0.04
    assert profile["pair_edge_leverage"] is False
    assert profile["pair_edge_min_entry_ic"] == 0.03
    assert profile["pair_edge_min_hold_ic"] == 0.02
    assert profile["min_pairs_for_top_k"] == 6
    assert profile["low_pair_top_k"] == 2


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
                "pair_edge_min_entry_ic": 0.025,
                "pair_edge_min_hold_ic": 0.015,
                "min_pairs_for_top_k": 7,
                "low_pair_top_k": 2,
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
    assert kwargs["pair_edge_min_entry_ic"] == 0.025
    assert kwargs["pair_edge_min_hold_ic"] == 0.015
    assert kwargs["min_pairs_for_top_k"] == 7
    assert kwargs["low_pair_top_k"] == 2


def test_rank_kwargs_expands_short_candidate_state_from_config() -> None:
    state_rel = "artifacts/factor_lab/mining/unit_rank_kwargs_state/state_0149.json"
    state_path = repo_paths.resolve_repo_path(state_rel)
    _write_json(state_path, {"survivors": []})
    cfg = StrategyLoopConfig.from_args(
        tag="unit_rank_kwargs_state",
        candidate_state=state_rel,
    )

    kwargs = _rank_kwargs(
        {"candidate_state": "state_0149.json", "top_k": 3},
        cfg,
        candidate_state=None,
        tag="effective",
    )

    assert kwargs["candidate_state"] == str(state_path)
    assert kwargs["top_k"] == 3


def test_rank_kwargs_rejects_leaky_legacy_candidate_state_for_triple_holdout() -> None:
    state_rel = "artifacts/factor_lab/mining/unit_leaky_legacy_state/state_0149.json"
    state_path = repo_paths.resolve_repo_path(state_rel)
    _write_json(
        state_path,
        {
            "timeframe": "1h",
            "config": {
                "eval_mode": "legacy",
                "oos": ["2025-11-01", "2026-04-12"],
            },
            "survivors": [],
        },
    )
    cfg = StrategyLoopConfig.from_args(
        tag="unit_leaky_legacy_state",
        candidate_state=state_rel,
        validation_protocol="triple_holdout",
        search_timerange="20251201-20260228",
        validation_timerange="20260301-20260331",
        blind_timerange="20260401-20260412",
    )

    with pytest.raises(ValueError, match="mining selection window overlaps formal search"):
        _rank_kwargs({}, cfg, candidate_state=None, tag="effective")


def test_rank_kwargs_allows_portfolio_state_with_pre_search_val3() -> None:
    state_rel = "artifacts/factor_lab/mining/unit_clean_portfolio_state/state_0149.json"
    state_path = repo_paths.resolve_repo_path(state_rel)
    _write_json(
        state_path,
        {
            "timeframe": "1h",
            "config": {
                "eval_mode": "portfolio",
                "oos": ["2025-11-01", "2026-04-12"],
                "val3": ["2025-07-01", "2025-12-01"],
            },
            "survivors": [],
        },
    )
    cfg = StrategyLoopConfig.from_args(
        tag="unit_clean_portfolio_state",
        candidate_state=state_rel,
        validation_protocol="triple_holdout",
        search_timerange="20251201-20260228",
        validation_timerange="20260301-20260331",
        blind_timerange="20260401-20260412",
    )

    kwargs = _rank_kwargs({}, cfg, candidate_state=None, tag="effective")

    assert kwargs["candidate_state"] == str(state_path)


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


def test_openai_compatible_env_and_model_aliases() -> None:
    from agent_market.factor_lab.strategy_loop import _openai_compatible_env, _openai_compatible_model

    env = _openai_compatible_env(
        {
            "LLM_BASE_URL": "http://127.0.0.1:8317",
            "LLM_API_KEY": "local-key",
            "LLM_MODEL": "custom/gpt-5.5",
        },
        load_dotenv=False,
    )

    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8317/v1"
    assert env["OPENAI_API_KEY"] == "local-key"
    assert _openai_compatible_model("", env) == "gpt-5.5"


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


def test_runner_uses_openai_compatible_agent_for_codegen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = StrategyLoopConfig.from_args(
        tag="unit_openai",
        run_id="unit_openai_run",
        agent="openai",
        model="gpt-5.5",
        max_turns=5,
        max_retries=1,
    )
    runner = StrategyLoopRunner(cfg)
    runner.state.iteration = 2
    captured: dict[str, object] = {}

    monkeypatch.setenv("OPENAI_API_KEY", "unit-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8317/v1")

    import agent_market.strategy_miner.agent_adapter as adapter

    class FakeStrategyAgent:
        def __init__(self, **kwargs: object) -> None:
            captured["kwargs"] = kwargs

        def run_result(self, prompt: str) -> types.SimpleNamespace:
            captured["prompt"] = prompt
            return types.SimpleNamespace(
                assistant_text=json.dumps(
                    {
                        "candidate_type": "rank_profile",
                        "name": "openai_candidate",
                        "rank_profile": {"top_k": 2, "rebalance_hours": 8},
                    }
                ),
                usage={"total_tokens": 7},
            )

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(adapter, "StrategyAgent", FakeStrategyAgent)
    (tmp_path / "context").mkdir()
    _write_json(tmp_path / "context" / "prepare.json", {"unit_context": True})

    runner._code_gen(tmp_path)

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["provider"] == "openai"
    assert kwargs["model"] == "gpt-5.5"
    assert kwargs["base_url"] == "http://127.0.0.1:8317/v1"
    assert "direct OpenAI-compatible strategy-loop adapter" in str(captured["prompt"])
    assert '"unit_context": true' in str(captured["prompt"])
    assert validate_candidate(tmp_path / "candidate.json")["name"] == "openai_candidate"
    assert runner.state.token_cost["2"] == {"total_tokens": 7}
    assert captured["closed"] is True


def test_openai_compatible_agent_repairs_invalid_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = StrategyLoopConfig.from_args(
        tag="unit_openai_repair",
        run_id="unit_openai_repair_run",
        agent="openai",
        model="gpt-5.5",
        max_retries=1,
    )
    runner = StrategyLoopRunner(cfg)
    runner.state.iteration = 2
    prompts: list[str] = []

    monkeypatch.setenv("OPENAI_API_KEY", "unit-key")

    import agent_market.strategy_miner.agent_adapter as adapter

    class FakeStrategyAgent:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run_result(self, prompt: str) -> types.SimpleNamespace:
            prompts.append(prompt)
            if len(prompts) == 1:
                return types.SimpleNamespace(
                    assistant_text='```json\n{"candidate_type":"rank_profile","name":"truncated"',
                    usage={"total_tokens": 11},
                )
            return types.SimpleNamespace(
                assistant_text=json.dumps(
                    {
                        "candidate_type": "rank_profile",
                        "name": "repaired_candidate",
                        "rank_profile": {"top_k": 2, "rebalance_hours": 8},
                    }
                ),
                usage={"total_tokens": 13},
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr(adapter, "StrategyAgent", FakeStrategyAgent)
    (tmp_path / "context").mkdir()
    _write_json(tmp_path / "context" / "prepare.json", {"unit_context": True})

    runner._code_gen(tmp_path)

    assert len(prompts) == 2
    assert "not valid complete JSON" in prompts[1]
    assert validate_candidate(tmp_path / "candidate.json")["name"] == "repaired_candidate"
    response = (tmp_path / "agent_response.txt").read_text(encoding="utf-8")
    assert "JSON repair attempt" in response
    assert runner.state.token_cost["2"]["repair"] == {"total_tokens": 13}


def test_openai_compatible_agent_repairs_structured_metadata_and_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_rel = "artifacts/factor_lab/mining/unit_openai_structured/state_0149.json"
    state_path = repo_paths.resolve_repo_path(state_rel)
    _write_json(state_path, {"survivors": []})
    cfg = StrategyLoopConfig.from_args(
        tag="unit_openai_structured",
        run_id="unit_openai_structured_run",
        agent="openai",
        model="gpt-5.5",
        candidate_state=state_rel,
    )
    runner = StrategyLoopRunner(cfg)
    runner.state.iteration = 2
    runner.state.exploration_mode = "structured"
    runner.state.best_candidate = {
        "candidate": {
            "candidate_type": "rank_profile",
            "rank_profile": {"top_k": 2, "candidate_state": state_rel},
        }
    }

    monkeypatch.setenv("OPENAI_API_KEY", "unit-key")

    import agent_market.strategy_miner.agent_adapter as adapter

    class FakeStrategyAgent:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run_result(self, prompt: str) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                assistant_text=json.dumps(
                    {
                        "candidate_type": "rank_profile",
                        "name": "structured_topk",
                        "rank_profile": {"top_k": 3, "candidate_state": "state_0149.json"},
                    }
                ),
                usage={"total_tokens": 7},
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr(adapter, "StrategyAgent", FakeStrategyAgent)
    (tmp_path / "context").mkdir()
    _write_json(tmp_path / "context" / "prepare.json", {"unit_context": True})

    runner._code_gen(tmp_path)

    candidate = validate_candidate(tmp_path / "candidate.json")
    assert candidate["metadata"]["search_mode"] == "structured_explore"
    assert candidate["metadata"]["candidate_state_repair"] == "expanded_short_state_path"
    assert candidate["rank_profile"]["candidate_state"] == state_rel
    assert repo_paths.resolve_repo_path(candidate["rank_profile"]["candidate_state"]) == state_path


def test_openai_compatible_agent_repairs_controller_contract_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_rel = "artifacts/factor_lab/mining/unit_openai_contract/state_0149.json"
    _write_json(repo_paths.resolve_repo_path(state_rel), {"survivors": []})
    cfg = StrategyLoopConfig.from_args(
        tag="unit_openai_contract",
        run_id="unit_openai_contract_run",
        agent="openai",
        model="gpt-5.5",
        candidate_state=state_rel,
        max_retries=1,
    )
    runner = StrategyLoopRunner(cfg)
    runner.state.iteration = 2
    runner.state.exploration_mode = "structured"
    runner.state.best_candidate = {
        "candidate": {
            "candidate_type": "rank_profile",
            "rank_profile": {"n": 50, "top_k": 2, "min_abs_score_z": 1.45, "candidate_state": state_rel},
        }
    }

    monkeypatch.setenv("OPENAI_API_KEY", "unit-key")

    import agent_market.strategy_miner.agent_adapter as adapter

    calls: list[str] = []

    class FakeStrategyAgent:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run_result(self, prompt: str) -> types.SimpleNamespace:
            calls.append(prompt)
            if len(calls) == 1:
                payload = {
                    "candidate_type": "rank_profile",
                    "name": "local_risk_only",
                    "metadata": {"search_mode": "structured_explore"},
                    "rank_profile": {
                        "n": 50,
                        "top_k": 2,
                        "min_abs_score_z": 1.45,
                        "risk_per_trade": 0.02,
                        "candidate_state": state_rel,
                    },
                }
            else:
                payload = {
                    "candidate_type": "rank_profile",
                    "name": "repaired_topk",
                    "metadata": {"search_mode": "structured_explore"},
                    "rank_profile": {"n": 50, "top_k": 3, "min_abs_score_z": 1.48, "candidate_state": state_rel},
                }
            return types.SimpleNamespace(assistant_text=json.dumps(payload), usage={"total_tokens": len(calls)})

        def close(self) -> None:
            pass

    monkeypatch.setattr(adapter, "StrategyAgent", FakeStrategyAgent)
    (tmp_path / "context").mkdir()
    _write_json(tmp_path / "context" / "prepare.json", {"loop_memory": {}})

    runner._code_gen(tmp_path)

    candidate = validate_candidate(tmp_path / "candidate.json")
    assert len(calls) == 2
    assert "failed the strategy-loop controller contract" in calls[1]
    assert candidate["name"] == "repaired_topk"
    assert candidate["rank_profile"]["top_k"] == 3
    assert runner.state.token_cost["2"]["contract_repair"] == {"total_tokens": 2}
    assert "contract repair attempt" in (tmp_path / "agent_response.txt").read_text(encoding="utf-8")


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


def test_direct_agent_context_compaction_keeps_signal_summary_small() -> None:
    row = {
        "iteration": 12,
        "candidate": {
            "candidate_type": "rank_profile",
            "name": "large_signal_row",
            "rank_profile": {"top_k": 3, "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json"},
        },
        "metrics": {"profit_pct": 1.2, "max_drawdown_pct": 0.5, "trades": 10, "profit_over_max_drawdown": 2.4},
        "window_metrics": {
            "validation": {
                "research_metrics": {"profit_pct": 1.0, "max_drawdown_pct": 0.5, "trades": 8, "profit_over_max_drawdown": 2.0},
                "freqtrade_backtest": {"metrics": {"profit_pct": 0.8, "max_drawdown_pct": 0.4, "trades": 8}},
            }
        },
        "behavior_novelty": {
            "status": "near_duplicate",
            "stage": "validation",
            "reason": "unit",
            "fingerprint": {
                "active_rows": 52,
                "active_days": 7,
                "active_pairs": 4,
                "pair_counts": {"BTC/USDT": 20},
                "daily_active_counts": {f"2026-03-{day:02d}": day for day in range(1, 32)},
                "action_signature": "a" * 64,
            },
        },
    }
    context = {
        "version": "unit",
        "objective": {"candidate_type": "rank_profile"},
        "loop_memory": {
            "recent_score_history": [row] * 30,
            "pareto_memory": {"best_validation_composite": [row] * 10},
            "avoid_repeating_rank_profile_signatures": [str(i) for i in range(100)],
        },
    }

    compact = strategy_loop_mod._compact_direct_agent_context(context)

    rendered = json.dumps(compact, sort_keys=True)
    assert len(rendered) < 50_000
    assert compact["context_compaction"]["source_chars"] > compact["context_compaction"]["compact_chars"]
    recent = compact["loop_memory"]["recent_score_history"]
    assert len(recent) == 6
    assert "daily_active_counts" not in rendered
    assert recent[-1]["behavior_novelty"]["fingerprint"]["active_rows"] == 52


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


def _write_signal_behavior_fixture(signal_dir: Path, *, weight: float = -1.0, active_pairs: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")) -> None:
    import pandas as pd

    signal_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    dates = pd.date_range("2026-03-01", periods=5, freq="1D", tz="UTC")
    for pair in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "pair": pair,
                    "rp_target_weight": weight if pair in active_pairs and idx < 4 else 0.0,
                }
            )
    pd.DataFrame(rows).to_feather(signal_dir / "all.feather")


def test_signal_behavior_fingerprint_detects_same_signal_path(tmp_path: Path) -> None:
    first_dir = tmp_path / "signals_a"
    second_dir = tmp_path / "signals_b"
    _write_signal_behavior_fixture(first_dir, weight=-1.0)
    _write_signal_behavior_fixture(second_dir, weight=-0.5)

    first = strategy_loop_mod._signal_behavior_fingerprint_from_signal_dir(first_dir)
    second = strategy_loop_mod._signal_behavior_fingerprint_from_signal_dir(second_dir)
    duplicate = strategy_loop_mod._signal_behavior_duplicate(second, first)

    assert first["active_rows"] == 8
    assert first["active_days"] == 4
    assert first["active_pairs"] == 2
    assert first["pair_counts"] == {"BTC/USDT": 4, "ETH/USDT": 4}
    assert first["action_signature"] != second["action_signature"]
    assert duplicate["status"] == "no_op"


def test_pareto_pool_excludes_signal_behavior_duplicates(tmp_path: Path) -> None:
    first_dir = tmp_path / "signals_a"
    second_dir = tmp_path / "signals_b"
    _write_signal_behavior_fixture(first_dir, weight=-1.0)
    _write_signal_behavior_fixture(second_dir, weight=-0.5)
    first_fp = strategy_loop_mod._signal_behavior_fingerprint_from_signal_dir(first_dir)

    rows = [
        {
            "run_id": "unit_behavior_pareto",
            "iteration": 1,
            "candidate_path": "iter_01/candidate.json",
            "candidate": {"candidate_type": "rank_profile", "name": "first", "rank_profile": {"top_k": 2}},
            "score": 10.0,
            "score_components": {"composite_score": 10.0},
            "constraints_ok": True,
            "research_metrics": {"profit_pct": 2.0, "profit_over_max_drawdown": 2.0, "trades": 12},
            "freqtrade_metrics": {"profit_pct": 2.0, "profit_over_max_drawdown": 2.0, "trades": 12},
            "window_metrics": {"validation": {"signal_dir": str(first_dir)}},
        },
        {
            "run_id": "unit_behavior_pareto",
            "iteration": 2,
            "candidate_path": "iter_02/candidate.json",
            "candidate": {"candidate_type": "rank_profile", "name": "same_path", "rank_profile": {"top_k": 3}},
            "score": 20.0,
            "score_components": {"composite_score": 20.0},
            "constraints_ok": True,
            "research_metrics": {"profit_pct": 3.0, "profit_over_max_drawdown": 3.0, "trades": 12},
            "freqtrade_metrics": {"profit_pct": 3.0, "profit_over_max_drawdown": 3.0, "trades": 12},
            "window_metrics": {"validation": {"signal_dir": str(second_dir)}},
        },
    ]

    pool = build_pareto_pool(rows, size_per_axis=3)
    excluded_pool = build_pareto_pool(rows, size_per_axis=3, excluded_signal_fingerprints=[first_fp])

    assert [item["iteration"] for item in pool["finalists"]] == [2]
    assert excluded_pool["finalists"] == []


def test_runner_marks_validation_pass_signal_noop_not_pareto(tmp_path: Path) -> None:
    prior_dir = tmp_path / "prior"
    current_dir = tmp_path / "current"
    _write_signal_behavior_fixture(prior_dir, weight=-1.0)
    _write_signal_behavior_fixture(current_dir, weight=-0.5)
    cfg = StrategyLoopConfig.from_args(
        tag="unit_behavior_gate",
        run_id="unit_behavior_gate_run",
        validation_protocol="triple_holdout",
        promote_policy="final",
    )
    runner = StrategyLoopRunner(cfg)
    runner.state.score_history = [
        {
            "run_id": cfg.run_id,
            "iteration": 1,
            "candidate_path": "iter_01/candidate.json",
            "constraints_ok": True,
            "window_metrics": {"validation": {"signal_dir": str(prior_dir)}},
        }
    ]
    evaluation = {
        "constraints_ok": True,
        "violations": [],
        "promotion_reason": "validation window passed selected hard gates",
        "window_metrics": {"validation": {"signal_dir": str(current_dir)}},
        "score_components": {},
    }

    runner._apply_behavior_novelty_gate(evaluation)

    assert evaluation["pareto_eligible"] is False
    assert evaluation["behavior_novelty"]["status"] == "no_op"
    assert "excluded from Pareto/blind" in evaluation["promotion_reason"]
    assert any("behavior_novelty" in item for item in evaluation["violations"])


def test_runner_marks_validation_fail_signal_noop_as_feedback(tmp_path: Path) -> None:
    prior_dir = tmp_path / "prior"
    current_dir = tmp_path / "current"
    _write_signal_behavior_fixture(prior_dir, weight=-1.0)
    _write_signal_behavior_fixture(current_dir, weight=-0.5)
    cfg = StrategyLoopConfig.from_args(
        tag="unit_behavior_fail_feedback",
        run_id="unit_behavior_fail_feedback_run",
        validation_protocol="triple_holdout",
        promote_policy="final",
    )
    runner = StrategyLoopRunner(cfg)
    runner.state.score_history = [
        {
            "run_id": cfg.run_id,
            "iteration": 1,
            "candidate_path": "iter_01/candidate.json",
            "constraints_ok": False,
            "window_metrics": {"validation": {"signal_dir": str(prior_dir)}},
        }
    ]
    evaluation = {
        "constraints_ok": False,
        "violations": ["research: trades=11 < 19"],
        "window_metrics": {"validation": {"signal_dir": str(current_dir)}},
        "score_components": {},
    }

    runner._apply_behavior_novelty_gate(evaluation)

    assert evaluation["pareto_eligible"] is True
    assert evaluation["behavior_novelty"]["status"] == "no_op"
    assert evaluation["behavior_novelty"]["stage"] == "validation"
    assert evaluation["behavior_novelty"]["gate_status"] == "failed"
    assert evaluation["score_components"]["behavior_novelty_status"] == "no_op"


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


def test_prepare_context_hides_final_blind_feedback_from_agent_memory() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit_blind_memory", run_id="unit_blind_memory_run", max_iterations=4)
    final_blind_status = {
        "selected": None,
        "promotion": {"promoted": False, "reason": "no finalist passed"},
        "finalists": [
            {
                "finalist": {
                    "iteration": 7,
                    "candidate_path": "artifacts/factor_strategy_loop/unit/iter_07/candidate.json",
                },
                "score": -2003001.0,
                "constraints_ok": False,
                "verification_status": "inconclusive",
                "lean_gate_status": VERIFICATION_FAILED,
                "lean_comparison_status": "partial",
                "promotion_eligible": False,
                "evaluation": {
                    "candidate": {
                        "candidate_type": "rank_profile",
                        "rank_profile": {"top_k": 3, "side_mode": "both"},
                    },
                    "metrics": {
                        "profit_pct": -0.55,
                        "max_drawdown_pct": 1.67,
                        "trades": 6,
                        "profit_over_max_drawdown": -0.33,
                    },
                    "lean_metrics": {
                        "total_return": 0.04,
                        "max_drawdown": 0.08,
                        "trades": 6,
                        "profit_over_max_drawdown": 0.5,
                    },
                    "violations": ["research: trades=6 < 7"],
                    "promotion_reason": "blind/verification/LEAN failed",
                },
            }
        ],
    }
    state = StrategyLoopState(run_id=cfg.run_id, iteration=9, final_blind_status=final_blind_status)
    save_checkpoint(cfg, state)

    context = prepare_context(cfg, cfg.run_id, 9)

    assert "final_blind_feedback" not in context["loop_memory"]
    assert "final_blind_feedback" not in json.dumps(strategy_loop_mod._compact_direct_agent_context(context), sort_keys=True)


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


def test_promotion_requires_passed_lean_gate_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    cfg = StrategyLoopConfig.from_args(
        tag="unit_promote_lean",
        run_id="unit_promote_lean_run",
        promote=True,
        lean_gate_mode="final",
    )
    iter_dir = tmp_path / "iter_01"
    iter_dir.mkdir()
    _write_json(iter_dir / "candidate.json", {"candidate_type": "rank_profile", "rank_profile": {"top_k": 2}})
    candidate = validate_candidate(iter_dir / "candidate.json")
    base_eval = {"constraints_ok": True, "score": 123.0, "promotion_reason": "passes hard gates"}

    missing = promote_candidate(candidate, base_eval, cfg, iter_dir=iter_dir)
    passed = promote_candidate(
        candidate,
        {**base_eval, "lean_gate": {"status": VERIFICATION_PASSED, "comparison_status": "ok"}},
        cfg,
        iter_dir=iter_dir,
    )

    assert missing["promoted"] is False
    assert "lean_gate_status=missing" in missing["reason"]
    assert passed["promoted"] is True


def test_lean_gate_drift_blocks_promotion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    _install_fake_lean_gate(monkeypatch, comparison_status="drift")
    cfg = StrategyLoopConfig.from_args(
        tag="unit_lean_drift",
        run_id="unit_lean_drift_run",
        promote=True,
        lean_gate_mode="final",
        lean_timeout=5,
    )
    idir = tmp_path / "iter_01"
    idir.mkdir()
    _write_json(idir / "backtest.json", {"signals": "unit"})
    candidate = {"candidate_type": "rank_profile", "rank_profile": {"top_k": 2}}
    evaluation = {"constraints_ok": True, "score": 10.0, "promotion_eligible": True, "promotion_reason": "passes hard gates"}

    gate = StrategyLoopRunner(cfg)._apply_lean_gate(idir, evaluation, stage="final", timerange=cfg.timerange)
    promoted = promote_candidate(candidate, evaluation, cfg, iter_dir=idir)

    assert gate["status"] == VERIFICATION_FAILED
    assert evaluation["promotion_eligible"] is False
    assert promoted["promoted"] is False
    assert "lean_gate_status=failed" in promoted["reason"]
    assert (idir / "lean_gate.json").exists()


def test_lean_gate_ok_allows_promotion_and_records_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    _install_fake_lean_gate(monkeypatch, comparison_status="ok")
    cfg = StrategyLoopConfig.from_args(
        tag="unit_lean_ok",
        run_id="unit_lean_ok_run",
        promote=True,
        lean_gate_mode="final",
        lean_timeout=5,
    )
    idir = tmp_path / "iter_01"
    idir.mkdir()
    _write_json(idir / "backtest.json", {"signals": "unit"})
    candidate = {"candidate_type": "rank_profile", "rank_profile": {"top_k": 2}}
    evaluation = {"constraints_ok": True, "score": 10.0, "promotion_eligible": True, "promotion_reason": "passes hard gates"}

    gate = StrategyLoopRunner(cfg)._apply_lean_gate(idir, evaluation, stage="final", timerange=cfg.timerange)
    promoted = promote_candidate(candidate, evaluation, cfg, iter_dir=idir)

    assert gate["status"] == VERIFICATION_PASSED
    assert evaluation["lean_comparison"]["status"] == "ok"
    assert promoted["promoted"] is True
    assert json.loads((idir / "lean_gate.json").read_text(encoding="utf-8"))["comparison_status"] == "ok"


def test_lean_gate_all_uses_validation_stage_artifact_for_triple_holdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    _install_fake_lean_gate(monkeypatch, comparison_status="ok")
    cfg = StrategyLoopConfig.from_args(
        tag="unit_lean_all",
        run_id="unit_lean_all_run",
        validation_protocol="triple_holdout",
        lean_gate_mode="all",
    )
    idir = tmp_path / "iter_01"
    idir.mkdir()
    _write_json(
        idir / "backtest.json",
        {
            "stages": {
                "search": {"signals": str(tmp_path / "search.feather"), "venue": "okx", "timeframe": "1h"},
                "validation": {"signals": str(tmp_path / "validation.feather"), "venue": "okx", "timeframe": "1h"},
            }
        },
    )
    evaluation = {"constraints_ok": True, "score": 10.0, "promotion_eligible": False, "promotion_reason": "validation only"}

    gate = StrategyLoopRunner(cfg)._apply_lean_gate(idir, evaluation, stage="iteration", timerange=cfg.validation_timerange)
    rank_artifact_path = repo_paths.resolve_repo_path(gate["artifacts"]["rank_artifact"])
    rank_artifact = json.loads(rank_artifact_path.read_text(encoding="utf-8"))

    assert gate["status"] == VERIFICATION_PASSED
    assert rank_artifact["signals"].endswith("validation.feather")
    assert gate["artifacts"]["rank_artifact"].endswith("lean_gate/iteration/rank_artifact.json")


def test_freqtrade_window_backtest_exports_stage_signals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = StrategyLoopConfig.from_args(
        tag="unit_freqtrade_stage",
        run_id="unit_freqtrade_stage_run",
        eval_mode="freqtrade",
    )
    runner = StrategyLoopRunner(cfg)
    captured: dict[str, object] = {}

    def fake_rank_export(**kwargs: object) -> dict:
        captured["rank_kwargs"] = kwargs
        return {"signals": {"all": str(tmp_path / "validation.feather")}, "exported": True}

    def fake_freqtrade(self, idir, research_result, timerange=None, stage="single") -> dict:
        captured["freqtrade_research_result"] = research_result
        return {"ok": True, "signal_dir": str(tmp_path)}

    monkeypatch.setattr(strategy_loop_mod, "_resolve_factor_state", lambda tag: (None, None))
    monkeypatch.setattr(strategy_loop_mod.rank_portfolio, "rank_export", fake_rank_export)
    monkeypatch.setattr(StrategyLoopRunner, "_run_fixed_freqtrade_backtest", fake_freqtrade)

    result = runner._run_window_backtest(
        tmp_path,
        {"candidate_type": "rank_profile", "rank_profile": {"top_k": 2}},
        stage="validation",
        timerange="20260301-20260331",
        run_freqtrade=True,
    )

    assert result["signals"]["all"].endswith("validation.feather")
    assert captured["freqtrade_research_result"]["signals"]["all"].endswith("validation.feather")
    assert result["freqtrade_backtest"] == {"ok": True, "signal_dir": str(tmp_path)}
    assert captured["rank_kwargs"]["tag"].endswith("_validation")


def test_fixed_freqtrade_backtest_accepts_signal_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    signal_dir = tmp_path / "signals"
    signal_dir.mkdir()
    signal_file = signal_dir / "all.feather"
    signal_file.write_bytes(b"unit")
    cfg = StrategyLoopConfig.from_args(tag="unit_signal_mapping", run_id="unit_signal_mapping_run")
    runner = StrategyLoopRunner(cfg)

    monkeypatch.setattr(strategy_loop_mod.repo_paths, "resolve_repo_path", lambda raw: tmp_path / "config.json")
    monkeypatch.setattr(strategy_loop_mod.repo_paths, "user_data_root", lambda: tmp_path / "user_data")

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    strategy_dir = tmp_path / "user_data" / "strategies"
    strategy_dir.mkdir(parents=True)
    (strategy_dir / f"{strategy_loop_mod.FIXED_FREQTRADE_STRATEGY}.py").write_text("# unit\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"strategy": {}}))

    monkeypatch.setattr(strategy_loop_mod.subprocess, "run", fake_run)

    result = runner._run_fixed_freqtrade_backtest(
        tmp_path,
        {"signals": {"all": str(signal_file)}},
        timerange="20260301-20260331",
        stage="validation",
    )

    assert result["ok"] is False
    assert str(result["error"]).startswith("no Freqtrade backtest result zip")
    assert result["signal_dir"].endswith("signals")
    assert "--timerange" in captured["cmd"]


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


def test_triple_holdout_records_validation_curve_regime_stability() -> None:
    cfg = StrategyLoopConfig.from_args(
        tag="unit_triple_regime_stability",
        validation_protocol="triple_holdout",
        eval_mode="freqtrade",
        score_mode="composite",
    )
    search = {
        "timerange": cfg.search_timerange,
        "total_return_pct": 20.0,
        "max_drawdown_pct": 5.0,
        "profit_over_max_drawdown": 4.0,
        "trades": 80,
    }
    validation = {
        "timerange": cfg.validation_timerange,
        "total_return_pct": 20.0,
        "max_drawdown_pct": 5.0,
        "profit_over_max_drawdown": 4.0,
        "trades": 40,
        "freqtrade_backtest": {
            "ok": True,
            "metrics": {"profit_pct": 20.0, "max_drawdown_pct": 5.0, "profit_over_max_drawdown": 4.0, "trades": 40},
        },
        "curve": [
            {"date": "2026-03-01T00:00:00Z", "equity": 1.00},
            {"date": "2026-03-07T00:00:00Z", "equity": 1.10},
            {"date": "2026-03-08T00:00:00Z", "equity": 1.10},
            {"date": "2026-03-14T00:00:00Z", "equity": 1.05},
            {"date": "2026-03-15T00:00:00Z", "equity": 1.05},
            {"date": "2026-03-21T00:00:00Z", "equity": 1.20},
        ],
    }

    evaluation = score_triple_holdout_backtest({"stages": {"search": search, "validation": validation}}, cfg)
    stability = evaluation["window_metrics"]["validation"]["regime_stability"]

    assert stability["subwindow_count"] == 3
    assert stability["positive_subwindows"] == 2
    assert stability["positive_subwindow_ratio"] == pytest.approx(2 / 3)
    assert stability["worst_subwindow"]["period"] == "2026-03-08..2026-03-14"
    assert stability["worst_subwindow_profit_pct"] == pytest.approx(-4.545455)
    assert evaluation["score_components"]["regime_stability_score"] == stability["score"]


def test_pareto_pool_uses_curve_regime_stability_axis_without_explicit_component() -> None:
    weak = {
        "run_id": "unit_regime_axis",
        "iteration": 1,
        "candidate_path": "iter_01/candidate.json",
        "candidate": {"candidate_type": "rank_profile", "name": "weak", "rank_profile": {"top_k": 2}},
        "score": 1.0,
        "score_components": {"composite_score": 1.0},
        "constraints_ok": True,
        "window_metrics": {
            "validation": {
                "regime_stability": {
                    "positive_subwindow_ratio": 0.5,
                    "worst_subwindow_profit_pct": -4.0,
                    "profit_std": 3.0,
                }
            }
        },
    }
    stable = {
        **weak,
        "iteration": 2,
        "candidate_path": "iter_02/candidate.json",
        "candidate": {"candidate_type": "rank_profile", "name": "stable", "rank_profile": {"top_k": 3}},
        "score_components": {"composite_score": 0.5},
        "window_metrics": {
            "validation": {
                "regime_stability": {
                    "positive_subwindow_ratio": 1.0,
                    "worst_subwindow_profit_pct": 1.0,
                    "profit_std": 0.5,
                }
            }
        },
    }

    pool = build_pareto_pool([weak, stable], size_per_axis=1)

    assert pool["axes"]["best_regime_stability"][0]["iteration"] == 2


def test_triple_holdout_search_window_scores_research_only_without_freqtrade_noise() -> None:
    cfg = StrategyLoopConfig.from_args(
        tag="unit_triple_search_research_only",
        validation_protocol="triple_holdout",
        eval_mode="two_stage",
        score_mode="composite",
    )
    search = {
        "timerange": cfg.search_timerange,
        "total_return_pct": 12.0,
        "max_drawdown_pct": 6.0,
        "profit_over_max_drawdown": 2.0,
        "trades": scaled_gate_values(cfg, cfg.search_timerange)["min_trades"] - 1,
    }

    evaluation = score_triple_holdout_backtest({"stages": {"search": search}}, cfg)
    search_window = evaluation["window_metrics"]["search"]

    assert evaluation["selected_window"] == "validation"
    assert evaluation["score"] > -1_000_000.0
    assert evaluation["constraints_ok"] is False
    assert search_window["violations"] == [f"trades={search['trades']} < {scaled_gate_values(cfg, cfg.search_timerange)['min_trades']}"]
    assert not any("freqtrade_backtest missing" in item for item in search_window["violations"])
    assert search_window["freqtrade_metrics"] == {}


def test_skipped_freqtrade_stage_reports_skip_reason_without_zero_metric_noise() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit_skipped_ft", eval_mode="two_stage", score_mode="composite")
    result = score_strategy_loop_backtest(
        {
            "total_return_pct": -5.0,
            "max_drawdown_pct": 8.0,
            "profit_over_max_drawdown": -0.625,
            "trades": 8,
            "freqtrade_backtest": {
                "ok": False,
                "skipped": True,
                "reason": "Validation research gates failed",
                "stage_a_violations": ["profit_over_max_drawdown=-0.625 < 1.2"],
            },
        },
        cfg,
    )

    assert "freqtrade: Validation research gates failed" in result["violations"]
    assert not any("freqtrade_backtest missing" in item for item in result["violations"])
    assert not any("freqtrade: trades=0" in item for item in result["violations"])


def test_rank_profile_repair_queue_generates_untried_trade_gate_repairs(tmp_path: Path) -> None:
    baseline = {
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 2,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.51,
        "rebalance_hours": 6,
        "risk_per_trade": 0.01785,
        "leverage_cap": 5.0,
        "short_max_mom_24h": 0.038,
        "short_max_market_mom_24h": 0.05,
        "short_exit_mom_24h": 0.04,
        "short_exit_market_mom_24h": 0.04,
        "max_entry_atr_pct": 0.05,
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_repair_queue",
        run_id="unit_repair_queue_run",
        validation_protocol="triple_holdout",
        score_mode="composite",
        baseline_profile=str(tmp_path / "optimized_profile.json"),
    )
    _write_json(
        tmp_path / "optimized_profile.json",
        {
            "candidate_state": baseline["candidate_state"],
            "selection": {"recompute_corr": False, "n": 50},
            "risk": baseline,
        },
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 1,
            "candidate": {"candidate_type": "rank_profile", "name": "baseline", "rank_profile": baseline},
            "parameter_signature": rank_profile_signature(baseline),
            "window_metrics": {
                "search": {
                    "research_metrics": {
                        "profit_pct": 32.0,
                        "max_drawdown_pct": 8.0,
                        "profit_over_max_drawdown": 4.0,
                        "trades": scaled_gate_values(cfg, cfg.search_timerange)["min_trades"] - 4,
                    }
                }
            },
        }
    ]

    queue = build_rank_profile_repair_queue(baseline, cfg, rows=rows)
    runner = StrategyLoopRunner(cfg)
    runner.state.iteration = 2
    runner.state.score_history = rows
    idir = tmp_path / "iter_02"
    idir.mkdir()

    assert queue
    assert queue[0]["metadata"]["source"] == "controller_rank_profile_repair_queue"
    assert queue[0]["rank_profile"]["candidate_state"] == baseline["candidate_state"]
    assert queue[0]["rank_profile"]["recompute_corr"] is False
    assert queue[0]["rank_profile"]["min_abs_score_z"] < baseline["min_abs_score_z"]
    assert runner._seed_rank_profile_repair_candidate(idir, idir / "candidate.json") is True
    candidate = validate_candidate(idir / "candidate.json")
    assert candidate["metadata"]["source"] == "controller_rank_profile_repair_queue"
    assert candidate["rank_profile"]["min_abs_score_z"] < baseline["min_abs_score_z"]


def test_rank_profile_repair_queue_anchors_near_pdd_misses(tmp_path: Path) -> None:
    baseline = {
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 2,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.51,
        "rebalance_hours": 6,
        "risk_per_trade": 0.01785,
        "leverage_cap": 5.0,
        "short_max_mom_24h": 0.038,
        "max_entry_atr_pct": 0.05,
    }
    anchor = {
        **baseline,
        "top_k": 3,
        "min_abs_score_z": 1.50,
        "risk_per_trade": 0.013,
        "leverage_cap": 4.0,
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_near_pdd_repair",
        run_id="unit_near_pdd_repair_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "optimized_profile.json"),
    )
    _write_json(
        tmp_path / "optimized_profile.json",
        {
            "candidate_state": baseline["candidate_state"],
            "selection": {"recompute_corr": False, "n": 50},
            "risk": baseline,
        },
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 10,
            "candidate": {"candidate_type": "rank_profile", "name": "near_pdd", "rank_profile": anchor},
            "parameter_signature": rank_profile_signature(anchor),
            "research_metrics": {
                "profit_pct": 13.4,
                "max_drawdown_pct": 11.5,
                "profit_over_max_drawdown": 1.17,
                "trades": scaled_gate_values(cfg, cfg.search_timerange)["min_trades"] + 20,
            },
        }
    ]

    queue = build_rank_profile_repair_queue(baseline, cfg, rows=rows)

    assert queue[0]["metadata"]["source"] == "controller_rank_profile_near_pdd_repair"
    assert queue[0]["metadata"]["parent_anchor"] == "iteration_10"
    assert queue[0]["rank_profile"]["top_k"] == 3
    assert queue[0]["rank_profile"]["min_abs_score_z"] > anchor["min_abs_score_z"]


def test_rank_profile_repair_queue_self_anchors_when_baseline_missing(tmp_path: Path) -> None:
    anchor = {
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 3,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.47,
        "rebalance_hours": 6,
        "risk_per_trade": 0.013,
        "leverage_cap": 4.0,
        "short_max_mom_24h": 0.038,
        "max_entry_atr_pct": 0.05,
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_missing_baseline_near_pdd",
        run_id="unit_missing_baseline_near_pdd_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 10,
            "candidate": {"candidate_type": "rank_profile", "name": "near_pdd", "rank_profile": anchor},
            "parameter_signature": rank_profile_signature(anchor),
            "research_metrics": {
                "profit_pct": 13.4,
                "max_drawdown_pct": 11.5,
                "profit_over_max_drawdown": 1.17,
                "trades": scaled_gate_values(cfg, cfg.search_timerange)["min_trades"] + 20,
            },
        }
    ]

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)
    runner = StrategyLoopRunner(cfg)
    runner.state.iteration = 11
    runner.state.score_history = rows
    idir = tmp_path / "iter_11"
    idir.mkdir()

    assert queue
    assert queue[0]["metadata"]["source"] == "controller_rank_profile_near_pdd_repair"
    assert queue[0]["metadata"]["parent_anchor"] == "iteration_10"
    assert queue[0]["rank_profile"]["candidate_state"] == anchor["candidate_state"]
    assert runner._seed_rank_profile_repair_candidate(idir, idir / "candidate.json") is True
    candidate = validate_candidate(idir / "candidate.json")
    assert candidate["metadata"]["source"] == "controller_rank_profile_near_pdd_repair"
    assert candidate["metadata"]["parent_anchor"] == "iteration_10"
    assert candidate["rank_profile"]["min_abs_score_z"] > anchor["min_abs_score_z"]
    assert "Parent: iteration_10" in (idir / "analysis.md").read_text(encoding="utf-8")


def test_rank_profile_repair_queue_self_anchors_search_trade_near_misses(tmp_path: Path) -> None:
    anchor = {
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 2,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.54,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 3.0,
        "short_max_mom_24h": 0.038,
        "max_entry_atr_pct": 0.05,
        "regime_mode": "hq",
        "regime_min_edge_ic": 0.01,
        "regime_min_pair_edge_ic": 0.01,
        "regime_min_pair_count": 3,
        "regime_short_max_market_mom_24h": 0.03,
        "regime_max_market_atr_pct": 0.04,
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_missing_baseline_search_trade",
        run_id="unit_missing_baseline_search_trade_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 43,
            "candidate": {"candidate_type": "rank_profile", "name": "search_trade_near_miss", "rank_profile": anchor},
            "parameter_signature": rank_profile_signature(anchor),
            "window_metrics": {
                "search": {
                    "constraints_ok": False,
                    "research_metrics": {
                        "profit_pct": 37.3,
                        "max_drawdown_pct": 6.6,
                        "profit_over_max_drawdown": 5.7,
                        "trades": scaled_gate_values(cfg, cfg.search_timerange)["min_trades"] - 3,
                    },
                    "violations": ["trades=51 < 54"],
                }
            },
        }
    ]

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)
    runner = StrategyLoopRunner(cfg)
    runner.state.iteration = 44
    runner.state.score_history = rows
    idir = tmp_path / "iter_44"
    idir.mkdir()

    assert queue
    assert queue[0]["metadata"]["source"] == "controller_rank_profile_search_trade_repair"
    assert queue[0]["metadata"]["parent_anchor"] == "iteration_43"
    assert queue[0]["metadata"]["hypothesis_family"] == "search_trade_topk_repair"
    assert queue[0]["rank_profile"]["top_k"] == 3
    assert queue[0]["rank_profile"]["min_abs_score_z"] == anchor["min_abs_score_z"]
    assert runner._seed_rank_profile_repair_candidate(idir, idir / "candidate.json") is True
    candidate = validate_candidate(idir / "candidate.json")
    assert candidate["metadata"]["source"] == "controller_rank_profile_search_trade_repair"
    assert candidate["metadata"]["parent_anchor"] == "iteration_43"


def test_rank_profile_repair_queue_defers_search_trade_after_repeated_validation_losses(tmp_path: Path) -> None:
    base_profile = {
        "n": 50,
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 1,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.51,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 3.0,
        "short_max_mom_24h": 0.038,
        "max_entry_atr_pct": 0.05,
        "regime_mode": "hq",
        "regime_min_edge_ic": 0.01,
        "regime_min_pair_edge_ic": 0.01,
        "regime_min_pair_count": 3,
        "regime_short_max_market_mom_24h": 0.03,
        "regime_max_market_atr_pct": 0.04,
    }
    search_near_miss = {**base_profile, "min_abs_score_z": 1.54}
    cfg = StrategyLoopConfig.from_args(
        tag="unit_validation_loss_defers_search_trade",
        run_id="unit_validation_loss_defers_search_trade_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 44,
            "candidate": {
                "candidate_type": "rank_profile",
                "name": "search_trade_near_miss",
                "rank_profile": search_near_miss,
            },
            "parameter_signature": rank_profile_signature(search_near_miss),
            "window_metrics": {
                "search": {
                    "constraints_ok": False,
                    "research_metrics": {
                        "profit_pct": 39.5,
                        "max_drawdown_pct": 6.5,
                        "profit_over_max_drawdown": 6.1,
                        "trades": scaled_gate_values(cfg, cfg.search_timerange)["min_trades"] - 2,
                    },
                    "violations": ["trades=52 < 54"],
                }
            },
        }
    ]
    for iteration, validation_profit, search_pdd in ((41, -2.8, 5.0), (46, -2.87, 5.8), (49, -1.81, 6.4)):
        row_profile = {**base_profile, "min_abs_score_z": 1.48 + iteration / 1000.0}
        rows.append(
            {
                "run_id": cfg.run_id,
                "iteration": iteration,
                "candidate": {
                    "candidate_type": "rank_profile",
                    "name": f"validation_loss_{iteration}",
                    "rank_profile": row_profile,
                },
                "parameter_signature": rank_profile_signature(row_profile),
                "window_metrics": {
                    "search": {
                        "constraints_ok": True,
                        "research_metrics": {
                            "profit_pct": 35.0 + iteration / 10.0,
                            "max_drawdown_pct": 6.0,
                            "profit_over_max_drawdown": search_pdd,
                            "trades": 58,
                        },
                        "violations": [],
                    },
                    "validation": {
                        "constraints_ok": False,
                        "research_metrics": {
                            "profit_pct": validation_profit,
                            "max_drawdown_pct": 5.5,
                            "profit_over_max_drawdown": validation_profit / 5.5,
                            "trades": scaled_gate_values(cfg, cfg.validation_timerange)["min_trades"] - 3,
                        },
                        "violations": ["research: validation loss after search pass"],
                    },
                },
            }
        )

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)

    assert queue
    assert queue[0]["metadata"]["source"] == "controller_rank_profile_validation_repair"
    assert queue[0]["metadata"]["hypothesis_family"] == "validation_factor_subset_repair"
    assert queue[0]["metadata"]["parent_anchor"] == "iteration_49"


def test_rank_profile_repair_queue_recovers_exclusion_search_trade_near_miss(tmp_path: Path) -> None:
    exclusion_anchor = {
        "n": 50,
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 3,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.52,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 3.0,
        "short_max_mom_24h": 0.038,
        "max_entry_atr_pct": 0.05,
        "exclude_pairs": ["BTC/USDT"],
        "regime_mode": "hq",
        "regime_min_edge_ic": 0.01,
        "regime_min_pair_edge_ic": 0.01,
        "regime_min_pair_count": 3,
        "regime_short_max_market_mom_24h": 0.03,
        "regime_max_market_atr_pct": 0.04,
    }
    validation_anchor = {k: v for k, v in exclusion_anchor.items() if k != "exclude_pairs"}
    validation_anchor["top_k"] = 2
    cfg = StrategyLoopConfig.from_args(
        tag="unit_exclusion_search_trade_repair",
        run_id="unit_exclusion_search_trade_repair_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 55,
            "candidate": {
                "candidate_type": "rank_profile",
                "name": "validation_exclude_btc_near_miss",
                "rank_profile": exclusion_anchor,
            },
            "parameter_signature": rank_profile_signature(exclusion_anchor),
            "window_metrics": {
                "search": {
                    "constraints_ok": False,
                    "research_metrics": {
                        "profit_pct": 23.4,
                        "max_drawdown_pct": 5.8,
                        "profit_over_max_drawdown": 4.0,
                        "trades": scaled_gate_values(cfg, cfg.search_timerange)["min_trades"] - 1,
                    },
                    "violations": ["trades=53 < 54"],
                }
            },
        }
    ]
    for iteration, validation_profit in ((46, -2.8), (49, -1.8), (50, -1.7)):
        row_profile = {**validation_anchor, "min_abs_score_z": 1.48 + iteration / 1000.0}
        rows.append(
            {
                "run_id": cfg.run_id,
                "iteration": iteration,
                "candidate": {
                    "candidate_type": "rank_profile",
                    "name": f"validation_loss_{iteration}",
                    "rank_profile": row_profile,
                },
                "parameter_signature": rank_profile_signature(row_profile),
                "window_metrics": {
                    "search": {
                        "constraints_ok": True,
                        "research_metrics": {
                            "profit_pct": 35.0,
                            "max_drawdown_pct": 6.0,
                            "profit_over_max_drawdown": 5.0 + iteration / 100.0,
                            "trades": 56,
                        },
                        "violations": [],
                    },
                    "validation": {
                        "constraints_ok": False,
                        "research_metrics": {
                            "profit_pct": validation_profit,
                            "max_drawdown_pct": 5.0,
                            "profit_over_max_drawdown": validation_profit / 5.0,
                            "trades": scaled_gate_values(cfg, cfg.validation_timerange)["min_trades"] - 3,
                        },
                        "violations": ["research: validation loss after search pass"],
                    },
                },
            }
        )

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)

    assert queue
    assert queue[0]["metadata"]["source"] == "controller_rank_profile_search_trade_repair"
    assert queue[0]["metadata"]["parent_anchor"] == "iteration_55"
    assert queue[0]["rank_profile"]["exclude_pairs"] == ["BTC/USDT"]
    assert queue[0]["rank_profile"]["top_k"] == 4


def test_rank_profile_repair_queue_prioritizes_positive_validation_trade_gap(tmp_path: Path) -> None:
    anchor = {
        "n": 50,
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 3,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.51,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 3.0,
        "short_max_mom_24h": 0.038,
        "max_entry_atr_pct": 0.05,
        "exclude_pairs": ["BTC/USDT"],
        "regime_mode": "hq",
        "regime_min_edge_ic": 0.01,
        "regime_min_pair_edge_ic": 0.01,
        "regime_min_pair_count": 3,
        "regime_short_max_market_mom_24h": 0.03,
        "regime_max_market_atr_pct": 0.04,
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_positive_validation_trade_gap",
        run_id="unit_positive_validation_trade_gap_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    rows = []
    for iteration, validation_profit in ((46, -2.8), (49, -1.8), (50, -1.7)):
        row_profile = {**anchor, "exclude_pairs": [], "min_abs_score_z": 1.48 + iteration / 1000.0}
        rows.append(
            {
                "run_id": cfg.run_id,
                "iteration": iteration,
                "candidate": {
                    "candidate_type": "rank_profile",
                    "name": f"validation_loss_{iteration}",
                    "rank_profile": row_profile,
                },
                "parameter_signature": rank_profile_signature(row_profile),
                "window_metrics": {
                    "search": {
                        "constraints_ok": True,
                        "research_metrics": {
                            "profit_pct": 35.0,
                            "max_drawdown_pct": 6.0,
                            "profit_over_max_drawdown": 5.0 + iteration / 100.0,
                            "trades": 56,
                        },
                        "violations": [],
                    },
                    "validation": {
                        "constraints_ok": False,
                        "research_metrics": {
                            "profit_pct": validation_profit,
                            "max_drawdown_pct": 5.0,
                            "profit_over_max_drawdown": validation_profit / 5.0,
                            "trades": scaled_gate_values(cfg, cfg.validation_timerange)["min_trades"] - 3,
                        },
                        "violations": ["research: validation loss after search pass"],
                    },
                },
            }
        )
    rows.append(
        {
            "run_id": cfg.run_id,
            "iteration": 60,
            "candidate": {
                "candidate_type": "rank_profile",
                "name": "validation_positive_undertraded",
                "rank_profile": anchor,
            },
            "parameter_signature": rank_profile_signature(anchor),
            "window_metrics": {
                "search": {
                    "constraints_ok": True,
                    "research_metrics": {
                        "profit_pct": 24.6,
                        "max_drawdown_pct": 5.8,
                        "profit_over_max_drawdown": 4.2,
                        "trades": 54,
                    },
                    "violations": [],
                },
                "validation": {
                    "constraints_ok": False,
                    "research_metrics": {
                        "profit_pct": 0.78,
                        "max_drawdown_pct": 3.3,
                        "profit_over_max_drawdown": 0.236,
                        "trades": scaled_gate_values(cfg, cfg.validation_timerange)["min_trades"] - 6,
                    },
                    "violations": ["research: trades=13 < 19"],
                },
            },
        }
    )
    search_near_miss = {**anchor, "min_abs_score_z": 1.52}
    rows.append(
        {
            "run_id": cfg.run_id,
            "iteration": 62,
            "candidate": {
                "candidate_type": "rank_profile",
                "name": "search_profitable_undertraded",
                "rank_profile": search_near_miss,
            },
            "parameter_signature": rank_profile_signature(search_near_miss),
            "window_metrics": {
                "search": {
                    "constraints_ok": False,
                    "research_metrics": {
                        "profit_pct": 24.5,
                        "max_drawdown_pct": 5.5,
                        "profit_over_max_drawdown": 4.4,
                        "trades": scaled_gate_values(cfg, cfg.search_timerange)["min_trades"] - 1,
                    },
                    "violations": ["research: trades below gate"],
                }
            },
        }
    )

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)

    assert queue
    assert queue[0]["metadata"]["hypothesis_family"] == "validation_exit_filter_repair_after_positive_validation"
    assert queue[0]["metadata"]["parent_anchor"] == "iteration_60"
    assert queue[0]["rank_profile"]["exclude_pairs"] == ["BTC/USDT"]
    assert queue[0]["rank_profile"]["short_exit_mom_24h"] == 0.0
    assert any(
        item["metadata"]["hypothesis_family"] == "validation_trade_repair_after_regime"
        and item["metadata"]["parent_anchor"] == "iteration_60"
        and item["rank_profile"]["top_k"] == 4
        for item in queue
    )


def test_rank_profile_repair_queue_adds_search_loser_exit_trade_repair(tmp_path: Path) -> None:
    import pandas as pd

    signal_dir = tmp_path / "search_signals"
    signal_dir.mkdir()
    dates = pd.date_range("2026-02-01", periods=5, freq="1h", tz="UTC")
    prices = {
        "SOL/USDT": [100.0, 102.0, 104.0, 106.0, 108.0],
        "ADA/USDT": [100.0, 99.0, 98.0, 97.0, 96.0],
    }
    rows = []
    for pair, values in prices.items():
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "pair": pair,
                    "open": values[idx],
                    "high": values[idx] * 1.01,
                    "low": values[idx] * 0.99,
                    "close": values[idx],
                    "rp_target_weight": -1.0 if idx < 3 else 0.0,
                    "rp_stop_pct": 0.2,
                }
            )
    pd.DataFrame(rows).to_feather(signal_dir / "all.feather")
    anchor = {
        "n": 50,
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 5,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.52,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 3.0,
        "short_max_mom_24h": 0.042,
        "short_exit_mom_24h": 0.0,
        "max_entry_atr_pct": 0.05,
        "exclude_pairs": ["BTC/USDT"],
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_search_loser_exit_trade_repair",
        run_id="unit_search_loser_exit_trade_repair_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    row = {
        "run_id": cfg.run_id,
        "iteration": 73,
        "candidate": {
            "candidate_type": "rank_profile",
            "name": "positive_validation_undertraded_exit_anchor",
            "rank_profile": anchor,
        },
        "parameter_signature": rank_profile_signature(anchor),
        "window_metrics": {
            "search": {
                "constraints_ok": True,
                "signal_dir": str(signal_dir),
                "research_metrics": {
                    "profit_pct": 8.5,
                    "max_drawdown_pct": 5.4,
                    "profit_over_max_drawdown": 1.56,
                    "trades": 72,
                },
                "violations": [],
            },
            "validation": {
                "constraints_ok": False,
                "research_metrics": {
                    "profit_pct": 1.46,
                    "max_drawdown_pct": 0.8,
                    "profit_over_max_drawdown": 1.82,
                    "trades": scaled_gate_values(cfg, cfg.validation_timerange)["min_trades"] - 4,
                },
                "violations": ["research: trades=15 < 19"],
            },
        },
    }

    queue = build_rank_profile_repair_queue({}, cfg, rows=[row])

    assert queue
    assert queue[0]["metadata"]["hypothesis_family"] == "validation_trade_search_loss_exclusion_repair"
    assert queue[0]["metadata"]["parent_anchor"] == "iteration_73"
    assert queue[0]["rank_profile"]["short_exit_mom_24h"] == -0.02
    assert queue[0]["rank_profile"]["min_abs_score_z"] == 1.45
    assert queue[0]["rank_profile"]["exclude_pairs"] == ["BTC/USDT", "SOL/USDT"]


def test_rank_profile_repair_queue_adds_validation_pair_exclusion_repairs(tmp_path: Path) -> None:
    import pandas as pd

    signal_dir = tmp_path / "signals"
    signal_dir.mkdir()
    dates = pd.date_range("2026-03-01", periods=5, freq="1h", tz="UTC")
    prices = {
        "BTC/USDT": [100.0, 102.0, 104.0, 106.0, 108.0],
        "ADA/USDT": [100.0, 99.0, 98.0, 97.0, 96.0],
        "ETH/USDT": [100.0, 100.0, 100.0, 100.0, 100.0],
    }
    signal_rows = []
    for pair, values in prices.items():
        for idx, date in enumerate(dates):
            weight = -1.0 if pair in {"BTC/USDT", "ADA/USDT"} and idx < 3 else 0.0
            price = values[idx]
            signal_rows.append(
                {
                    "date": date,
                    "pair": pair,
                    "open": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "close": price,
                    "rp_target_weight": weight,
                    "rp_stop_pct": 0.2,
                }
            )
    pd.DataFrame(signal_rows).to_feather(signal_dir / "all.feather")

    anchor = {
        "n": 50,
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 2,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.52,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 3.0,
        "short_max_mom_24h": 0.038,
        "max_entry_atr_pct": 0.05,
        "regime_mode": "hq",
        "regime_min_edge_ic": 0.01,
        "regime_min_pair_edge_ic": 0.01,
        "regime_min_pair_count": 3,
        "regime_short_max_market_mom_24h": 0.03,
        "regime_max_market_atr_pct": 0.04,
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_validation_pair_exclusion_repair",
        run_id="unit_validation_pair_exclusion_repair_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 50,
            "candidate": {"candidate_type": "rank_profile", "name": "validation_pair_loss", "rank_profile": anchor},
            "parameter_signature": rank_profile_signature(anchor),
            "window_metrics": {
                "search": {
                    "constraints_ok": True,
                    "research_metrics": {
                        "profit_pct": 36.0,
                        "max_drawdown_pct": 6.0,
                        "profit_over_max_drawdown": 6.0,
                        "trades": 56,
                    },
                    "violations": [],
                },
                "validation": {
                    "constraints_ok": False,
                    "research_metrics": {
                        "profit_pct": -2.0,
                        "max_drawdown_pct": 5.0,
                        "profit_over_max_drawdown": -0.4,
                        "trades": 16,
                    },
                    "signal_dir": str(signal_dir),
                    "violations": ["research: validation loss after search pass"],
                },
            },
        }
    ]

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)

    assert queue
    assert queue[0]["metadata"]["hypothesis_family"] == "validation_pair_exclusion_repair"
    assert queue[0]["metadata"]["search_mode"] == "structured_explore"
    assert queue[0]["rank_profile"]["top_k"] == 3
    assert queue[0]["rank_profile"]["exclude_pairs"] == ["BTC/USDT"]


def test_rank_profile_repair_queue_adds_validation_pass_pair_robustness_repairs(tmp_path: Path) -> None:
    import pandas as pd

    signal_dir = tmp_path / "signals"
    signal_dir.mkdir()
    dates = pd.date_range("2026-03-01", periods=5, freq="1h", tz="UTC")
    prices = {
        "LINK/USDT": [100.0, 104.0, 108.0, 112.0, 116.0],
        "ADA/USDT": [100.0, 99.0, 98.0, 97.0, 96.0],
    }
    signal_rows = []
    for pair, values in prices.items():
        for idx, date in enumerate(dates):
            signal_rows.append(
                {
                    "date": date,
                    "pair": pair,
                    "open": values[idx],
                    "high": values[idx] * 1.001,
                    "low": values[idx] * 0.999,
                    "close": values[idx],
                    "rp_target_weight": -1.0 if idx < 3 else 0.0,
                    "rp_stop_pct": 0.2,
                }
            )
    pd.DataFrame(signal_rows).to_feather(signal_dir / "all.feather")

    anchor = {
        "n": 50,
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 5,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.45,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 3.0,
        "short_max_mom_24h": 0.042,
        "short_exit_mom_24h": -0.02,
        "max_entry_atr_pct": 0.05,
        "exclude_pairs": ["BTC/USDT", "SOL/USDT"],
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_validation_pass_pair_robustness_repair",
        run_id="unit_validation_pass_pair_robustness_repair_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 1,
            "candidate": {"candidate_type": "rank_profile", "name": "validation_passed", "rank_profile": anchor},
            "parameter_signature": rank_profile_signature(anchor),
            "window_metrics": {
                "search": {
                    "constraints_ok": True,
                    "research_metrics": {
                        "profit_pct": 5.0,
                        "max_drawdown_pct": 3.0,
                        "profit_over_max_drawdown": 1.66,
                        "trades": 80,
                    },
                    "violations": [],
                },
                "validation": {
                    "constraints_ok": True,
                    "research_metrics": {
                        "profit_pct": 1.8,
                        "max_drawdown_pct": 0.8,
                        "profit_over_max_drawdown": 2.25,
                        "trades": 20,
                    },
                    "signal_dir": str(signal_dir),
                    "violations": [],
                },
            },
        }
    ]

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)

    assert queue
    assert queue[0]["metadata"]["source"] == "controller_rank_profile_validation_pass_robustness_repair"
    assert queue[0]["metadata"]["parent_anchor"] == "iteration_1"
    assert queue[0]["rank_profile"]["top_k"] == 6
    assert queue[0]["rank_profile"]["exclude_pairs"] == ["BTC/USDT", "SOL/USDT", "LINK/USDT"]


def test_rank_profile_repair_queue_prioritizes_validation_pass_activity_repairs(tmp_path: Path) -> None:
    import pandas as pd

    signal_dir = tmp_path / "sparse_validation_signals"
    signal_dir.mkdir()
    dates = pd.date_range("2026-03-01", periods=20, freq="1D", tz="UTC")
    signal_rows = []
    for pair in ("ADA/USDT", "DOT/USDT"):
        for idx, date in enumerate(dates):
            price = 100.0 + idx
            signal_rows.append(
                {
                    "date": date,
                    "pair": pair,
                    "open": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "close": price,
                    "rp_target_weight": -1.0 if idx < 3 else 0.0,
                    "rp_stop_pct": 0.2,
                }
            )
    pd.DataFrame(signal_rows).to_feather(signal_dir / "all.feather")

    anchor = {
        "n": 50,
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 5,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.45,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 3.0,
        "short_max_mom_24h": 0.042,
        "short_exit_mom_24h": -0.02,
        "max_entry_atr_pct": 0.05,
        "exclude_pairs": ["BTC/USDT", "SOL/USDT"],
        "regime_mode": "hq",
        "regime_min_edge_ic": 0.01,
        "regime_min_pair_edge_ic": 0.01,
        "regime_min_pair_count": 3,
        "regime_short_max_market_mom_24h": 0.03,
        "regime_max_market_atr_pct": 0.04,
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_validation_pass_activity_repair",
        run_id="unit_validation_pass_activity_repair_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 7,
            "candidate": {
                "candidate_type": "rank_profile",
                "name": "validation_passed_sparse",
                "rank_profile": anchor,
            },
            "parameter_signature": rank_profile_signature(anchor),
            "window_metrics": {
                "search": {
                    "constraints_ok": True,
                    "research_metrics": {
                        "profit_pct": 5.0,
                        "max_drawdown_pct": 3.0,
                        "profit_over_max_drawdown": 1.66,
                        "trades": 80,
                    },
                    "violations": [],
                },
                "validation": {
                    "constraints_ok": True,
                    "research_metrics": {
                        "profit_pct": 1.8,
                        "max_drawdown_pct": 0.8,
                        "profit_over_max_drawdown": 2.25,
                        "trades": 20,
                    },
                    "signal_dir": str(signal_dir),
                    "violations": [],
                },
            },
        }
    ]

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)

    assert queue
    assert queue[0]["metadata"]["source"] == "controller_rank_profile_validation_pass_activity_repair"
    assert queue[0]["metadata"]["parent_anchor"] == "iteration_7"
    assert queue[0]["metadata"]["hypothesis_family"] == "validation_activity_regime_coverage_repair"
    assert queue[0]["metadata"]["validation_activity_summary"]["active_days"] == 3
    assert queue[0]["metadata"]["validation_activity_summary"]["total_days"] == 20
    assert queue[0]["rank_profile"]["regime_min_pair_count"] == 2


def test_rank_profile_repair_queue_prioritizes_validation_pass_stability_repairs(tmp_path: Path) -> None:
    anchor = {
        "n": 50,
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 5,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.45,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 3.0,
        "short_max_mom_24h": 0.042,
        "short_exit_mom_24h": 0.04,
        "short_exit_market_mom_24h": 0.04,
        "max_entry_atr_pct": 0.05,
        "regime_mode": "hq",
        "regime_min_edge_ic": 0.01,
        "regime_min_pair_edge_ic": 0.01,
        "regime_min_pair_count": 3,
        "regime_short_max_market_mom_24h": 0.03,
        "regime_max_market_atr_pct": 0.04,
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_validation_pass_stability_repair",
        run_id="unit_validation_pass_stability_repair_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 8,
            "candidate": {
                "candidate_type": "rank_profile",
                "name": "validation_passed_unstable",
                "rank_profile": anchor,
            },
            "parameter_signature": rank_profile_signature(anchor),
            "window_metrics": {
                "search": {
                    "constraints_ok": True,
                    "research_metrics": {
                        "profit_pct": 10.0,
                        "max_drawdown_pct": 4.0,
                        "profit_over_max_drawdown": 2.5,
                        "trades": 80,
                    },
                    "violations": [],
                },
                "validation": {
                    "constraints_ok": True,
                    "research_metrics": {
                        "profit_pct": 4.0,
                        "max_drawdown_pct": 2.0,
                        "profit_over_max_drawdown": 2.0,
                        "trades": 20,
                    },
                    "regime_stability": {
                        "subwindow_count": 4,
                        "positive_subwindows": 2,
                        "positive_subwindow_ratio": 0.5,
                        "worst_subwindow_profit_pct": -3.2,
                        "profit_std": 4.1,
                        "max_subwindow_drawdown_pct": 6.5,
                        "worst_subwindow": {"period": "2026-03-15..2026-03-21", "profit_pct": -3.2},
                    },
                    "violations": [],
                },
            },
        }
    ]

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)

    assert queue
    assert queue[0]["metadata"]["source"] == "controller_rank_profile_validation_pass_stability_repair"
    assert queue[0]["metadata"]["parent_anchor"] == "iteration_8"
    assert queue[0]["metadata"]["hypothesis_family"] == "validation_subwindow_tail_exit_repair"
    assert queue[0]["metadata"]["validation_regime_stability"]["worst_subwindow_profit_pct"] == -3.2
    assert queue[0]["rank_profile"]["short_exit_mom_24h"] == 0.0


def test_rank_profile_repair_queue_skips_behavior_duplicate_activity_repairs(tmp_path: Path) -> None:
    import pandas as pd

    signal_dir = tmp_path / "sparse_validation_signals"
    signal_dir.mkdir()
    dates = pd.date_range("2026-03-01", periods=20, freq="1D", tz="UTC")
    rows_out = []
    for pair in ("ADA/USDT", "DOT/USDT"):
        for idx, date in enumerate(dates):
            rows_out.append({"date": date, "pair": pair, "rp_target_weight": -1.0 if idx < 3 else 0.0})
    pd.DataFrame(rows_out).to_feather(signal_dir / "all.feather")

    anchor = {
        "n": 50,
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 5,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.45,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 3.0,
        "short_max_mom_24h": 0.042,
        "short_exit_mom_24h": -0.02,
        "max_entry_atr_pct": 0.05,
        "exclude_pairs": ["BTC/USDT", "SOL/USDT"],
        "regime_mode": "hq",
        "regime_min_edge_ic": 0.01,
        "regime_min_pair_edge_ic": 0.01,
        "regime_min_pair_count": 4,
        "regime_short_max_market_mom_24h": 0.03,
        "regime_max_market_atr_pct": 0.04,
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_duplicate_activity_repair",
        run_id="unit_duplicate_activity_repair_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 7,
            "candidate": {"candidate_type": "rank_profile", "name": "validation_passed_sparse", "rank_profile": anchor},
            "parameter_signature": rank_profile_signature(anchor),
            "window_metrics": {
                "search": {"constraints_ok": True, "research_metrics": {"profit_pct": 5.0, "max_drawdown_pct": 3.0, "profit_over_max_drawdown": 1.66, "trades": 80}},
                "validation": {
                    "constraints_ok": True,
                    "research_metrics": {"profit_pct": 1.8, "max_drawdown_pct": 0.8, "profit_over_max_drawdown": 2.25, "trades": 20},
                    "signal_dir": str(signal_dir),
                    "violations": [],
                },
            },
        },
        {
            "run_id": cfg.run_id,
            "iteration": 8,
            "candidate": {
                "candidate_type": "rank_profile",
                "name": "duplicate_activity",
                "rank_profile": {**anchor, "regime_min_pair_count": 3},
                "metadata": {
                    "source": "controller_rank_profile_validation_pass_activity_repair",
                    "hypothesis_family": "validation_activity_regime_coverage_repair",
                    "changed_keys": ["regime_min_pair_count"],
                },
            },
            "parameter_signature": rank_profile_signature({**anchor, "regime_min_pair_count": 3}),
            "behavior_novelty": {"status": "duplicate"},
        },
    ]

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)

    assert queue
    assert all(
        item["metadata"].get("changed_keys") != ["regime_min_pair_count"]
        or item["metadata"].get("source") != "controller_rank_profile_validation_pass_activity_repair"
        for item in queue
    )


def test_rank_profile_repair_queue_infers_duplicate_signal_repairs(tmp_path: Path) -> None:
    prior_dir = tmp_path / "prior_validation"
    duplicate_dir = tmp_path / "duplicate_validation"
    _write_signal_behavior_fixture(prior_dir, weight=-1.0)
    _write_signal_behavior_fixture(duplicate_dir, weight=-0.5)
    anchor = {
        "n": 50,
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 5,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.5,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 3.0,
        "short_max_mom_24h": 0.034,
        "short_exit_mom_24h": -0.02,
        "max_entry_atr_pct": 0.05,
        "regime_mode": "hq",
        "regime_min_edge_ic": 0.01,
        "regime_min_pair_edge_ic": 0.01,
        "regime_min_pair_count": 3,
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_inferred_duplicate_repair",
        run_id="unit_inferred_duplicate_repair_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    validation_metrics = {
        "profit_pct": 1.1,
        "max_drawdown_pct": 0.3,
        "profit_over_max_drawdown": 3.6,
        "trades": scaled_gate_values(cfg, cfg.validation_timerange)["min_trades"] - 8,
    }
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 45,
            "candidate": {"candidate_type": "rank_profile", "name": "prior_undertraded", "rank_profile": anchor},
            "parameter_signature": rank_profile_signature(anchor),
            "window_metrics": {
                "search": {
                    "constraints_ok": True,
                    "research_metrics": {"profit_pct": 3.6, "max_drawdown_pct": 2.8, "profit_over_max_drawdown": 1.25, "trades": 64},
                },
                "validation": {
                    "constraints_ok": False,
                    "research_metrics": validation_metrics,
                    "signal_dir": str(prior_dir),
                    "violations": ["research: trades=11 < 19"],
                },
            },
        },
        {
            "run_id": cfg.run_id,
            "iteration": 46,
            "candidate": {
                "candidate_type": "rank_profile",
                "name": "duplicate_pair_count_repair",
                "metadata": {
                    "source": "controller_rank_profile_positive_validation_trade_repair",
                    "hypothesis_family": "validation_trade_regime_coverage_repair",
                    "changed_keys": ["regime_min_pair_count"],
                },
                "rank_profile": {**anchor, "regime_min_pair_count": 2},
            },
            "parameter_signature": rank_profile_signature({**anchor, "regime_min_pair_count": 2}),
            "window_metrics": {
                "search": {
                    "constraints_ok": True,
                    "research_metrics": {"profit_pct": 3.6, "max_drawdown_pct": 2.8, "profit_over_max_drawdown": 1.25, "trades": 64},
                },
                "validation": {
                    "constraints_ok": False,
                    "research_metrics": validation_metrics,
                    "signal_dir": str(duplicate_dir),
                    "violations": ["research: trades=11 < 19"],
                },
            },
        },
    ]

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)

    assert queue
    assert all(
        item["metadata"].get("source") != "controller_rank_profile_positive_validation_trade_repair"
        or item["metadata"].get("changed_keys") != ["regime_min_pair_count"]
        for item in queue
    )
    assert any(
        item["metadata"].get("source") == "controller_rank_profile_positive_validation_trade_repair"
        and item["metadata"].get("changed_keys") == ["regime_min_edge_ic", "regime_min_pair_edge_ic"]
        for item in queue
    )


def test_rank_profile_repair_queue_prioritizes_search_quality_after_duplicate_paths(tmp_path: Path) -> None:
    anchor = {
        "n": 50,
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 5,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.42,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 3.0,
        "short_max_mom_24h": 0.042,
        "max_entry_atr_pct": 0.05,
        "regime_mode": "hq",
        "regime_min_pair_count": 3,
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_search_quality_after_duplicate",
        run_id="unit_search_quality_after_duplicate_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 23,
            "candidate": {
                "candidate_type": "rank_profile",
                "name": "duplicate_path",
                "rank_profile": anchor,
                "metadata": {
                    "source": "controller_rank_profile_validation_pass_activity_repair",
                    "hypothesis_family": "validation_activity_market_coverage_repair",
                    "changed_keys": ["regime_short_max_market_mom_24h"],
                },
            },
            "parameter_signature": rank_profile_signature(anchor),
            "behavior_novelty": {"status": "duplicate"},
        },
        {
            "run_id": cfg.run_id,
            "iteration": 24,
            "candidate": {"candidate_type": "rank_profile", "name": "active_search_low_quality", "rank_profile": anchor},
            "parameter_signature": rank_profile_signature({**anchor, "min_abs_score_z": 1.41}),
            "window_metrics": {
                "search": {
                    "constraints_ok": False,
                    "research_metrics": {
                        "profit_pct": 4.0,
                        "max_drawdown_pct": 5.0,
                        "profit_over_max_drawdown": 0.8,
                        "trades": scaled_gate_values(cfg, cfg.search_timerange)["min_trades"] + 20,
                    },
                    "violations": ["profit_over_max_drawdown=0.8 < 1.2"],
                }
            },
        },
    ]

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)

    assert queue
    assert queue[0]["metadata"]["source"] == "controller_rank_profile_search_quality_repair"
    assert queue[0]["metadata"]["parent_anchor"] == "iteration_24"
    assert queue[0]["metadata"]["behavior_feedback"]
    assert queue[0]["rank_profile"]["min_abs_score_z"] > anchor["min_abs_score_z"]


def test_rank_profile_repair_queue_adds_search_pair_focus_repairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    anchor = {
        "n": 50,
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 5,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.45,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 3.0,
        "exclude_pairs": ["BTC/USDT", "SOL/USDT"],
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_search_pair_focus",
        run_id="unit_search_pair_focus_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    monkeypatch.setattr(
        strategy_loop_mod,
        "_pair_pnl_order_from_signal_dir",
        lambda raw_signal_dir, anchor_profile: [
            ("DOT/USDT", -0.04),
            ("BNB/USDT", -0.02),
            ("ADA/USDT", 0.03),
            ("ETH/USDT", 0.02),
        ],
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 7,
            "candidate": {"candidate_type": "rank_profile", "name": "active_search_loss", "rank_profile": anchor},
            "parameter_signature": rank_profile_signature(anchor),
            "window_metrics": {
                "search": {
                    "signal_dir": "artifacts/unit/signals",
                    "constraints_ok": False,
                    "research_metrics": {
                        "profit_pct": -5.0,
                        "max_drawdown_pct": 6.0,
                        "profit_over_max_drawdown": -0.8,
                        "trades": scaled_gate_values(cfg, cfg.search_timerange)["min_trades"] + 20,
                    },
                    "violations": ["profit_over_max_drawdown=-0.8 < 1.2"],
                }
            },
        }
    ]

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)

    assert queue
    assert queue[0]["metadata"]["source"] == "controller_rank_profile_search_pair_focus_repair"
    assert queue[0]["metadata"]["parent_anchor"] == "iteration_7"
    assert queue[0]["rank_profile"]["top_k"] == 2
    assert queue[0]["rank_profile"]["exclude_pairs"] == ["BTC/USDT", "SOL/USDT", "DOT/USDT", "BNB/USDT"]
    assert queue[0]["metadata"]["search_pair_pnl_summary"]["profitable_pairs"] == ["ADA/USDT", "ETH/USDT"]


def test_rank_profile_repair_queue_prioritizes_validation_trade_repair_after_search_pass(tmp_path: Path) -> None:
    anchor = {
        "n": 50,
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 5,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.50,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 3.0,
        "short_max_mom_24h": 0.034,
        "short_exit_mom_24h": -0.02,
        "max_entry_atr_pct": 0.05,
        "exclude_pairs": ["BTC/USDT", "SOL/USDT", "LINK/USDT", "DOGE/USDT"],
        "regime_mode": "hq",
        "regime_min_pair_count": 3,
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_validation_trade_after_search_pass",
        run_id="unit_validation_trade_after_search_pass_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 23,
            "candidate": {
                "candidate_type": "rank_profile",
                "name": "duplicate_path",
                "rank_profile": anchor,
                "metadata": {
                    "source": "controller_rank_profile_validation_pass_activity_repair",
                    "hypothesis_family": "validation_activity_market_coverage_repair",
                    "changed_keys": ["regime_short_max_market_mom_24h"],
                },
            },
            "parameter_signature": rank_profile_signature(anchor),
            "behavior_novelty": {"status": "duplicate"},
        },
        {
            "run_id": cfg.run_id,
            "iteration": 45,
            "candidate": {"candidate_type": "rank_profile", "name": "search_near_quality", "rank_profile": {**anchor, "min_abs_score_z": 1.48}},
            "parameter_signature": rank_profile_signature({**anchor, "min_abs_score_z": 1.48}),
            "window_metrics": {
                "search": {
                    "constraints_ok": False,
                    "research_metrics": {
                        "profit_pct": 3.5,
                        "max_drawdown_pct": 3.0,
                        "profit_over_max_drawdown": 1.18,
                        "trades": scaled_gate_values(cfg, cfg.search_timerange)["min_trades"] + 10,
                    },
                }
            },
        },
        {
            "run_id": cfg.run_id,
            "iteration": 46,
            "candidate": {"candidate_type": "rank_profile", "name": "search_pass_validation_undertraded", "rank_profile": anchor},
            "parameter_signature": rank_profile_signature(anchor),
            "window_metrics": {
                "search": {
                    "constraints_ok": True,
                    "research_metrics": {
                        "profit_pct": 3.6,
                        "max_drawdown_pct": 2.8,
                        "profit_over_max_drawdown": 1.25,
                        "trades": scaled_gate_values(cfg, cfg.search_timerange)["min_trades"] + 10,
                    },
                },
                "validation": {
                    "constraints_ok": False,
                    "research_metrics": {
                        "profit_pct": 1.1,
                        "max_drawdown_pct": 0.3,
                        "profit_over_max_drawdown": 3.6,
                        "trades": scaled_gate_values(cfg, cfg.validation_timerange)["min_trades"] - 8,
                    },
                    "violations": ["research: trades=11 < 19"],
                },
            },
        },
    ]

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)

    assert queue
    assert queue[0]["metadata"]["source"] == "controller_rank_profile_positive_validation_trade_repair"
    assert queue[0]["metadata"]["parent_anchor"] == "iteration_46"
    names = [item["name"] for item in queue[:16]]
    assert "validation_trade_regime_pair_count_minus_1_iter_46" in names
    assert "validation_trade_regime_edge_minus_005_iter_46" in names
    assert "validation_trade_regime_market_mom_plus_003_iter_46" in names
    assert "validation_trade_regime_market_mom_plus_005_z_plus_001_iter_46" in names
    assert "validation_trade_low_pair_topk_plus_1_iter_46" in names
    assert "validation_trade_min_pairs_for_topk_minus_2_iter_46" in names
    assert "validation_trade_topk_plus_1_z_minus_002_iter_46" in names


def test_rank_profile_repair_queue_prioritizes_validation_failures(tmp_path: Path) -> None:
    near_pdd = {
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 3,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.47,
        "rebalance_hours": 6,
        "risk_per_trade": 0.013,
        "leverage_cap": 4.0,
        "short_max_mom_24h": 0.038,
        "max_entry_atr_pct": 0.05,
    }
    validation_fail = {
        **near_pdd,
        "risk_per_trade": 0.0117,
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_validation_repair",
        run_id="unit_validation_repair_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 13,
            "candidate": {"candidate_type": "rank_profile", "name": "near_pdd", "rank_profile": near_pdd},
            "parameter_signature": rank_profile_signature(near_pdd),
            "window_metrics": {
                "search": {
                    "constraints_ok": False,
                    "research_metrics": {
                        "profit_pct": 13.8,
                        "max_drawdown_pct": 11.7,
                        "profit_over_max_drawdown": 1.18,
                        "trades": 101,
                    },
                    "violations": ["profit_over_max_drawdown=1.18 < 1.2"],
                }
            },
        },
        {
            "run_id": cfg.run_id,
            "iteration": 21,
            "candidate": {"candidate_type": "rank_profile", "name": "validation_fail", "rank_profile": validation_fail},
            "parameter_signature": rank_profile_signature(validation_fail),
            "window_metrics": {
                "search": {
                    "constraints_ok": True,
                    "research_metrics": {
                        "profit_pct": 14.1,
                        "max_drawdown_pct": 11.1,
                        "profit_over_max_drawdown": 1.27,
                        "trades": 101,
                    },
                    "violations": [],
                },
                "validation": {
                    "constraints_ok": False,
                    "research_metrics": {
                        "profit_pct": -3.8,
                        "max_drawdown_pct": 6.8,
                        "profit_over_max_drawdown": -0.56,
                        "trades": 24,
                    },
                    "violations": ["research: profit_over_max_drawdown=-0.56 < 1.2"],
                },
            },
        },
    ]

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)
    runner = StrategyLoopRunner(cfg)
    runner.state.iteration = 22
    runner.state.score_history = rows
    idir = tmp_path / "iter_22"
    idir.mkdir()

    assert queue
    assert queue[0]["metadata"]["source"] == "controller_rank_profile_validation_repair"
    assert queue[0]["metadata"]["parent_anchor"] == "iteration_21"
    assert queue[0]["metadata"]["anchor_validation_profit_pct"] == -3.8
    assert queue[0]["rank_profile"]["regime_mode"] == "hq"
    assert runner._seed_rank_profile_repair_candidate(idir, idir / "candidate.json") is True
    candidate = validate_candidate(idir / "candidate.json")
    assert candidate["metadata"]["source"] == "controller_rank_profile_validation_repair"
    assert candidate["metadata"]["parent_anchor"] == "iteration_21"


def test_rank_profile_repair_queue_recovers_validation_trade_gap_after_regime(tmp_path: Path) -> None:
    anchor = {
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 2,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.48,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 4.0,
        "short_max_mom_24h": 0.038,
        "max_entry_atr_pct": 0.05,
        "regime_mode": "hq",
        "regime_min_edge_ic": 0.01,
        "regime_min_pair_edge_ic": 0.01,
        "regime_min_pair_count": 3,
        "regime_short_max_market_mom_24h": 0.03,
        "regime_max_market_atr_pct": 0.04,
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_validation_trade_repair",
        run_id="unit_validation_trade_repair_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 25,
            "candidate": {"candidate_type": "rank_profile", "name": "validation_regime", "rank_profile": anchor},
            "parameter_signature": rank_profile_signature(anchor),
            "window_metrics": {
                "search": {
                    "constraints_ok": True,
                    "research_metrics": {
                        "profit_pct": 35.7,
                        "max_drawdown_pct": 6.5,
                        "profit_over_max_drawdown": 5.45,
                        "trades": 60,
                    },
                    "violations": [],
                },
                "validation": {
                    "constraints_ok": False,
                    "research_metrics": {
                        "profit_pct": -2.1,
                        "max_drawdown_pct": 5.5,
                        "profit_over_max_drawdown": -0.38,
                        "trades": scaled_gate_values(cfg, cfg.validation_timerange)["min_trades"] - 3,
                    },
                    "violations": ["research: trades=16 < 19"],
                },
            },
        }
    ]

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)

    assert queue
    assert queue[0]["metadata"]["source"] == "controller_rank_profile_validation_repair"
    assert queue[0]["metadata"]["hypothesis_family"] == "validation_trade_repair_after_regime"
    assert queue[0]["metadata"]["parent_anchor"] == "iteration_25"
    assert queue[0]["rank_profile"]["top_k"] == 3
    assert queue[0]["rank_profile"]["regime_mode"] == "hq"


def test_rank_profile_repair_queue_adds_validation_side_structural_repairs(tmp_path: Path) -> None:
    anchor = {
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 2,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.48,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 4.0,
        "short_max_mom_24h": 0.038,
        "max_entry_atr_pct": 0.05,
        "regime_mode": "hq",
        "regime_min_edge_ic": 0.01,
        "regime_min_pair_edge_ic": 0.01,
        "regime_min_pair_count": 3,
        "regime_short_max_market_mom_24h": 0.03,
        "regime_max_market_atr_pct": 0.04,
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_validation_side_repair",
        run_id="unit_validation_side_repair_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 31,
            "candidate": {"candidate_type": "rank_profile", "name": "validation_short_loss", "rank_profile": anchor},
            "parameter_signature": rank_profile_signature(anchor),
            "window_metrics": {
                "search": {
                    "constraints_ok": True,
                    "research_metrics": {
                        "profit_pct": 39.2,
                        "max_drawdown_pct": 6.5,
                        "profit_over_max_drawdown": 5.98,
                        "trades": 58,
                    },
                    "violations": [],
                },
                "validation": {
                    "constraints_ok": False,
                    "research_metrics": {
                        "profit_pct": -2.1,
                        "max_drawdown_pct": 5.5,
                        "profit_over_max_drawdown": -0.38,
                        "trades": 24,
                    },
                    "violations": ["research: profit_over_max_drawdown=-0.38 < 1.2"],
                },
            },
        }
    ]

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)

    assert queue
    assert queue[0]["metadata"]["source"] == "controller_rank_profile_validation_repair"
    assert queue[0]["metadata"]["hypothesis_family"] == "validation_side_structure_repair"
    assert queue[0]["metadata"]["search_mode"] == "structured_explore"
    assert queue[0]["rank_profile"]["side_mode"] == "both"
    assert queue[0]["rank_profile"]["long_min_mom_24h"] == 0.0


def test_rank_profile_repair_queue_adds_validation_factor_subset_repairs(tmp_path: Path) -> None:
    anchor = {
        "n": 50,
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 2,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "long",
        "long_min_mom_24h": 0.0,
        "min_abs_score_z": 1.48,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 4.0,
        "max_entry_atr_pct": 0.05,
        "regime_mode": "hq",
        "regime_min_edge_ic": 0.01,
        "regime_min_pair_edge_ic": 0.01,
        "regime_min_pair_count": 3,
        "regime_short_max_market_mom_24h": 0.03,
        "regime_max_market_atr_pct": 0.04,
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_validation_factor_subset_repair",
        run_id="unit_validation_factor_subset_repair_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    rows = [
        {
            "run_id": cfg.run_id,
            "iteration": 40,
            "candidate": {"candidate_type": "rank_profile", "name": "validation_factor_loss", "rank_profile": anchor},
            "parameter_signature": rank_profile_signature(anchor),
            "window_metrics": {
                "search": {
                    "constraints_ok": True,
                    "research_metrics": {
                        "profit_pct": 20.0,
                        "max_drawdown_pct": 6.0,
                        "profit_over_max_drawdown": 3.0,
                        "trades": 80,
                    },
                    "violations": [],
                },
                "validation": {
                    "constraints_ok": False,
                    "research_metrics": {
                        "profit_pct": -3.0,
                        "max_drawdown_pct": 5.0,
                        "profit_over_max_drawdown": -0.6,
                        "trades": 24,
                    },
                    "violations": ["research: profit_over_max_drawdown=-0.6 < 1.2"],
                },
            },
        }
    ]

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)

    assert queue
    assert queue[0]["metadata"]["source"] == "controller_rank_profile_validation_repair"
    assert queue[0]["metadata"]["hypothesis_family"] == "validation_factor_subset_repair"
    assert queue[0]["metadata"]["search_mode"] == "structured_explore"
    assert queue[0]["rank_profile"]["n"] == 25
    assert queue[0]["rank_profile"]["candidate_state"] == anchor["candidate_state"]


def test_rank_profile_repair_queue_prioritizes_factor_subset_after_repeated_validation_losses(tmp_path: Path) -> None:
    anchor = {
        "n": 50,
        "candidate_state": "artifacts/factor_lab/mining/unit/state_0149.json",
        "recompute_corr": False,
        "top_k": 2,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.52,
        "rebalance_hours": 6,
        "risk_per_trade": 0.015,
        "leverage_cap": 4.0,
        "short_max_mom_24h": 0.038,
        "short_max_market_mom_24h": 0.03,
        "max_entry_atr_pct": 0.05,
        "regime_mode": "hq",
        "regime_min_edge_ic": 0.01,
        "regime_min_pair_edge_ic": 0.01,
        "regime_min_pair_count": 3,
        "regime_short_max_market_mom_24h": 0.03,
        "regime_max_market_atr_pct": 0.04,
    }
    cfg = StrategyLoopConfig.from_args(
        tag="unit_validation_factor_priority",
        run_id="unit_validation_factor_priority_run",
        validation_protocol="triple_holdout",
        baseline_profile=str(tmp_path / "missing_optimized_profile.json"),
    )
    rows = []
    for iteration, validation_profit in ((31, -2.1), (36, -1.8), (37, -1.7), (38, -1.9)):
        row_profile = {**anchor, "min_abs_score_z": 1.50 + (iteration % 10) * 0.01}
        rows.append(
            {
                "run_id": cfg.run_id,
                "iteration": iteration,
                "candidate": {
                    "candidate_type": "rank_profile",
                    "name": f"validation_loss_{iteration}",
                    "rank_profile": row_profile,
                },
                "parameter_signature": rank_profile_signature(row_profile),
                "window_metrics": {
                    "search": {
                        "constraints_ok": True,
                        "research_metrics": {
                            "profit_pct": 30.0 + iteration,
                            "max_drawdown_pct": 6.5,
                            "profit_over_max_drawdown": 4.0 + iteration / 100.0,
                            "trades": 58,
                        },
                        "violations": [],
                    },
                    "validation": {
                        "constraints_ok": False,
                        "research_metrics": {
                            "profit_pct": validation_profit,
                            "max_drawdown_pct": 5.5,
                            "profit_over_max_drawdown": validation_profit / 5.5,
                            "trades": scaled_gate_values(cfg, cfg.validation_timerange)["min_trades"] - 3,
                        },
                        "violations": ["research: validation loss after search pass"],
                    },
                },
            }
        )

    queue = build_rank_profile_repair_queue({}, cfg, rows=rows)

    assert queue
    assert queue[0]["metadata"]["source"] == "controller_rank_profile_validation_repair"
    assert queue[0]["metadata"]["hypothesis_family"] == "validation_factor_subset_repair"
    assert queue[0]["metadata"]["parent_anchor"] == "iteration_37"
    assert queue[0]["rank_profile"]["n"] == 25
    assert queue[0]["rank_profile"]["top_k"] == 2


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
    header_only_csv = tmp_path / "lookahead_header_only.csv"
    header_only_csv.write_text(
        "filename,strategy,has_bias,total_signals,biased_entry_signals,biased_exit_signals,biased_indicators\n",
        encoding="utf-8",
    )
    header_only_log = tmp_path / "lookahead_header_only.log"
    header_only_log.write_text(
        "2026-05-17 23:06:28,368 - freqtrade.optimize.analysis.lookahead - INFO - "
        "ELRankPortfolioLeverageStrategy: no bias detected\n"
        "\u2502 ELRankPortfolioLeverageStrategy.py \u2502 ELRankPortfolioLeverageStrategy \u2502 "
        "No \u2502 19 \u2502 0 \u2502 0 \u2502 \u2502\n",
        encoding="utf-8",
    )
    recursive_ok = tmp_path / "recursive_ok.csv"
    recursive_ok.write_text("indicator,diff\nrsi,0\n", encoding="utf-8")
    recursive_bad = tmp_path / "recursive_bad.csv"
    recursive_bad.write_text("indicator,diff\nrsi,0.001\n", encoding="utf-8")
    recursive_ok_log = tmp_path / "recursive_ok.log"
    recursive_ok_log.write_text(
        "2026-05-14 05:31:56,112 - freqtrade.optimize.analysis.recursive - INFO - Start checking for recursive bias\n"
        "2026-05-14 05:31:56,114 - freqtrade.optimize.analysis.recursive - INFO - No variance on indicator(s) found due to recursive formula.\n"
        "2026-05-14 05:31:56,114 - freqtrade.optimize.analysis.recursive - INFO - Start checking for lookahead bias on indicators only\n"
        "2026-05-14 05:31:56,116 - freqtrade.optimize.analysis.recursive - INFO - No lookahead bias on indicators found.\n",
        encoding="utf-8",
    )

    assert parse_lookahead_csv(ok_csv, min_trades=5)["status"] == VERIFICATION_PASSED
    assert parse_lookahead_csv(bad_csv, min_trades=5)["status"] == VERIFICATION_FAILED
    assert parse_lookahead_csv(ok_csv, min_trades=50)["status"] == VERIFICATION_FAILED
    assert parse_lookahead_csv(header_only_csv, min_trades=19, log_path=header_only_log)["status"] == VERIFICATION_PASSED
    assert parse_lookahead_csv(header_only_csv, min_trades=20, log_path=header_only_log)["status"] == VERIFICATION_FAILED
    assert parse_recursive_output(recursive_ok)["status"] == VERIFICATION_PASSED
    assert parse_recursive_output(recursive_ok_log)["status"] == VERIFICATION_PASSED
    assert parse_recursive_output(recursive_bad)["status"] == VERIFICATION_FAILED
    assert parse_recursive_output(tmp_path / "missing.csv")["status"] == VERIFICATION_INCONCLUSIVE


def test_validation_gates_bound_recursive_startup_candles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit_validation_gates", run_id="unit_validation_gates_run")
    runner = StrategyLoopRunner(cfg)
    signal_dir = tmp_path / "signals"
    config_path = tmp_path / "config.json"
    strategy_dir = tmp_path / "strategies"
    signal_dir.mkdir()
    strategy_dir.mkdir()
    config_path.write_text("{}", encoding="utf-8")
    (strategy_dir / f"{strategy_loop_mod.FIXED_FREQTRADE_STRATEGY}.py").write_text("# unit\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_freqtrade_validation_base", lambda _stage_result: (signal_dir, config_path, strategy_dir))
    commands: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        if "lookahead-analysis" in cmd:
            export_path = Path(cmd[cmd.index("--lookahead-analysis-exportfilename") + 1])
            export_path.write_text(
                "strategy,has_bias,biased_entry_signals,biased_exit_signals,biased_indicators,total_signals\n"
                "ELRankPortfolioLeverageStrategy,false,0,0,,200\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="lookahead ok\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="indicator,diff\nrsi,0\n")

    monkeypatch.setattr("agent_market.factor_lab.strategy_loop.subprocess.run", fake_run)

    result = runner._run_validation_gates(tmp_path, {"tag": "unit"}, timerange="20260301-20260331", gate_label="validation")

    recursive_cmd = next(cmd for cmd in commands if "recursive-analysis" in cmd)
    idx = recursive_cmd.index("--startup-candle")
    assert recursive_cmd[idx + 1:idx + 4] == list(strategy_loop_mod.RECURSIVE_ANALYSIS_STARTUP_CANDLES)
    assert result["status"] == VERIFICATION_PASSED


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


def test_pareto_pool_excludes_failed_iteration_rows() -> None:
    rows = [
        {
            "run_id": "unit_pareto_failed",
            "iteration": 1,
            "candidate_path": "artifacts/factor_strategy_loop/unit/iter_01/candidate.json",
            "candidate": {"candidate_type": "rank_profile", "rank_profile": {"top_k": 2, "regime_mode": "bad"}},
            "parameter_signature": "",
            "score": -1_000_000.0,
            "score_components": {"composite_score": 10_000.0},
            "verification_status": VERIFICATION_FAILED,
        },
        {
            "run_id": "unit_pareto_failed",
            "iteration": 2,
            "candidate_path": "artifacts/factor_strategy_loop/unit/iter_02/candidate.json",
            "candidate": {"candidate_type": "rank_profile", "rank_profile": {"top_k": 4}},
            "parameter_signature": "failed_positive_score",
            "score": 10.0,
            "score_components": {"composite_score": 10_000.0},
            "research_metrics": {"profit_pct": -2.0, "profit_over_max_drawdown": -1.0, "trades": 80},
            "constraints_ok": False,
        },
        {
            "run_id": "unit_pareto_failed",
            "iteration": 3,
            "candidate_path": "artifacts/factor_strategy_loop/unit/iter_03/candidate.json",
            "candidate": {"candidate_type": "rank_profile", "rank_profile": {"top_k": 3}},
            "parameter_signature": "valid",
            "score": 1.0,
            "score_components": {"composite_score": 1.0},
            "constraints_ok": True,
        },
    ]

    pool = build_pareto_pool(rows)

    assert [row["iteration"] for row in pool["axes"]["best_validation_composite"]] == [3]
    assert all(row["iteration"] != 1 for row in pool["finalists"])
    assert all(row["iteration"] != 2 for row in pool["finalists"])


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


def test_structured_mode_accepts_signal_filter_changes() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit_structured_filter", run_id="unit_structured_filter_run")
    runner = StrategyLoopRunner(cfg)
    runner.state.exploration_mode = "structured"
    runner.state.best_candidate = {
        "candidate": {
            "candidate_type": "rank_profile",
            "rank_profile": {
                "top_k": 2,
                "side_mode": "short",
                "min_abs_score_z": 1.45,
                "short_exit_mom_24h": 0.04,
            },
        }
    }
    threshold_change = {
        "candidate_type": "rank_profile",
        "metadata": {"search_mode": "structured_explore"},
        "rank_profile": {"top_k": 2, "side_mode": "short", "min_abs_score_z": 1.46, "short_exit_mom_24h": 0.04},
    }
    exit_change = {
        "candidate_type": "rank_profile",
        "metadata": {"search_mode": "structured_explore"},
        "rank_profile": {"top_k": 2, "side_mode": "short", "min_abs_score_z": 1.45, "short_exit_mom_24h": 0.0},
    }

    runner._validate_unique_candidate(threshold_change)
    runner._validate_unique_candidate(exit_change)


def test_structured_mode_accepts_regime_gate_changes() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit_structured_regime", run_id="unit_structured_regime_run")
    runner = StrategyLoopRunner(cfg)
    runner.state.exploration_mode = "structured"
    runner.state.best_candidate = {
        "candidate": {
            "candidate_type": "rank_profile",
            "rank_profile": {
                "top_k": 5,
                "side_mode": "short",
                "regime_mode": "hq",
                "regime_min_pair_count": 3,
                "regime_min_edge_ic": 0.01,
                "regime_min_pair_edge_ic": 0.01,
                "regime_short_max_market_mom_24h": 0.03,
                "regime_max_market_atr_pct": 0.04,
            },
        }
    }
    regime_change = {
        "candidate_type": "rank_profile",
        "metadata": {"search_mode": "structured_explore"},
        "rank_profile": {
            "top_k": 5,
            "side_mode": "short",
            "regime_mode": "hq",
            "regime_min_pair_count": 2,
            "regime_min_edge_ic": 0.01,
            "regime_min_pair_edge_ic": 0.01,
            "regime_short_max_market_mom_24h": 0.03,
            "regime_max_market_atr_pct": 0.04,
        },
    }

    runner._validate_unique_candidate(regime_change)
