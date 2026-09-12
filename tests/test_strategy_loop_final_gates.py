from __future__ import annotations

import json

import pytest

from agent_market import paths
from agent_market.factor_lab import strategy_loop as loop


@pytest.fixture
def finalist(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    config = loop.StrategyLoopConfig.from_args(
        tag="benchmark_unit", run_id="benchmark_unit", promote_policy="final",
        validation_protocol="triple_holdout", verify_policy="pareto",
        eval_mode="research", benchmark_suite="benchmark_pack/default",
    )
    runner = loop.StrategyLoopRunner(config)
    root = loop.loop_root(config.run_id)
    source = root / "iter_01"
    source.mkdir(parents=True)
    candidate_path = source / "candidate.json"
    loop.write_json(candidate_path, {"candidate_type": "rank_profile", "rank_profile": {"top_k": 2}})
    strategy_path = tmp_path / "BenchmarkedStrategy.py"
    strategy_path.write_text(
        "from freqtrade.strategy import IStrategy\n"
        "class BenchmarkedStrategy(IStrategy):"
        "\n    timeframe = '1h'"
        "\n    def populate_indicators(self, dataframe, metadata): return dataframe"
        "\n    def populate_entry_trend(self, dataframe, metadata): return dataframe"
        "\n    def populate_exit_trend(self, dataframe, metadata): return dataframe\n",
        encoding="utf-8",
    )
    summary = {
        "profit_total_pct": 12.0, "trades": 100, "positive_days_ratio": 0.6,
        "observation_days": 20, "metrics_trusted": True, "metric_flags": [],
        "walkforward": {"folds_completed": 3, "folds_total": 3},
        "results_per_pair": [{"key": "BTC/USDT", "profit_total_pct": 2.0}],
    }
    freqtrade = {
        "ok": True, "summary": summary,
        "command": ["freqtrade", "--strategy", "BenchmarkedStrategy", "--strategy-path", str(tmp_path)],
    }
    selection = {"freqtrade_backtest": freqtrade}
    loop.write_json(source / "backtest.json", {"stages": {"validation": selection}})
    blind = {
        "ok": True,
        "profit_pct": 12.0, "max_drawdown_pct": 2.0, "trades": 100,
        "freqtrade_backtest": {**freqtrade, "summary": {**summary, "profit_total_pct": 10.0}},
    }
    monkeypatch.setattr(runner, "_refresh_pareto_pool", lambda: {
        "finalists": [{"iteration": 1, "candidate_path": str(candidate_path)}],
    })
    monkeypatch.setattr(runner, "_run_window_backtest", lambda *args, **kwargs: blind)
    monkeypatch.setattr(runner, "_run_validation_gates", lambda *args, **kwargs: {"status": "passed"})
    return runner, root, source


@pytest.mark.parametrize("suite", ["", "benchmark_pack/not-present"])
def test_finalizer_requires_existing_benchmark_pack(finalist, suite):
    runner, root, _ = finalist
    runner.config.benchmark_suite = suite
    promotion = runner._finalize_triple_holdout()
    verdict = json.loads((root / "blind_1/benchmark_verdict.json").read_text())
    assert promotion["promoted"] is False
    assert verdict["status"] == "insufficient_evidence"
    assert verdict["passed"] is False
    assert verdict["failed_ids"] == []
    assert not (paths.artifacts_root() / "rank_portfolio/benchmark_unit/optimized_profile.json").exists()
    assert json.loads((root / "benchmark_verdict.json").read_text())["status"] == "insufficient_evidence"


def test_finalizer_never_substitutes_blind_evidence_for_missing_selection(finalist):
    runner, root, source = finalist
    (source / "backtest.json").unlink()
    assert runner._finalize_triple_holdout()["promoted"] is False
    verdict = json.loads((root / "blind_1/benchmark_verdict.json").read_text())
    assert verdict["status"] == "insufficient_evidence"
    assert "selection and blind" in verdict["reason"]


def test_finalizer_distinguishes_measured_benchmark_failure(finalist):
    runner, root, source = finalist
    backtest_path = source / "backtest.json"
    backtest = json.loads(backtest_path.read_text())
    backtest["stages"]["validation"]["freqtrade_backtest"]["summary"]["observation_days"] = 1
    loop.write_json(backtest_path, backtest)
    assert runner._finalize_triple_holdout()["promoted"] is False
    verdict = json.loads((root / "blind_1/benchmark_verdict.json").read_text())
    assert verdict["status"] == "failed"
    assert "observation_days" in verdict["failed_ids"]
    assert verdict["suite_manifest_path"].endswith("benchmark_pack/default/manifest.json")


def test_finalizer_runs_real_pack_and_audits_once_before_promotion(finalist, monkeypatch):
    runner, root, _ = finalist
    audit_inputs = []
    original = runner._deepresearch_sidecar

    def audit(status):
        audit_inputs.append(json.loads(json.dumps(status)))
        return original(status)

    monkeypatch.setattr(runner, "_deepresearch_sidecar", audit)
    promotion = runner._finalize_triple_holdout()
    assert promotion["promoted"] is True, (promotion, runner.state.final_blind_status)
    assert len(audit_inputs) == 1
    status = json.loads((root / "final_blind_status.json").read_text())
    context_path = paths.resolve_repo_path(status["deepresearch"]["artifacts"]["context"])
    context = json.loads(context_path.read_text())
    assert context["final_status"] == audit_inputs[0]
    assert context["final_status"]["promoted"] is False
    assert context["final_status"]["promotion"]["reason"] == "pending deepresearch audit"
    assert context["final_status"]["selected"] == status["selected"]
    assert status["selected"]["benchmark"]["passed"] is True
    assert status["selected"]["benchmark"]["checks"]
    assert status["selected"]["benchmark"]["suite_manifest_path"].endswith("benchmark_pack/default/manifest.json")
    assert status["selected"]["benchmark"]["holdout"]["delta_pct"] == 2.0


def test_formal_direct_promotion_without_executed_benchmark_is_insufficient(finalist):
    runner, root, source = finalist
    candidate = loop.validate_candidate(source / "candidate.json")
    result = loop.promote_candidate(
        candidate,
        {"constraints_ok": True, "blind_final": True, "verification_status": "passed"},
        runner.config, iter_dir=source, final=True,
    )
    assert result["promoted"] is False
    assert "insufficient_evidence" in result["reason"]
    assert json.loads((source / "benchmark_verdict.json").read_text())["passed"] is False


def test_resume_preserves_frozen_benchmark_and_rejects_replacement(finalist):
    runner, _, _ = finalist
    loop.save_checkpoint(runner.config, runner.state)
    resumed = loop.StrategyLoopRunner(loop.StrategyLoopConfig.from_args(
        tag=runner.config.tag, run_id=runner.config.run_id, resume=True,
        validation_protocol="single",
    ))
    assert resumed.config.benchmark_suite == runner.config.benchmark_suite
    with pytest.raises(ValueError, match="cannot change the frozen benchmark_suite"):
        loop.StrategyLoopRunner(loop.StrategyLoopConfig.from_args(
            tag=runner.config.tag, run_id=runner.config.run_id, resume=True,
            benchmark_suite="another-pack",
        ))
