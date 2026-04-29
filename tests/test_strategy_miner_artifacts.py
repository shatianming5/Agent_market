"""Tests for Strategy Miner standardized artifacts."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agent_market.strategy_miner.artifacts import (
    build_failure_pareto,
    candidate_promotion_eligible,
    candidate_verification_status,
    build_multiagent_summary,
    leaderboard_path,
    multiagent_summary_path,
    proposal_path,
    run_manifest_path,
    write_backtest_summary,
    write_candidate_snapshot,
    write_failure_pareto,
    write_leaderboard,
    write_multiagent_summary,
    write_proposal,
    write_run_manifest,
)
from agent_market.strategy_miner.dtypes import MinerConfig, MinerState, StrategyCandidate


def test_write_proposal_creates_file():
    cfg = MinerConfig(model="", max_iterations=1)
    with tempfile.TemporaryDirectory() as td:
        miner_dir = Path(td)
        out = write_proposal(miner_dir, run_id="abc123", config=cfg)
        assert out == proposal_path(miner_dir)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["run_id"] == "abc123"
        assert "config" in data


def test_write_candidate_snapshot_writes_code_and_meta():
    with tempfile.TemporaryDirectory() as td:
        miner_dir = Path(td)
        candidate = StrategyCandidate(
            name="MyStrat",
            code="class X: pass\n",
            strategy_path=Path(td) / "sandbox" / "user_data" / "strategies" / "MyStrat.py",
            iteration=2,
        )
        out = write_candidate_snapshot(miner_dir, candidate)
        assert out["code"].exists()
        assert out["meta"].exists()
        assert "class X" in out["code"].read_text(encoding="utf-8")
        meta = json.loads(out["meta"].read_text(encoding="utf-8"))
        assert meta["candidate"]["name"] == "MyStrat"
        assert meta["candidate"]["iteration"] == 2


def test_write_backtest_summary_writes_iter_scoped_summary():
    with tempfile.TemporaryDirectory() as td:
        miner_dir = Path(td)
        candidate = StrategyCandidate(
            name="MyStrat",
            code="class X: pass\n",
            strategy_path=Path(td) / "sandbox" / "user_data" / "strategies" / "MyStrat.py",
            iteration=3,
            validation_passed=True,
        )
        candidate.reward = 0.42
        candidate.backtest_summary = {"profit_total_pct": 1.2, "trades": 10}
        out = write_backtest_summary(miner_dir, candidate, zip_path=Path(td) / "bt.zip")
        assert out is not None
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["candidate"]["name"] == "MyStrat"
        assert data["candidate"]["iteration"] == 3
        assert data["backtest_zip"].endswith("bt.zip")


def test_write_leaderboard_sorts_by_reward_desc():
    cfg = MinerConfig(min_trades=0, max_abs_drawdown=999, min_winrate=0.0)
    with tempfile.TemporaryDirectory() as td:
        miner_dir = Path(td)
        state = MinerState(run_id="r1")
        c1 = StrategyCandidate("A", "code", Path(td) / "a.py", iteration=0)
        c2 = StrategyCandidate("B", "code", Path(td) / "b.py", iteration=1)
        c1.reward = 0.1
        c2.reward = 0.9
        state.candidates = [c1, c2]

        out = write_leaderboard(miner_dir, state, config=cfg)
        assert out == leaderboard_path(miner_dir)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["items"][0]["name"] == "B"
        assert data["items"][1]["name"] == "A"



def test_write_leaderboard_filters_rejected_by_constraints():
    cfg = MinerConfig(min_trades=10, max_abs_drawdown=50.0, min_winrate=0.0)
    with tempfile.TemporaryDirectory() as td:
        miner_dir = Path(td)
        state = MinerState(run_id="r2")

        bad = StrategyCandidate("Bad", "code", Path(td) / "bad.py", iteration=0)
        bad.reward = 0.9
        bad.constraints_ok = False
        bad.constraint_violations = ["min_trades:1<10"]

        good = StrategyCandidate("Good", "code", Path(td) / "good.py", iteration=1)
        good.reward = 0.1
        good.constraints_ok = True

        state.candidates = [bad, good]

        out = write_leaderboard(miner_dir, state, config=cfg)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert [i["name"] for i in data["items"]] == ["Good"]
        assert [i["name"] for i in data["rejected"]] == ["Bad"]


def test_leaderboard_adds_shared_promotion_fields():
    cfg = MinerConfig(min_trades=0, max_abs_drawdown=999, min_winrate=0.0)
    with tempfile.TemporaryDirectory() as td:
        miner_dir = Path(td)
        state = MinerState(run_id="r3")
        c = StrategyCandidate("Winner", "code", Path(td) / "winner.py", iteration=2, stage="holdout_tested")
        c.reward = 1.1
        c.constraints_ok = True
        c.funnel_state = {"holdout": {"overfitting_flag": False}}
        state.candidates = [c]
        state.best_candidate = c
        state.best_score = 1.1

        out = write_leaderboard(miner_dir, state, config=cfg)
        data = json.loads(out.read_text(encoding="utf-8"))

        assert candidate_verification_status(c) == "passed"
        assert candidate_promotion_eligible(c) is False
        assert data["items"][0]["verification_status"] == "passed"
        assert data["items"][0]["promotion_eligible"] is False
        assert data["items"][0]["promotion_controller"] == "agent_market.factor_lab.strategy-loop"
        pareto = build_failure_pareto(state)
        assert pareto["promoted_count"] == 0
        assert pareto["search_passed_pending_blind_count"] == 1
        assert pareto["failure_count"] == 0


def test_failure_pareto_classifies_common_failures():
    state = MinerState(run_id="failrun")
    syntax = StrategyCandidate("SyntaxBad", "bad", Path("/tmp/syntax.py"))
    syntax.failure_category = "validation.syntax"
    syntax.diagnosis = "Syntax error"

    sample = StrategyCandidate("TooFewTrades", "code", Path("/tmp/sample.py"))
    sample.reward = 0.1
    sample.constraints_ok = False
    sample.constraint_violations = ["min_trades:2<10"]
    sample.backtest_summary = {"profit_total_pct": 1.0, "trades": 2}

    loss = StrategyCandidate("Loser", "code", Path("/tmp/loss.py"))
    loss.reward = -0.5
    loss.backtest_summary = {"profit_total_pct": -3.0, "profit_factor": 0.5, "trades": 20}

    state.candidates = [syntax, sample, loss]
    pareto = build_failure_pareto(state)

    categories = {item["category"]: item["count"] for item in pareto["categories"]}
    assert categories["syntax_failure"] == 1
    assert categories["insufficient_sample"] == 1
    assert categories["unprofitable_failure"] == 1


def test_write_run_manifest_and_failure_pareto():
    cfg = MinerConfig(max_iterations=1)
    with tempfile.TemporaryDirectory() as td:
        miner_dir = Path(td)
        state = MinerState(run_id="manifest1")
        c = StrategyCandidate("Candidate", "code", Path(td) / "c.py", iteration=0)
        c.reward = 0.1
        c.backtest_summary = {"profit_total_pct": -1.0}
        state.candidates = [c]
        state.best_candidate = c
        state.best_score = 0.1
        (miner_dir / "checkpoint.json").write_text("{}", encoding="utf-8")
        write_failure_pareto(miner_dir, state)
        out = write_run_manifest(miner_dir, state, config=cfg)

        assert out == run_manifest_path(miner_dir)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["promotion_controller"] == "agent_market.factor_lab.strategy-loop"
        assert data["files"]["failure_pareto.json"]["exists"] is True


def test_strategy_multiagent_summary_records_search_only_scope():
    cfg = MinerConfig(
        multiagent_enabled=True,
        candidates_per_iteration=6,
        max_parallel_candidates=3,
        max_parallel_roles=2,
        repair_attempts=5,
        max_iterations=8,
    )
    with tempfile.TemporaryDirectory() as td:
        miner_dir = Path(td)
        trace_dir = miner_dir / "agent_traces" / "iter_0000" / "cand_00"
        trace_dir.mkdir(parents=True)
        failure_trace = trace_dir / "failure.json"
        failure_trace.write_text(
            json.dumps({"payload": {"failure_category": "invalid_json"}}, ensure_ascii=False),
            encoding="utf-8",
        )
        c = StrategyCandidate("Candidate", "code", Path(td) / "c.py", iteration=0)
        c.agent_traces = {"planner": str(trace_dir / "planner.json"), "failure": str(failure_trace)}
        c.failure_category = "validation.syntax"
        state = MinerState(run_id="ma1")
        state.candidates = [c]

        orphan_dir = miner_dir / "agent_traces" / "iter_0001" / "cand_03"
        orphan_dir.mkdir(parents=True)
        (orphan_dir / "failure.json").write_text(
            json.dumps({"role": "failure", "payload": {"failure_category": "illegal_code"}}, ensure_ascii=False),
            encoding="utf-8",
        )

        summary = build_multiagent_summary(state, config=cfg, miner_dir=miner_dir)
        assert summary["enabled"] is True
        assert summary["promotion_controller"] == "agent_market.factor_lab.strategy-loop"
        assert summary["config"]["candidates_per_iteration"] == 6
        assert summary["failure_taxonomy"]["invalid_json"] == 1
        assert summary["failure_taxonomy"]["illegal_code"] == 1
        assert summary["failure_taxonomy"]["validation.syntax"] == 1
        assert summary["counts"]["orphan_failure_traces"] == 1
        assert summary["counts"]["promoted_by_strategy_miner"] == 0

        out = write_multiagent_summary(miner_dir, state, config=cfg)
        assert out == multiagent_summary_path(miner_dir)
        written = json.loads(out.read_text(encoding="utf-8"))
        assert written["counts"]["candidates"] == 1
        assert written["failure_taxonomy"]["illegal_code"] == 1
        pareto = build_failure_pareto(state, miner_dir=miner_dir)
        pareto_categories = {item["category"]: item["count"] for item in pareto["categories"]}
        assert pareto_categories["illegal_code"] == 1
        assert pareto["orphan_trace_failure_count"] == 1


def test_multiagent_explore_config_parses_high_exploration_budget():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "agent_flow_strategy_factory_multiagent_explore.json"
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = MinerConfig.from_dict(payload["strategy_miner"])

    assert cfg.multiagent_enabled is True
    assert cfg.candidates_per_iteration == 6
    assert cfg.max_parallel_candidates == 3
    assert cfg.max_parallel_roles == 2
    assert cfg.repair_attempts == 5
    assert cfg.max_iterations == 8
