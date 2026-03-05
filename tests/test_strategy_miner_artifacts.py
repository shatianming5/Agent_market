"""Tests for Strategy Miner standardized artifacts."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agent_market.strategy_miner.artifacts import (
    leaderboard_path,
    proposal_path,
    write_backtest_summary,
    write_candidate_snapshot,
    write_leaderboard,
    write_proposal,
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
