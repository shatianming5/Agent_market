"""API integration tests for strategy_miner endpoints."""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_market import paths


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from server.app import create_app

    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /strategy-miner/start
# ---------------------------------------------------------------------------


def test_start_miner_returns_job_and_run_id(client):
    with patch("server.api.routes.strategy_miner.jobs") as mock_jobs:
        mock_jobs.start.return_value = "test-job-123"

        resp = client.post(
            "/strategy-miner/start",
            json={"config": "configs/strategy_miner_default.json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert data["job_id"] == "test-job-123"
        assert data["kind"] == "strategy_miner"
        assert re.fullmatch(r"[0-9a-f]{12}", data["run_id"] or "")
        assert "--run-id" in data["cmd"]
        mock_jobs.start.assert_called_once()


def test_start_miner_with_overrides(client):
    with patch("server.api.routes.strategy_miner.jobs") as mock_jobs:
        mock_jobs.start.return_value = "job-456"

        resp = client.post(
            "/strategy-miner/start",
            json={
                "config": "configs/strategy_miner_default.json",
                "model": "openai/gpt-4o",
                "max_iterations": 2,
                "run_id": "aabb00cc1123",
            },
        )
        data = resp.json()
        assert data["status"] == "started"
        assert data["run_id"] == "aabb00cc1123"
        cmd = data["cmd"]
        assert "--model" in cmd
        assert "openai/gpt-4o" in cmd
        assert "--max-iterations" in cmd
        assert "2" in cmd
        assert "--run-id" in cmd
        assert "aabb00cc1123" in cmd


# ---------------------------------------------------------------------------
# GET /strategy-miner/status/{job_id}
# ---------------------------------------------------------------------------


def test_status_not_found(client):
    with patch("server.api.routes.strategy_miner.jobs") as mock_jobs:
        mock_jobs.logs.return_value = {"error": "Job not found"}

        resp = client.get("/strategy-miner/status/bad-id")
        data = resp.json()
        assert data["status"] == "error"


def test_status_running(client):
    with patch("server.api.routes.strategy_miner.jobs") as mock_jobs:
        mock_jobs.logs.return_value = {
            "running": True,
            "code": None,
            "logs": [
                "2026-03-04 INFO === Iteration 2 | Phase backtest ===",
                "2026-03-04 INFO best_reward=0.5432",
            ],
        }

        resp = client.get("/strategy-miner/status/job-1")
        data = resp.json()
        assert data["running"] is True
        assert data["iteration"] == 2
        assert data["best_reward"] == pytest.approx(0.5432)


# ---------------------------------------------------------------------------
# Run storage: /runs + /results aliases
# ---------------------------------------------------------------------------


def test_runs_empty(client):
    with tempfile.TemporaryDirectory() as td:
        with patch.dict(os.environ, {"AGENT_MARKET_RUNS_ROOT": td}, clear=False):
            resp = client.get("/strategy-miner/runs")
            data = resp.json()
            assert data["items"] == []
            assert data["count"] == 0


def test_results_with_checkpoint(client):
    with tempfile.TemporaryDirectory() as td:
        with patch.dict(os.environ, {"AGENT_MARKET_RUNS_ROOT": td}, clear=False):
            run_id = "abc123"
            miner_dir = paths.run_dir(run_id) / "strategy_miner"
            miner_dir.mkdir(parents=True, exist_ok=True)
            cp = miner_dir / "checkpoint.json"
            cp.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "phase": "complete",
                        "iteration": 3,
                        "best_reward": 0.75,
                        "best_candidate": {
                            "name": "TopStrat",
                            "code": "...",
                            "strategy_path": str(miner_dir / "iter_0" / "sandbox" / "user_data" / "strategies" / "TopStrat.py"),
                        },
                        "candidates": [],
                        "history": [],
                    }
                ),
                encoding="utf-8",
            )

            # /results is a backward-compatible alias for /runs
            resp = client.get("/strategy-miner/results")
            data = resp.json()
            found = [i for i in data["items"] if i["run_id"] == run_id]
            assert len(found) == 1
            assert found[0]["best_reward"] == 0.75
            assert found[0]["best_candidate"] == "TopStrat"


def test_result_detail_found(client):
    with tempfile.TemporaryDirectory() as td:
        with patch.dict(os.environ, {"AGENT_MARKET_RUNS_ROOT": td}, clear=False):
            run_id = "aabb00cc1123"
            miner_dir = paths.run_dir(run_id) / "strategy_miner"
            miner_dir.mkdir(parents=True, exist_ok=True)
            cp = miner_dir / "checkpoint.json"
            cp.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "phase": "complete",
                        "iteration": 2,
                        "best_reward": 0.6,
                        "best_candidate": None,
                        "candidates": [],
                        "history": [{"iteration": 0, "reward": 0.3}],
                    }
                ),
                encoding="utf-8",
            )

            resp = client.get(f"/strategy-miner/results/{run_id}")
            data = resp.json()
            assert data["run_id"] == run_id
            assert data["phase"] == "complete"
            assert data["iteration"] == 2
            assert data["best_reward"] == 0.6
            assert len(data["history"]) == 1


def test_run_status_checkpoint_based(client):
    with tempfile.TemporaryDirectory() as td:
        with patch.dict(os.environ, {"AGENT_MARKET_RUNS_ROOT": td}, clear=False):
            run_id = "aabb00cc1123"
            miner_dir = paths.run_dir(run_id) / "strategy_miner"
            miner_dir.mkdir(parents=True, exist_ok=True)
            (miner_dir / "checkpoint.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "phase": "analysis",
                        "iteration": 1,
                        "best_reward": 0.1,
                        "best_candidate": None,
                        "candidates": [],
                        "history": [],
                    }
                ),
                encoding="utf-8",
            )

            resp = client.get(f"/strategy-miner/runs/{run_id}/status")
            data = resp.json()
            assert data["run_id"] == run_id
            assert data["phase"] == "analysis"
            assert data["iteration"] == 1


def test_candidates_endpoint(client):
    with tempfile.TemporaryDirectory() as td:
        with patch.dict(os.environ, {"AGENT_MARKET_RUNS_ROOT": td}, clear=False):
            run_id = "aabb00cc1123"
            miner_dir = paths.run_dir(run_id) / "strategy_miner"
            miner_dir.mkdir(parents=True, exist_ok=True)
            (miner_dir / "checkpoint.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "phase": "analysis",
                        "iteration": 1,
                        "best_reward": 0.1,
                        "best_candidate": None,
                        "candidates": [
                            {
                                "name": "S1",
                                "code": "class X: pass",
                                "strategy_path": str(miner_dir / "iter_0" / "sandbox" / "user_data" / "strategies" / "S1.py"),
                                "iteration": 0,
                                "validation_passed": True,
                                "reward": 0.2,
                                "diagnosis": "ok",
                            }
                        ],
                        "history": [],
                    }
                ),
                encoding="utf-8",
            )

            resp = client.get(f"/strategy-miner/runs/{run_id}/candidates")
            data = resp.json()
            assert data["run_id"] == run_id
            assert data["count"] == 1
            assert data["items"][0]["name"] == "S1"


def test_approve_endpoint_copies_strategy(client):
    with tempfile.TemporaryDirectory() as td_runs:
        with tempfile.TemporaryDirectory() as td_user:
            env = {
                "AGENT_MARKET_RUNS_ROOT": td_runs,
                "AGENT_MARKET_USER_DATA_ROOT": td_user,
            }
            with patch.dict(os.environ, env, clear=False):
                run_id = "aabb00cc1123"
                miner_dir = paths.run_dir(run_id) / "strategy_miner"
                strat_dir = miner_dir / "iter_0" / "sandbox" / "user_data" / "strategies"
                strat_dir.mkdir(parents=True, exist_ok=True)
                strat_path = strat_dir / "S1.py"
                strat_path.write_text("# strategy", encoding="utf-8")

                (miner_dir / "checkpoint.json").write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "phase": "complete",
                            "iteration": 0,
                            "best_reward": 0.0,
                            "best_candidate": {"name": "S1", "code": "", "strategy_path": str(strat_path)},
                            "candidates": [],
                            "history": [],
                        }
                    ),
                    encoding="utf-8",
                )

                resp = client.post(f"/strategy-miner/runs/{run_id}/approve", json={})
                data = resp.json()
                assert data["status"] == "ok"
                dest = Path(data["dest"])
                assert dest.exists()
                assert dest.read_text(encoding="utf-8") == "# strategy"


def test_backtest_endpoint_starts_job(client):
    with tempfile.TemporaryDirectory() as td:
        with patch.dict(os.environ, {"AGENT_MARKET_RUNS_ROOT": td}, clear=False):
            run_id = "aabb00cc1123"
            miner_dir = paths.run_dir(run_id) / "strategy_miner"
            miner_dir.mkdir(parents=True, exist_ok=True)
            (miner_dir / "checkpoint.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "phase": "analysis",
                        "iteration": 1,
                        "best_reward": 0.1,
                        "best_candidate": None,
                        "candidates": [],
                        "history": [],
                    }
                ),
                encoding="utf-8",
            )

            with patch("server.api.routes.strategy_miner.jobs") as mock_jobs:
                mock_jobs.start.return_value = "job-1"
                resp = client.post(f"/strategy-miner/runs/{run_id}/backtest", json={"candidate": None})
                data = resp.json()
                assert data["status"] == "started"
                assert data["job_id"] == "job-1"
                assert data["run_id"] == run_id
                assert "strategy_miner_backtest" in data["kind"]
