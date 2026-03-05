"""API integration tests for strategy_miner endpoints."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_start_miner_returns_job(client):
    """POST /strategy-miner/start should return a job_id."""
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
        mock_jobs.start.assert_called_once()


def test_start_miner_with_overrides(client):
    """POST /strategy-miner/start with model and max_iterations."""
    with patch("server.api.routes.strategy_miner.jobs") as mock_jobs:
        mock_jobs.start.return_value = "job-456"

        resp = client.post(
            "/strategy-miner/start",
            json={
                "config": "configs/strategy_miner_default.json",
                "model": "openai/gpt-4o",
                "max_iterations": 2,
            },
        )
        data = resp.json()
        assert data["status"] == "started"
        # Check that CLI args include overrides
        cmd = data["cmd"]
        assert "--model" in cmd
        assert "openai/gpt-4o" in cmd
        assert "--max-iterations" in cmd
        assert "2" in cmd


# ---------------------------------------------------------------------------
# GET /strategy-miner/status/{job_id}
# ---------------------------------------------------------------------------


def test_status_not_found(client):
    """GET /strategy-miner/status/bad-id should return error."""
    with patch("server.api.routes.strategy_miner.jobs") as mock_jobs:
        mock_jobs.logs.return_value = {"error": "Job not found"}

        resp = client.get("/strategy-miner/status/bad-id")
        data = resp.json()
        assert data["status"] == "error"


def test_status_running(client):
    """GET /strategy-miner/status should parse running job logs."""
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
# GET /strategy-miner/results
# ---------------------------------------------------------------------------


def test_results_empty(client):
    """GET /strategy-miner/results with no runs should return empty list."""
    resp = client.get("/strategy-miner/results")
    data = resp.json()
    assert isinstance(data["items"], list)
    assert data["count"] >= 0


def test_results_with_checkpoint(client):
    """GET /strategy-miner/results should find existing checkpoints."""
    miner_root = paths.artifacts_root() / "strategy_miner"
    run_dir = miner_root / "test_abc123"
    run_dir.mkdir(parents=True, exist_ok=True)
    cp = run_dir / "checkpoint.json"
    cp.write_text(json.dumps({
        "run_id": "test_abc123",
        "phase": "complete",
        "iteration": 3,
        "best_reward": 0.75,
        "best_candidate": {"name": "TopStrat", "code": "...", "strategy_path": "/tmp/x.py"},
        "candidates": [],
        "history": [],
    }), encoding="utf-8")

    try:
        resp = client.get("/strategy-miner/results")
        data = resp.json()
        found = [i for i in data["items"] if i["run_id"] == "test_abc123"]
        assert len(found) == 1
        assert found[0]["best_reward"] == 0.75
        assert found[0]["best_candidate"] == "TopStrat"
    finally:
        import shutil
        shutil.rmtree(run_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# GET /strategy-miner/results/{run_id}
# ---------------------------------------------------------------------------


def test_result_detail_not_found(client):
    """GET /strategy-miner/results/nonexistent should return error."""
    resp = client.get("/strategy-miner/results/aabbcc112233")
    data = resp.json()
    assert data["status"] == "error"


def test_result_detail_invalid_id(client):
    """GET /strategy-miner/results/INVALID should return error."""
    resp = client.get("/strategy-miner/results/NOT-VALID!")
    data = resp.json()
    assert data["status"] == "error"
    assert "INVALID_RUN_ID" in data["code"]


def test_result_detail_found(client):
    """GET /strategy-miner/results/{run_id} should return checkpoint data."""
    miner_root = paths.artifacts_root() / "strategy_miner"
    run_dir = miner_root / "aabb00cc1123"
    run_dir.mkdir(parents=True, exist_ok=True)
    cp = run_dir / "checkpoint.json"
    cp.write_text(json.dumps({
        "run_id": "aabb00cc1123",
        "phase": "complete",
        "iteration": 2,
        "best_reward": 0.6,
        "best_candidate": None,
        "candidates": [],
        "history": [{"iteration": 0, "reward": 0.3}],
    }), encoding="utf-8")

    try:
        resp = client.get("/strategy-miner/results/aabb00cc1123")
        data = resp.json()
        assert data["run_id"] == "aabb00cc1123"
        assert data["phase"] == "complete"
        assert data["iteration"] == 2
        assert data["best_reward"] == 0.6
        assert len(data["history"]) == 1
    finally:
        import shutil
        shutil.rmtree(run_dir, ignore_errors=True)
