"""scan_runner tests with mocked WQSession."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_market.wq_brain.dtypes import AlphaSettings, SimulationResult
from agent_market.wq_brain.scan_runner import ScanConfig, run_scan


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


def _make_seeds_file(tmp_path: Path) -> Path:
    cfg = {
        "seeds": [
            "rank(close)",
            "rank(ts_mean(<F1>, <WIN>))",
        ],
        "grid": {"<F1>": ["close", "open"], "<WIN>": [5, 10]},
    }
    p = tmp_path / "seeds.json"
    p.write_text(json.dumps(cfg))
    return p


def test_scan_dry_run_produces_expansion_summary(tmp_path, isolated_artifacts):
    seeds = _make_seeds_file(tmp_path)
    config = ScanConfig(tag="t1", seed_file=seeds, dry_run=True, max_candidates=100)
    summary = run_scan(config)
    assert summary["dry_run"] is True
    assert summary["candidates"] >= 4  # 1 + 4 from grid
    assert summary["simulated"] == 0
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "expansion.json").exists()


def test_scan_full_with_mock_session_passes_quality_filter(tmp_path, isolated_artifacts):
    seeds = _make_seeds_file(tmp_path)
    config = ScanConfig(tag="t1", seed_file=seeds, max_candidates=20, auto_submit=False)

    mock_session = MagicMock()

    def fake_batch_simulate(candidates, **kw):
        for i, c in enumerate(candidates):
            if i == 0:
                # First passes
                c.sim_result = SimulationResult(
                    sharpe=1.5, fitness=1.2, returns=0.15, turnover=0.18,
                    alpha_id=f"A{i}", status="COMPLETE",
                )
            else:
                # Others fail
                c.sim_result = SimulationResult(
                    sharpe=0.8, fitness=0.5, returns=0.05, turnover=0.30,
                    alpha_id=f"A{i}", status="COMPLETE",
                )
        return candidates

    mock_session.batch_simulate.side_effect = fake_batch_simulate
    summary = run_scan(config, session=mock_session)
    assert summary["passed"] == 1
    assert summary["submitted"] == 0  # auto_submit=False


def test_scan_with_auto_submit_calls_submit(tmp_path, isolated_artifacts):
    seeds = _make_seeds_file(tmp_path)
    config = ScanConfig(tag="t1", seed_file=seeds, max_candidates=20, auto_submit=True)

    mock_session = MagicMock()

    def fake_batch_simulate(candidates, **kw):
        for c in candidates:
            c.sim_result = SimulationResult(
                sharpe=1.5, fitness=1.2, returns=0.15, turnover=0.18,
                alpha_id="A_pass", status="COMPLETE",
            )
        return candidates

    mock_session.batch_simulate.side_effect = fake_batch_simulate
    mock_session.submit_alpha.return_value = {"ok": True}

    summary = run_scan(config, session=mock_session)
    assert summary["passed"] == summary["candidates"]
    # submit_alpha called for all (but pool dedupes by alpha_id)
    assert mock_session.submit_alpha.call_count >= 1
    # Pool only retains one (same alpha_id A_pass)
    assert summary["pool_size"] == 1


def test_scan_invalid_expressions_filtered_before_simulate(tmp_path, isolated_artifacts):
    cfg = {
        "seeds": [
            "rank(close)",
            "rank(ts_std(<F1>, 20))",  # ts_std is unavailable
            "lambda x: x",
        ],
        "grid": {"<F1>": ["close"]},
    }
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps(cfg))
    config = ScanConfig(tag="t1", seed_file=seeds, dry_run=True)
    summary = run_scan(config)
    # Only rank(close) should remain (1 valid out of 3)
    assert summary["candidates"] == 1
