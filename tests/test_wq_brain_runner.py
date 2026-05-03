"""End-to-end runner tests with mocked LLM and WQ API."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_market.wq_brain.client import WQSession
from agent_market.wq_brain.dtypes import (
    AlphaCandidate,
    AlphaSettings,
    SimulationResult,
    WQBrainConfig,
)
from agent_market.wq_brain.runner import (
    WQBrainRunner,
    _load_checkpoint,
    _parse_candidate_json,
    _save_checkpoint,
    make_wqb_run_id,
    WQBrainState,
)


class TestRunId:
    def test_format(self):
        run_id = make_wqb_run_id("test_tag")
        assert run_id.startswith("wqbrain_test_tag_")
        assert len(run_id) > 20


class TestCheckpoint:
    def test_save_and_load_roundtrip(self, tmp_path):
        from agent_market.wq_brain.paths import wq_brain_run_dir
        config = WQBrainConfig(tag="test", run_id="wqbrain_test_20260101_000000_abcd1234")
        state = WQBrainState(config=config, iteration=3)
        run_dir = tmp_path / "runs" / config.run_id
        run_dir.mkdir(parents=True)
        _save_checkpoint(state, run_dir)
        loaded = _load_checkpoint(run_dir)
        assert loaded is not None
        assert loaded.iteration == 3
        assert loaded.config.tag == "test"

    def test_load_missing_returns_none(self, tmp_path):
        result = _load_checkpoint(tmp_path / "nonexistent" / "run")
        assert result is None


class TestParseCandidateJson:
    def test_valid_candidates(self, tmp_path):
        data = {
            "candidates": [
                {"expr": "rank(close)", "rationale": "momentum"},
                {"expr": "rank(-returns(close, 5))", "rationale": "reversal"},
            ],
            "prompt_version": "wqb-v1.0",
            "iteration": 1,
        }
        candidate_path = tmp_path / "candidate.json"
        candidate_path.write_text(json.dumps(data), encoding="utf-8")
        settings = AlphaSettings()
        candidates = _parse_candidate_json(candidate_path, settings)
        assert len(candidates) == 2
        assert candidates[0].expr == "rank(close)"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _parse_candidate_json(tmp_path / "candidate.json", AlphaSettings())

    def test_empty_candidates_raises(self, tmp_path):
        data = {"candidates": [], "iteration": 1}
        p = tmp_path / "candidate.json"
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="empty"):
            _parse_candidate_json(p, AlphaSettings())


class TestWQBrainRunnerMocked:
    """End-to-end runner test with fully mocked LLM + WQ API."""

    def _make_config(self, tag: str, run_dir: Path) -> WQBrainConfig:
        return WQBrainConfig(
            tag=tag,
            run_id=f"wqbrain_{tag}_20260101_000000_test0001",
            max_iterations=1,
            batch_size=2,
            auto_submit=False,
            dry_run=True,
        )

    def _make_mock_session(self) -> MagicMock:
        session = MagicMock(spec=WQSession)
        session.batch_simulate.side_effect = lambda candidates, **kwargs: [
            setattr(c, "sim_result", SimulationResult(
                sharpe=1.9, fitness=1.6, returns=0.12,
                turnover=0.4, alpha_id=f"alpha_{i}", status="COMPLETE"
            )) or c
            for i, c in enumerate(candidates)
        ]
        session.get_alpha_correlations.return_value = [{"value": 0.3}]
        return session

    def test_run_one_iteration(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))

        tag = "smoke_test"
        config = self._make_config(tag, tmp_path)
        mock_session = self._make_mock_session()

        # Mock hermes CLI: writes candidate.json directly
        def mock_hermes(idir: Path, prompt: str, cfg: WQBrainConfig, **kwargs):
            candidate_data = {
                "candidates": [
                    {"expr": "rank(close / ts_mean(close, 20))", "rationale": "test"},
                    {"expr": "rank(-returns(close, 5))", "rationale": "reversal"},
                ],
                "prompt_version": "wqb-v1.0",
                "iteration": 1,
            }
            (idir / "candidate.json").write_text(json.dumps(candidate_data))

        runner = WQBrainRunner(config, session=mock_session)

        with patch(
            "agent_market.wq_brain.runner._run_hermes_cli",
            side_effect=mock_hermes,
        ):
            runner.run()

        # Check evaluation.json was written
        eval_files = list(
            (tmp_path / "artifacts" / "wq_brain" / "runs" / config.run_id).rglob("evaluation.json")
        )
        assert eval_files, "evaluation.json not written"
        eval_data = json.loads(eval_files[0].read_text())
        assert eval_data["iteration"] == 1
        assert eval_data["simulated"] == 2
