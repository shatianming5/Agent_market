"""Tests for wq_brain session_log module."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_market.wq_brain.session_log import (
    _yaml_dump,
    _yaml_scalar,
    append_round,
    write_final_summary,
    write_session_metadata,
)


class TestYamlScalar:
    def test_none(self):
        assert _yaml_scalar(None) == "null"

    def test_bool(self):
        assert _yaml_scalar(True) == "true"
        assert _yaml_scalar(False) == "false"

    def test_float(self):
        assert _yaml_scalar(1.7700) == "1.7700"

    def test_plain_string(self):
        assert _yaml_scalar("hello") == "hello"

    def test_string_with_colon(self):
        result = _yaml_scalar("key: value")
        assert result.startswith("'")
        assert "key: value" in result

    def test_string_with_hash(self):
        result = _yaml_scalar("rank(-ts_mean(returns,60))")
        assert result.startswith("'")

    def test_int(self):
        assert _yaml_scalar(42) == "42"


class TestYamlDump:
    def test_empty_dict(self):
        assert _yaml_dump({}) == "{}"

    def test_empty_list(self):
        assert _yaml_dump([]) == "[]"

    def test_flat_dict(self):
        result = _yaml_dump({"a": 1, "b": "hello"})
        assert "a: 1" in result
        assert "b: hello" in result

    def test_nested_dict(self):
        result = _yaml_dump({"outer": {"inner": 42}})
        assert "outer:" in result
        assert "inner: 42" in result

    def test_dict_with_list(self):
        result = _yaml_dump({"items": [1, 2, 3]})
        assert "items:" in result
        assert "- 1" in result

    def test_float_precision(self):
        result = _yaml_dump({"sharpe": 1.77})
        assert "1.7700" in result

    def test_none_value(self):
        result = _yaml_dump({"x": None})
        assert "x: null" in result

    def test_bool_values(self):
        result = _yaml_dump({"flag": True, "other": False})
        assert "flag: true" in result
        assert "other: false" in result

    def test_string_quoting(self):
        result = _yaml_dump({"expr": "rank(-ts_mean(returns,60))"})
        assert "'rank(-ts_mean(returns,60))'" in result


class TestWriteSessionMetadata:
    def test_creates_file(self, tmp_path):
        cfg = {
            "run_id": "run_001",
            "tag": "test_tag",
            "region": "USA",
            "universe": "TOP3000",
            "max_iterations": 50,
            "batch_size": 5,
            "decay": 6,
            "model": "MiniMax-M2.7",
            "hermes_provider": "openrouter",
        }
        write_session_metadata(tmp_path, cfg)
        meta_file = tmp_path / "session_metadata.yml"
        assert meta_file.exists()
        content = meta_file.read_text()
        assert "run_id: run_001" in content
        assert "tag: test_tag" in content
        assert "region: USA" in content
        assert "universe: TOP3000" in content
        assert "max_iterations: 50" in content
        assert "started_at:" in content


class TestAppendRound:
    def test_creates_round_yml(self, tmp_path):
        record = {
            "iteration": 1,
            "generated": 5,
            "simulated": 5,
            "passed": 0,
            "submitted": 0,
            "top_sharpe": 1.77,
            "top_fitness": 0.93,
            "top_expr": "rank(-ts_delta(close,1)/close*volume/adv20)",
            "failure_summary": {"LOW_FITNESS": 5, "HIGH_TURNOVER": 2},
            "pool_size": 0,
            "used_mutation": False,
            "kb_hits": 4,
        }
        append_round(tmp_path, record)
        rnd_file = tmp_path / "round.yml"
        assert rnd_file.exists()
        content = rnd_file.read_text()
        assert "iteration: 1" in content
        assert "simulated: 5" in content
        assert "passed: 0" in content
        assert "1.7700" in content
        assert "LOW_FITNESS: 5" in content
        assert "used_mutation: false" in content
        assert "kb_hits: 4" in content

    def test_none_sharpe(self, tmp_path):
        record = {"iteration": 2, "top_sharpe": None, "top_fitness": None}
        append_round(tmp_path, record)
        content = (tmp_path / "round.yml").read_text()
        assert "top_sharpe: null" in content

    def test_default_values(self, tmp_path):
        append_round(tmp_path, {"iteration": 3})
        content = (tmp_path / "round.yml").read_text()
        assert "iteration: 3" in content
        assert "generated: 0" in content


class TestWriteFinalSummary:
    def test_creates_markdown(self, tmp_path):
        state = {
            "config": {"run_id": "run_test", "tag": "smoke"},
            "iteration": 3,
            "total_generated": 15,
            "total_simulated": 15,
            "total_passed": 1,
            "total_submitted": 1,
            "score_history": [
                {
                    "iteration": 1,
                    "simulated": 5,
                    "passed": 0,
                    "top_sharpe": 1.77,
                    "top_fitness": 0.93,
                    "failure_summary": {"LOW_FITNESS": 5},
                },
                {
                    "iteration": 2,
                    "simulated": 5,
                    "passed": 0,
                    "top_sharpe": 1.90,
                    "top_fitness": 0.87,
                    "failure_summary": {"LOW_FITNESS": 4},
                },
                {
                    "iteration": 3,
                    "simulated": 5,
                    "passed": 1,
                    "top_sharpe": 2.10,
                    "top_fitness": 1.05,
                    "failure_summary": {},
                },
            ],
        }
        write_final_summary(tmp_path, state)
        summary_file = tmp_path / "final_summary.md"
        assert summary_file.exists()
        content = summary_file.read_text()
        assert "# WQ Brain Run Summary" in content
        assert "run_test" in content
        assert "Iterations completed: 3" in content
        assert "1.77" in content
        assert "1.90" in content
        assert "2.10" in content
        assert "LOW_FITNESS:5" in content or "LOW_FITNESS" in content

    def test_empty_history(self, tmp_path):
        write_final_summary(tmp_path, {"config": {}, "score_history": []})
        content = (tmp_path / "final_summary.md").read_text()
        assert "# WQ Brain Run Summary" in content
