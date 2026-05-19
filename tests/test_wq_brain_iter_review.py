"""Per-iteration review (iter_review.json) tests."""
from __future__ import annotations

import json
import time

import pytest

from agent_market.wq_brain.agent_runner import _build_iter_review, _format_review_oneline
from agent_market.wq_brain.paths import tried_exprs_path
from agent_market.wq_brain.tried_log import append_tried


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


def test_build_iter_review_filters_by_timestamp(isolated_artifacts):
    tag = "smoke"
    p = tried_exprs_path(tag)

    # Pre-iter entry
    append_tried(p, expr="rank(close)", sharpe=0.5, fitness=0.3, turnover=0.2,
                 alpha_id="A0", status="COMPLETE")
    # Manually rewrite ts to be in the past
    lines = p.read_text(encoding="utf-8").splitlines()
    rec0 = json.loads(lines[0]); rec0["ts"] = 100.0
    p.write_text(json.dumps(rec0) + "\n", encoding="utf-8")

    iter_start = time.time()
    # In-iter entries
    append_tried(p, expr="rank(open)", sharpe=1.5, fitness=1.2, turnover=0.18,
                 alpha_id="A1", status="COMPLETE")
    append_tried(p, expr="rank(volume)", sharpe=0.5, fitness=0.4, turnover=0.3,
                 alpha_id="A2", status="COMPLETE")
    iter_end = time.time()

    review = _build_iter_review(
        tag=tag, start_ts=iter_start, end_ts=iter_end,
        sharpe_min=1.25, fitness_min=1.0,
    )

    assert review["iter_simulated"] == 2, "should exclude pre-iter entry A0"
    assert review["iter_completed"] == 2
    assert review["iter_passed"] == 1, "only A1 passes thresholds"
    assert "A1" in review["passed_alpha_ids"]
    assert review["top_3_by_fitness"][0]["alpha_id"] == "A1"


def test_build_iter_review_counts_errors(isolated_artifacts):
    tag = "smoke"
    p = tried_exprs_path(tag)
    iter_start = time.time()
    append_tried(p, expr="rank(close)", sharpe=None, fitness=None, turnover=None,
                 alpha_id=None, status="ERROR", error="compile failed")
    append_tried(p, expr="rank(open)", sharpe=1.0, fitness=0.5, turnover=0.2,
                 alpha_id="A1", status="COMPLETE")
    iter_end = time.time()

    review = _build_iter_review(
        tag=tag, start_ts=iter_start, end_ts=iter_end,
        sharpe_min=1.25, fitness_min=1.0,
    )
    assert review["iter_simulated"] == 2
    assert review["iter_completed"] == 1
    assert review["iter_errored"] == 1
    assert review["iter_passed"] == 0


def test_build_iter_review_empty_returns_zero(isolated_artifacts):
    review = _build_iter_review(
        tag="empty", start_ts=time.time(), end_ts=time.time() + 1,
        sharpe_min=1.25, fitness_min=1.0,
    )
    assert review["iter_simulated"] == 0
    assert review["top_3_by_fitness"] == []
    assert review["passed_alpha_ids"] == []


def test_format_review_oneline_includes_key_fields():
    review = {
        "tag": "t1",
        "iter_simulated": 5,
        "iter_completed": 4,
        "iter_passed": 1,
        "iter_errored": 1,
        "top_3_by_fitness": [{"alpha_id": "A1", "sh": 1.5, "fi": 1.2, "to": 0.18}],
        "pool_size_after": 3,
    }
    line = _format_review_oneline(review)
    assert "tag=t1" in line
    assert "sim=5" in line
    assert "complete=4" in line
    assert "passed=1" in line
    assert "err=1" in line
    assert "top_id=A1" in line
    assert "pool=3" in line
