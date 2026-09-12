"""Tests for pool salvage CLI + cmd_simulate auto-persist.

Production data showed ~70% loss rate of high-fitness candidates: the
agent's LLM session computes fi=1.95 alphas but never submits them.
These tests cover the salvage hook (backfill from tried_log) and the
auto-persist hook (cmd_simulate writes pool UNSUBMITTED on the spot).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure scripts/ on path
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


def _run(cmd_func, **kwargs):
    """Run a cmd_* function with argparse.Namespace built from kwargs.
    Captures stdout (the JSON _emit) and returns the parsed dict + exit_code."""
    ns = argparse.Namespace(**kwargs)
    captured = io.StringIO()
    code_holder = {"code": 0}
    real_exit = sys.exit

    def fake_exit(c=0):
        code_holder["code"] = c
        raise SystemExit(c)

    with patch("sys.stdout", captured):
        with patch("sys.exit", fake_exit):
            try:
                cmd_func(ns)
            except SystemExit:
                pass
    out = captured.getvalue()
    try:
        parsed = json.loads(out) if out.strip() else None
    except ValueError:
        parsed = None
    return parsed, code_holder["code"]


# ── pool salvage ────────────────────────────────────────────────────────


def test_salvage_finds_lost_high_fi_candidates(isolated_artifacts):
    """Seed tried_log with 4 high-fi candidates; pool empty.
    salvage should find all 4."""
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.tried_log import append_tried
    from wq_brain import cmd_pool_salvage

    tag = "salvage_t1"
    p = tried_exprs_path(tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("a1", 1.5, 1.2, 0.3, "rank(close)"),
        ("a2", 1.4, 1.1, 0.4, "rank(volume)"),
        ("a3", 1.9, 1.5, 0.2, "ts_decay_linear(rank(returns), 20)"),
        ("a4", 2.0, 1.95, 0.18, "ts_decay_linear(rank(sales/equity), 20)"),
        # below thresholds:
        ("a5", 1.0, 0.5, 0.3, "rank(open)"),
        ("a6", 1.5, 0.9, 0.3, "rank(high)"),  # fi too low
    ]
    for aid, sh, fi, to, ex in rows:
        append_tried(p, expr=ex, sharpe=sh, fitness=fi, turnover=to,
                     alpha_id=aid, status="COMPLETE")

    out, code = _run(cmd_pool_salvage, tag=tag, sharpe_min=1.25,
                     fitness_min=1.0, top_n=0, dry_run=False)
    assert out["ok"] is True
    assert out["salvaged"] == 4
    assert out["pool_after"] == 4

    # Verify pool actually got the 4 alphas
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool
    pool = AlphaPool(alpha_pool_path(tag))
    ids = {e.alpha_id for e in pool.entries}
    assert ids == {"a1", "a2", "a3", "a4"}
    # All marked UNSUBMITTED so a later submit pass can pick them up
    for e in pool.entries:
        assert e.verified_status == "UNSUBMITTED"
        assert e.source == "salvage"


def test_salvage_skips_alpha_already_in_pool(isolated_artifacts):
    """Alphas already in pool must NOT be re-added (idempotent)."""
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.paths import alpha_pool_path, tried_exprs_path
    from agent_market.wq_brain.pool import AlphaPool
    from agent_market.wq_brain.tried_log import append_tried
    from wq_brain import cmd_pool_salvage

    tag = "salvage_t2"
    p = tried_exprs_path(tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    append_tried(p, expr="rank(close)", sharpe=1.5, fitness=1.2, turnover=0.3,
                 alpha_id="X1", status="COMPLETE")
    # Pre-seed pool with X1
    pool = AlphaPool(alpha_pool_path(tag))
    pool.add(AlphaPoolEntry(
        alpha_id="X1", expr="rank(close)", settings_dict={},
        sharpe=1.5, fitness=1.2, returns=0.0, turnover=0.3,
        tag=tag, source="agent", verified_status="ACTIVE",
        verified_at=1.0, rejection_reasons=[],
    ))

    out, code = _run(cmd_pool_salvage, tag=tag, sharpe_min=1.25,
                     fitness_min=1.0, top_n=0, dry_run=False)
    assert out["salvaged"] == 0
    assert out["pool_after"] == 1


def test_salvage_dry_run_does_not_modify_pool(isolated_artifacts):
    from agent_market.wq_brain.paths import alpha_pool_path, tried_exprs_path
    from agent_market.wq_brain.pool import AlphaPool
    from agent_market.wq_brain.tried_log import append_tried
    from wq_brain import cmd_pool_salvage

    tag = "salvage_t3"
    p = tried_exprs_path(tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    append_tried(p, expr="rank(close)", sharpe=1.5, fitness=1.2, turnover=0.3,
                 alpha_id="DR1", status="COMPLETE")

    out, code = _run(cmd_pool_salvage, tag=tag, sharpe_min=1.25,
                     fitness_min=1.0, top_n=0, dry_run=True)
    assert out["dry_run"] is True
    assert out["would_add"] == 1
    pool = AlphaPool(alpha_pool_path(tag))
    assert len(pool) == 0  # pool untouched


def test_salvage_top_n_limits_selection(isolated_artifacts):
    """top_n picks only the top-N by fitness."""
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.tried_log import append_tried
    from wq_brain import cmd_pool_salvage

    tag = "salvage_t4"
    p = tried_exprs_path(tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    for i, fi in enumerate([1.1, 1.5, 1.9, 1.3, 1.7], 1):
        append_tried(p, expr=f"e{i}", sharpe=1.5, fitness=fi, turnover=0.3,
                     alpha_id=f"K{i}", status="COMPLETE")

    out, code = _run(cmd_pool_salvage, tag=tag, sharpe_min=1.25,
                     fitness_min=1.0, top_n=2, dry_run=False)
    assert out["salvaged"] == 2

    # Top 2 by fitness should be K3 (1.9) and K5 (1.7)
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool
    pool = AlphaPool(alpha_pool_path(tag))
    ids = {e.alpha_id for e in pool.entries}
    assert ids == {"K3", "K5"}


def test_salvage_filters_incomplete_status(isolated_artifacts):
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.tried_log import append_tried
    from wq_brain import cmd_pool_salvage

    tag = "salvage_t5"
    p = tried_exprs_path(tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    append_tried(p, expr="bad", sharpe=2.0, fitness=1.5, turnover=0.3,
                 alpha_id="ERR", status="ERROR", error="boom")
    append_tried(p, expr="ok", sharpe=1.5, fitness=1.2, turnover=0.3,
                 alpha_id="OK1", status="COMPLETE")

    out, code = _run(cmd_pool_salvage, tag=tag, sharpe_min=1.25,
                     fitness_min=1.0, top_n=0, dry_run=False)
    assert out["salvaged"] == 1


def test_salvage_dedupes_alpha_id_within_tried(isolated_artifacts):
    """Multiple tried rows with same alpha_id should count once."""
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.tried_log import append_tried
    from wq_brain import cmd_pool_salvage

    tag = "salvage_t6"
    p = tried_exprs_path(tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        append_tried(p, expr="dup", sharpe=1.5, fitness=1.2, turnover=0.3,
                     alpha_id="DUP", status="COMPLETE")

    out, code = _run(cmd_pool_salvage, tag=tag, sharpe_min=1.25,
                     fitness_min=1.0, top_n=0, dry_run=False)
    assert out["salvaged"] == 1
