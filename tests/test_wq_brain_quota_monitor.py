"""quota_monitor tests — record / check / summary + persistence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_market.wq_brain.quota_monitor import (
    QuotaUsage,
    check_quota,
    get_usage,
    quota_path,
    quota_summary,
    record_action,
)


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


# ── record_action / get_usage ───────────────────────────────────────────


def test_record_action_increments_counter(isolated_artifacts):
    record_action("simulate", day="2025-12-01")
    record_action("simulate", day="2025-12-01")
    record_action("submit", day="2025-12-01")
    u = get_usage("2025-12-01")
    assert u.counts["simulate"] == 2
    assert u.counts["submit"] == 1


def test_record_action_persists_atomically(isolated_artifacts):
    record_action("simulate", day="2025-12-01", n=5)
    p = quota_path("2025-12-01")
    assert p.exists()
    data = json.loads(p.read_text())
    assert data["counts"]["simulate"] == 5


def test_record_action_separates_days(isolated_artifacts):
    record_action("simulate", day="2025-12-01", n=3)
    record_action("simulate", day="2025-12-02", n=1)
    assert get_usage("2025-12-01").counts["simulate"] == 3
    assert get_usage("2025-12-02").counts["simulate"] == 1


def test_get_usage_missing_day_returns_empty(isolated_artifacts):
    u = get_usage("2099-01-01")
    assert u.counts == {}
    assert u.day == "2099-01-01"


# ── check_quota ─────────────────────────────────────────────────────────


def test_check_quota_ok_when_below_soft(isolated_artifacts):
    record_action("simulate", day="2025-12-01", n=10)
    r = check_quota("simulate", day="2025-12-01",
                    soft_limit=100, hard_limit=200)
    assert r["status"] == "ok"
    assert r["count"] == 10
    assert r["remaining"] == 190


def test_check_quota_throttle_at_soft_limit(isolated_artifacts):
    record_action("simulate", day="2025-12-01", n=100)
    r = check_quota("simulate", day="2025-12-01",
                    soft_limit=100, hard_limit=200)
    assert r["status"] == "throttle"
    assert "Approaching" in r["recommendation"]


def test_check_quota_block_at_hard_limit(isolated_artifacts):
    record_action("simulate", day="2025-12-01", n=200)
    r = check_quota("simulate", day="2025-12-01",
                    soft_limit=100, hard_limit=200)
    assert r["status"] == "block"
    assert "exhausted" in r["recommendation"]


def test_check_quota_uses_default_limits_when_unspecified(isolated_artifacts, monkeypatch):
    monkeypatch.setenv("WQ_QUOTA_SIM_HARD", "10")
    record_action("simulate", day="2025-12-01", n=11)
    r = check_quota("simulate", day="2025-12-01")
    # n=11 above hard_limit=10 → block
    assert r["status"] == "block"
    assert r["hard_limit"] == 10


def test_check_quota_unknown_action_passes(isolated_artifacts):
    r = check_quota("unknown_action", day="2025-12-01")
    # Never recorded → count=0; default limits = 10^9 → "ok"
    assert r["status"] == "ok"
    assert r["count"] == 0


# ── quota_summary ───────────────────────────────────────────────────────


def test_quota_summary_includes_recorded_actions(isolated_artifacts):
    record_action("simulate", day="2025-12-01", n=5)
    record_action("submit", day="2025-12-01", n=2)
    s = quota_summary("2025-12-01")
    assert s["day"] == "2025-12-01"
    assert s["actions"]["simulate"]["count"] == 5
    assert s["actions"]["submit"]["count"] == 2


def test_quota_summary_always_includes_simulate_and_submit(isolated_artifacts):
    s = quota_summary("2099-12-31")
    # Even with no recorded action, simulate + submit should be reported
    # (so the agent always sees both quota lines)
    assert "simulate" in s["actions"]
    assert "submit" in s["actions"]


# ── tried_log checkpoint ───────────────────────────────────────────────


def test_tried_log_checkpoint_roundtrip(isolated_artifacts, tmp_path):
    from agent_market.wq_brain.tried_log import (
        checkpoint_path, read_checkpoint, write_checkpoint,
    )
    p = tmp_path / "tried_exprs.jsonl"
    write_checkpoint(p, session_id="abc-123", last_iter=42)
    ck = read_checkpoint(p)
    assert ck["session_id"] == "abc-123"
    assert ck["last_iter"] == 42
    assert ck["ts"] > 0
    # Sidecar lives next to the tried_exprs file, not on top of it
    assert checkpoint_path(p).name.endswith(".checkpoint.json")
    assert not p.exists()  # tried_exprs.jsonl untouched


def test_tried_log_checkpoint_overwrites_atomically(tmp_path):
    from agent_market.wq_brain.tried_log import read_checkpoint, write_checkpoint
    p = tmp_path / "tried.jsonl"
    write_checkpoint(p, session_id="s1", last_iter=10)
    write_checkpoint(p, session_id="s2", last_iter=11)
    ck = read_checkpoint(p)
    assert ck["session_id"] == "s2"
    assert ck["last_iter"] == 11


def test_tried_log_checkpoint_missing_returns_none(tmp_path):
    from agent_market.wq_brain.tried_log import read_checkpoint
    assert read_checkpoint(tmp_path / "nope.jsonl") is None


def test_tried_log_checkpoint_corrupt_returns_none(tmp_path):
    from agent_market.wq_brain.tried_log import checkpoint_path, read_checkpoint
    p = tmp_path / "tried.jsonl"
    cp = checkpoint_path(p)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text("{not valid json", encoding="utf-8")
    assert read_checkpoint(p) is None


def test_tried_log_checkpoint_carries_extra_metadata(tmp_path):
    from agent_market.wq_brain.tried_log import read_checkpoint, write_checkpoint
    p = tmp_path / "tried.jsonl"
    write_checkpoint(
        p, session_id="s", last_iter=1,
        extra={"run_dir": "/tmp/run1", "tag": "wqb_v6_loop"},
    )
    ck = read_checkpoint(p)
    assert ck["extra"]["tag"] == "wqb_v6_loop"
