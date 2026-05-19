"""quota_monitor tests — record / check / summary + persistence."""
from __future__ import annotations

import json

import pytest

from agent_market.wq_brain.quota_monitor import (
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


# ── reserve_action / release_action (TOCTOU closure) ────────────────────


def test_reserve_action_increments_atomically(isolated_artifacts):
    from agent_market.wq_brain.quota_monitor import (
        get_usage, release_action, reserve_action,
    )
    r = reserve_action("simulate", day="2025-12-01",
                       soft_limit=100, hard_limit=200)
    assert r["status"] == "ok"
    assert r["reserved"] is True
    assert r["count"] == 1
    assert get_usage("2025-12-01").counts["simulate"] == 1
    # Releasing rolls back when the operation aborted before producing a
    # billable side-effect
    release_action("simulate", day="2025-12-01")
    assert get_usage("2025-12-01").counts["simulate"] == 0


def test_reserve_action_blocks_at_hard_limit(isolated_artifacts):
    from agent_market.wq_brain.quota_monitor import (
        get_usage, record_action, reserve_action,
    )
    record_action("simulate", day="2025-12-01", n=200)
    r = reserve_action("simulate", day="2025-12-01",
                       soft_limit=100, hard_limit=200)
    assert r["status"] == "block"
    assert r["reserved"] is False
    # Counter unchanged when blocked
    assert get_usage("2025-12-01").counts["simulate"] == 200


def test_reserve_action_returns_throttle_above_soft(isolated_artifacts):
    from agent_market.wq_brain.quota_monitor import (
        record_action, reserve_action,
    )
    record_action("simulate", day="2025-12-01", n=99)
    r = reserve_action("simulate", day="2025-12-01",
                       soft_limit=100, hard_limit=200)
    # 99 + 1 = 100 ≥ soft, < hard → throttle, reserved=True
    assert r["status"] == "throttle"
    assert r["reserved"] is True


def test_release_action_never_below_zero(isolated_artifacts):
    from agent_market.wq_brain.quota_monitor import (
        get_usage, release_action,
    )
    # release without prior reserve — counter must clamp at 0
    release_action("simulate", day="2025-12-01")
    assert get_usage("2025-12-01").counts.get("simulate", 0) == 0


def test_quota_path_uses_utc_by_default(isolated_artifacts, monkeypatch):
    """quota_path() with day=None must derive the day from UTC, not local."""
    import time as _t
    from agent_market.wq_brain.quota_monitor import quota_path

    # Pin gmtime to 2026-06-15 23:30:00 UTC
    fixed = _t.struct_time((2026, 6, 15, 23, 30, 0, 0, 0, 0))
    monkeypatch.setattr(_t, "gmtime", lambda *a: fixed)
    p = quota_path()
    assert p.name == "2026-06-15.json"


def test_process_lock_no_deadlock_on_windows_path(monkeypatch, isolated_artifacts):
    """When fcntl is unavailable (Windows), _process_lock must NOT return
    the same threading lock the caller already holds — that would deadlock
    on the second `with`. The fix returns a NoopLock instead.
    """
    from agent_market.wq_brain import quota_monitor as qm

    # Force the fcntl-import branch to fail
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *a, **kw):
        if name == "fcntl":
            raise ImportError("simulated Windows: no fcntl")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", fake_import)

    lock = qm._process_lock(day="2025-12-01")
    # Must not be the threading lock — that's caller's job
    assert lock is not qm._LOCK
    # Must be a context manager that doesn't acquire anything
    with lock:
        # Now also acquire _LOCK as production code does — must NOT deadlock
        with qm._LOCK:
            pass


def test_reserve_action_works_under_simulated_windows(monkeypatch, isolated_artifacts):
    """End-to-end reserve under no-fcntl branch (Windows) — no deadlock."""
    from agent_market.wq_brain import quota_monitor as qm

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *a, **kw):
        if name == "fcntl":
            raise ImportError("simulated Windows: no fcntl")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", fake_import)
    r = qm.reserve_action("simulate", day="2025-12-01",
                          soft_limit=100, hard_limit=200)
    assert r["status"] == "ok"
    assert r["count"] == 1


def test_release_with_explicit_day_decrements_correct_bucket(isolated_artifacts):
    """Refund crossing UTC midnight must decrement the same bucket the
    reservation incremented, not 'today'.

    Simulates the failure: reserve on day A, then refund without passing
    day A — refund accidentally hits day B instead.
    """
    from agent_market.wq_brain.quota_monitor import (
        get_usage, record_action, release_action, reserve_action,
    )
    record_action("submit", day="2025-12-01", n=10)
    r = reserve_action("submit", day="2025-12-01",
                       soft_limit=20, hard_limit=50)
    assert r["status"] == "ok"
    assert r["day"] == "2025-12-01"
    # Now release pinning the day — the original bucket should drop by 1
    release_action("submit", day=r["day"])
    assert get_usage("2025-12-01").counts["submit"] == 10
    # And the next-day bucket must stay at 0 (no accidental decrement)
    assert get_usage("2025-12-02").counts.get("submit", 0) == 0


def test_check_quota_message_says_utc(isolated_artifacts):
    """The 'block' recommendation must reference 'UTC midnight'."""
    from agent_market.wq_brain.quota_monitor import check_quota, record_action
    record_action("simulate", day="2025-12-01", n=200)
    r = check_quota("simulate", day="2025-12-01",
                    soft_limit=100, hard_limit=200)
    assert "UTC" in r["recommendation"]


def test_tried_log_checkpoint_carries_extra_metadata(tmp_path):
    from agent_market.wq_brain.tried_log import read_checkpoint, write_checkpoint
    p = tmp_path / "tried.jsonl"
    write_checkpoint(
        p, session_id="s", last_iter=1,
        extra={"run_dir": "/tmp/run1", "tag": "wqb_v6_loop"},
    )
    ck = read_checkpoint(p)
    assert ck["extra"]["tag"] == "wqb_v6_loop"
