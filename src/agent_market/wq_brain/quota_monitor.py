"""Per-day WQ quota tracking + advisory throttle.

WQ BRAIN doesn't expose a usage endpoint — clients have to count their
own simulate / submit calls and back off before they hit the daily cap.
This module persists per-day counters under
``artifacts/wq_brain/quota/{YYYY-MM-DD}.json`` and exposes an advisory
:func:`check_quota` so callers can preemptively switch to ``local_sim``
when they're near the cap.

API:

  * :func:`record_action` — atomically bump a counter (call this *after*
    a successful CLI hit). Multi-process safe via fsync + replace.
  * :func:`get_usage` — read today's counter dict.
  * :func:`check_quota` — return ``"ok"`` | ``"throttle"`` | ``"block"``
    for a given action with caller-supplied soft / hard limits.

Defaults match WQ's documented daily caps for the BRAIN free / Bronze
tiers (1500 simulations / day, ~50 submissions / day) but caller can
override via env or kwargs to match their actual subscription.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .paths import wq_brain_root

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


# ── Defaults (overridable via env) ─────────────────────────────────────


_UNBOUNDED = 10**9


def _default_limits() -> dict[str, tuple[int, int]]:
    """Return (soft, hard) per-action limits.

    Default is **unbounded** — WQ BRAIN does not publish a documented daily
    cap, and the prior conservative ceilings (1500 sim / 50 submit) were
    purely client-side advisory and could starve the agent prematurely.
    Operators who want a real cap set ``WQ_QUOTA_SIM_HARD`` /
    ``WQ_QUOTA_SUBMIT_HARD`` env vars. ``check_quota`` and
    ``reserve_action`` still record per-day counts for telemetry.
    """
    sim_hard_env = os.environ.get("WQ_QUOTA_SIM_HARD")
    sub_hard_env = os.environ.get("WQ_QUOTA_SUBMIT_HARD")
    sim_hard = int(sim_hard_env) if sim_hard_env else _UNBOUNDED
    sub_hard = int(sub_hard_env) if sub_hard_env else _UNBOUNDED
    # Soft = 90% of hard when bounded, else _UNBOUNDED (never throttles)
    sim_soft = max(1, int(sim_hard * 0.9)) if sim_hard < _UNBOUNDED else _UNBOUNDED
    sub_soft = max(1, int(sub_hard * 0.9)) if sub_hard < _UNBOUNDED else _UNBOUNDED
    return {
        "simulate": (sim_soft, sim_hard),
        "submit":   (sub_soft, sub_hard),
    }


# ── Storage layout ─────────────────────────────────────────────────────


def quota_root() -> Path:
    return wq_brain_root() / "quota"


def quota_path(day: Optional[str] = None) -> Path:
    """Return the per-day quota file path. ``day`` is a UTC ``YYYY-MM-DD``
    string. Defaults to today (UTC) so the documented "midnight UTC reset"
    (see ``check_quota`` recommendation) actually matches the storage key.
    """
    day = day or time.strftime("%Y-%m-%d", time.gmtime())
    return quota_root() / f"{day}.json"


# ── Counter dataclass ──────────────────────────────────────────────────


@dataclass
class QuotaUsage:
    day: str
    counts: dict[str, int] = field(default_factory=dict)
    last_updated: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _read(path: Path) -> QuotaUsage:
    if not path.exists():
        return QuotaUsage(day=path.stem, counts={}, last_updated=0.0)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return QuotaUsage(
            day=data.get("day", path.stem),
            counts=dict(data.get("counts", {})),
            last_updated=float(data.get("last_updated", 0.0)),
        )
    except (ValueError, OSError):
        return QuotaUsage(day=path.stem, counts={}, last_updated=0.0)


def _write_atomic(path: Path, usage: QuotaUsage) -> None:
    """Atomic write with file + parent-dir fsync — durable even on power-loss.

    The previous version skipped fsync, so a multi-process race could land a
    half-written tmp on disk before rename. This version:

      1. write payload to ``<path>.tmp``
      2. fsync the tmp file
      3. rename onto target
      4. fsync the parent directory so the rename is durable
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, json.dumps(usage.to_dict(), indent=2).encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.replace(path)
    # parent dir fsync makes rename durable across crashes
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _process_lock_path(day: Optional[str]) -> Path:
    """Lock file path used for cross-process serialization."""
    return quota_path(day).with_suffix(".lock")


class _NoopLock:
    """Context manager that does nothing.

    Used as the cross-process lock on Windows / platforms lacking ``fcntl``;
    callers separately hold the in-process ``_LOCK`` (a threading.Lock) so
    single-process safety is still guaranteed. Returning the same threading
    lock from here would deadlock since callers do ``with _process_lock(),
    _LOCK:`` and threading.Lock is non-reentrant.
    """

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _process_lock(day: Optional[str]):
    """Cross-process advisory lock around quota counter updates.

    Uses fcntl on POSIX; on platforms without fcntl (Windows) returns a
    no-op so the caller's ``_LOCK`` (threading.Lock) still serializes
    *within* the process — best we can do without fcntl/msvcrt portability.
    Lock file lives next to the quota json so it rotates with the day.
    """
    try:
        import fcntl
    except ImportError:
        return _NoopLock()

    class _FileLock:
        def __enter__(self):
            self._path = _process_lock_path(day)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(str(self._path),
                                os.O_RDWR | os.O_CREAT, 0o644)
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            return self

        def __exit__(self, *a):
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
            return False

    return _FileLock()


# ── Public API ─────────────────────────────────────────────────────────


def record_action(action: str, *, day: Optional[str] = None, n: int = 1) -> QuotaUsage:
    """Atomically increment ``counts[action]`` for ``day`` (default = today UTC).

    Cross-process safe via fcntl advisory lock (POSIX) — multiple
    parallel ``wq_brain.py simulate`` invocations can't lose updates. On
    Windows the lock degrades to thread-only (single-process safety only).
    """
    p = quota_path(day)
    with _process_lock(day), _LOCK:
        usage = _read(p)
        usage.counts[action] = int(usage.counts.get(action, 0)) + n
        usage.last_updated = time.time()
        _write_atomic(p, usage)
    return usage


def reserve_action(
    action: str,
    *,
    day: Optional[str] = None,
    soft_limit: Optional[int] = None,
    hard_limit: Optional[int] = None,
) -> dict:
    """Atomically check + reserve a quota slot in one critical section.

    Closes the TOCTOU window between :func:`check_quota` and
    :func:`record_action` (caller could overspend if many processes
    raced their checks before recording). Returns ``check_quota``-shape
    dict; when ``status == "ok"`` or ``"throttle"`` the counter has been
    incremented and the caller MUST proceed (or call ``release_action``
    on failure). When ``status == "block"`` the counter is left unchanged.
    """
    p = quota_path(day)
    with _process_lock(day), _LOCK:
        usage = _read(p)
        defaults = _default_limits()
        soft, hard = defaults.get(action, (10**9, 10**9))
        if soft_limit is not None:
            soft = soft_limit
        if hard_limit is not None:
            hard = hard_limit
        if soft > hard:
            soft = hard
        count = int(usage.counts.get(action, 0))
        if count >= hard:
            return {
                "action": action, "day": usage.day, "count": count,
                "soft_limit": soft, "hard_limit": hard,
                "remaining": 0, "status": "block",
                "recommendation": (
                    f"Daily {action} quota exhausted ({count}/{hard}). "
                    "Switch to `local-simulate` until the next UTC midnight."
                ),
                "reserved": False,
            }
        # Reserve: increment NOW so a parallel reserve_action sees the bump
        usage.counts[action] = count + 1
        usage.last_updated = time.time()
        _write_atomic(p, usage)
    return {
        "action": action, "day": usage.day, "count": count + 1,
        "soft_limit": soft, "hard_limit": hard,
        "remaining": max(0, hard - count - 1),
        "status": "throttle" if (count + 1) >= soft else "ok",
        "recommendation": (
            f"Approaching daily {action} cap ({count + 1}/{hard}). "
            "Prefer `local-simulate` for screening."
            if (count + 1) >= soft else ""
        ),
        "reserved": True,
    }


def release_action(action: str, *, day: Optional[str] = None, n: int = 1) -> QuotaUsage:
    """Decrement a previously-reserved counter (call when the operation
    aborts before producing a billable side-effect).

    Idempotent at the storage layer: never goes below 0.
    """
    p = quota_path(day)
    with _process_lock(day), _LOCK:
        usage = _read(p)
        cur = int(usage.counts.get(action, 0))
        usage.counts[action] = max(0, cur - n)
        usage.last_updated = time.time()
        _write_atomic(p, usage)
    return usage


def get_usage(day: Optional[str] = None) -> QuotaUsage:
    return _read(quota_path(day))


def check_quota(
    action: str,
    *,
    day: Optional[str] = None,
    soft_limit: Optional[int] = None,
    hard_limit: Optional[int] = None,
) -> dict:
    """Advisory check: ``"ok"`` < soft ≤ ``"throttle"`` < hard ≤ ``"block"``.

    Returns a dict with status + count + remaining + recommendation.
    """
    defaults = _default_limits()
    soft, hard = defaults.get(action, (10**9, 10**9))
    if soft_limit is not None:
        soft = soft_limit
    if hard_limit is not None:
        hard = hard_limit
    if soft > hard:
        soft = hard
    usage = get_usage(day)
    count = int(usage.counts.get(action, 0))
    if count >= hard:
        status = "block"
        rec = (
            f"Daily {action} quota exhausted ({count}/{hard}). "
            "Stop spending remote calls — switch to `local-simulate` until "
            "the next UTC midnight."
        )
    elif count >= soft:
        status = "throttle"
        rec = (
            f"Approaching daily {action} cap ({count}/{hard}). Prefer "
            "`local-simulate` for screening; reserve remote for the strongest "
            "candidates."
        )
    else:
        status = "ok"
        rec = ""
    return {
        "action": action,
        "day": usage.day,
        "count": count,
        "soft_limit": soft,
        "hard_limit": hard,
        "remaining": max(0, hard - count),
        "status": status,
        "recommendation": rec,
    }


def quota_summary(day: Optional[str] = None) -> dict:
    """Roll up all tracked actions for one day into a single dict."""
    usage = get_usage(day)
    out = {"day": usage.day, "actions": {}}
    for action in usage.counts.keys() | {"simulate", "submit"}:
        out["actions"][action] = check_quota(action, day=day)
    return out
