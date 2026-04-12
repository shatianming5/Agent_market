from __future__ import annotations
import logging

import json
import os
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional

# Maximum number of log lines kept in memory per job.
LOG_RING_SIZE = 5000
DEFAULT_KEEP_RECENT = 200


class JobQueueFullError(RuntimeError):
    """Raised when the job queue is full and a new job cannot be accepted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    cmd: List[str]
    cwd: Optional[Path] = None
    env: Optional[dict] = None
    started_at: str = field(default_factory=_utc_now)
    process: Optional[subprocess.Popen] = None
    logs: Deque[str] = field(default_factory=lambda: deque(maxlen=LOG_RING_SIZE))
    total_lines: int = 0  # monotonic counter — survives ring eviction
    returncode: Optional[int] = None
    running: bool = False
    logfile: Optional[Path] = None
    timeout_sec: Optional[int] = None
    kind: Optional[str] = None
    meta: Optional[dict] = None
    cancelled: bool = False
    ended_at: Optional[str] = None
    created_at_epoch: float = field(default_factory=time.time)
    queued: bool = False


class JobManager:
    def __init__(
        self,
        *,
        log_dir: Path,
        registry_dir: Path,
        keep_recent: int = DEFAULT_KEEP_RECENT,
        max_running: int = 0,
        max_queued: int = 0,
    ) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._log_dir = Path(log_dir).resolve()
        self._registry_dir = Path(registry_dir).resolve()
        self._keep_recent = int(keep_recent) if int(keep_recent) > 0 else DEFAULT_KEEP_RECENT
        self._max_running = int(max_running) if int(max_running) >= 0 else 0
        self._max_queued = int(max_queued) if int(max_queued) >= 0 else 0
        self._queue: Deque[str] = deque()

    def _running_count_locked(self) -> int:
        return sum(1 for job in self._jobs.values() if job.running)

    def _registry_path(self, job_id: str) -> Path:
        return (self._registry_dir / f"{job_id}.json").resolve()

    def _write_registry(self, job: Job) -> None:
        try:
            self._registry_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "id": job.id,
                "kind": job.kind,
                "cmd": list(job.cmd),
                "cwd": str(job.cwd) if job.cwd else None,
                "started_at": job.started_at,
                "ended_at": job.ended_at,
                "running": bool(job.running),
                "queued": bool(job.queued),
                "returncode": job.returncode,
                "cancelled": bool(job.cancelled),
                "logfile": str(job.logfile) if job.logfile else None,
                "timeout_sec": job.timeout_sec,
                "meta": job.meta,
                "total_lines": job.total_lines,
            }
            path = self._registry_path(job.id)
            tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            logging.getLogger(__name__).debug("Failed to write job registry", exc_info=True)

    def _load_registry(self, job_id: str) -> Optional[dict]:
        path = self._registry_path(job_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _terminate_process(self, proc: subprocess.Popen, *, kill: bool = False) -> None:
        """Terminate a process, preferring process-group termination on Unix."""
        try:
            if proc.poll() is not None:
                return
            if os.name != "nt":
                pgid = proc.pid
                os.killpg(pgid, signal.SIGKILL if kill else signal.SIGTERM)
                return
        except Exception:
            pass
        try:
            (proc.kill() if kill else proc.terminate())
        except Exception:
            pass

    def _prune_locked(self) -> None:
        """Drop old completed jobs from memory (disk registry keeps the record)."""
        if self._keep_recent <= 0:
            return
        completed = [job for job in self._jobs.values() if not job.running and job.returncode is not None]
        if len(completed) <= self._keep_recent:
            return
        completed.sort(key=lambda j: float(j.created_at_epoch))
        for job in completed[: max(0, len(completed) - self._keep_recent)]:
            self._jobs.pop(job.id, None)

    def start(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
        timeout_sec: Optional[int] = None,
        kind: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> str:
        job_id = uuid.uuid4().hex[:12]
        logdir = self._log_dir
        logdir.mkdir(parents=True, exist_ok=True)
        logfile = (logdir / f"{job_id}.log").resolve()
        try:
            logfile.touch(exist_ok=True)
        except Exception:
            pass
        job = Job(id=job_id, cmd=cmd, cwd=cwd, env=env, logfile=logfile, timeout_sec=timeout_sec, kind=kind, meta=meta)
        should_spawn = True
        with self._lock:
            running = self._running_count_locked()
            if self._max_running > 0 and running >= self._max_running:
                should_spawn = False
                if self._max_queued > 0 and len(self._queue) < self._max_queued:
                    job.queued = True
                    self._jobs[job_id] = job
                    self._queue.append(job_id)
                    self._write_registry(job)
                    return job_id
                raise JobQueueFullError(
                    f"job queue full (running={running}/{self._max_running}, queued={len(self._queue)}/{self._max_queued})"
                )
            self._jobs[job_id] = job
            self._write_registry(job)
        if should_spawn:
            self._spawn(job)
        return job_id

    def _spawn(self, job: Job) -> None:
        job.running = True
        job.queued = False
        proc = subprocess.Popen(
            job.cmd,
            cwd=str(job.cwd) if job.cwd else None,
            env=job.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            start_new_session=(os.name != "nt"),
        )
        job.process = proc

        def _reader() -> None:
            try:
                assert proc.stdout is not None
                f = job.logfile.open('a', encoding='utf-8') if job.logfile else None
                for line in proc.stdout:
                    text = line.rstrip("\n")
                    with self._lock:
                        job.logs.append(text)
                        job.total_lines += 1
                        if job.total_lines % 50 == 0:
                            self._write_registry(job)
                    if f:
                        try:
                            f.write(text + "\n")
                        except Exception as _exc:
                            logging.getLogger(__name__).debug("Suppressed: %s", _exc)
            finally:
                try:
                    f and f.close()
                except Exception as _exc:
                    logging.getLogger(__name__).debug("Suppressed: %s", _exc)
                proc.wait()
                with self._lock:
                    job.returncode = proc.returncode
                    job.running = False
                    job.ended_at = _utc_now()
                    self._write_registry(job)
                    self._prune_locked()
                self._drain_queue()

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        # optional timeout guard
        if job.timeout_sec and job.timeout_sec > 0:
            def _guard():
                try:
                    proc.wait(timeout=job.timeout_sec)
                except Exception:
                    with self._lock:
                        if job.running and job.process and job.process.poll() is None:
                            try:
                                job.logs.append('[job] timeout reached, terminating')
                                job.total_lines += 1
                                self._write_registry(job)
                                self._terminate_process(job.process, kill=False)
                            except Exception as _exc:
                                logging.getLogger(__name__).debug("Suppressed: %s", _exc)
                    # give it a moment, then hard-kill if still alive
                    time.sleep(2.0)
                    with self._lock:
                        if job.running and job.process and job.process.poll() is None:
                            self._terminate_process(job.process, kill=True)
            threading.Thread(target=_guard, daemon=True).start()

    def _drain_queue(self) -> None:
        """Start queued jobs if capacity is available."""
        while True:
            with self._lock:
                if not self._queue:
                    return
                running = self._running_count_locked()
                if self._max_running > 0 and running >= self._max_running:
                    return
                job_id = self._queue.popleft()
                job = self._jobs.get(job_id)
                if not job:
                    continue
                if job.cancelled or job.returncode is not None:
                    continue
                job.queued = False
                self._write_registry(job)
            # Spawn outside the lock.
            self._spawn(job)

    def status(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if not job:
            payload = self._load_registry(job_id)
            if payload:
                cancelled = bool(payload.get("cancelled"))
                return {
                    "id": payload.get("id") or job_id,
                    "cmd": payload.get("cmd") or [],
                    "running": bool(payload.get("running")),
                    "returncode": payload.get("returncode"),
                    "lines": int(payload.get("total_lines") or 0),
                    "status": (
                        "running"
                        if payload.get("running")
                        else ("queued" if payload.get("queued") else "completed")
                    ),
                    "code": (
                        "CANCELLED"
                        if cancelled
                        else (
                            None
                            if payload.get("returncode") is None
                            else ("OK" if int(payload.get("returncode") or 0) == 0 else "SCRIPT_FAILED")
                        )
                    ),
                    "kind": payload.get("kind"),
                    "meta": payload.get("meta"),
                    "queued": bool(payload.get("queued")),
                    "cancelled": cancelled,
                }
            return {"error": "job not found"}
        with self._lock:
            cancelled = bool(job.cancelled)
            return {
                "id": job.id,
                "cmd": job.cmd,
                "running": job.running,
                "returncode": job.returncode,
                "lines": job.total_lines,
                "status": (
                    "running"
                    if job.running
                    else ("queued" if job.queued else ("completed" if job.returncode is not None else "pending"))
                ),
                "code": (
                    "CANCELLED"
                    if cancelled
                    else (None if job.returncode is None else ("OK" if job.returncode == 0 else "SCRIPT_FAILED"))
                ),
                "kind": job.kind,
                "meta": job.meta,
                "queued": bool(job.queued),
                "cancelled": cancelled,
            }

    def logs(self, job_id: str, offset: int = 0, limit: int = 0) -> dict:
        """Return log lines starting from *offset* (global line number).

        *limit* caps the number of returned lines (0 = unlimited).
        When the ring buffer has evicted old lines, the earliest available
        offset is ``total_lines - len(logs)``; if the caller's offset is
        below that, a ``truncated`` flag is set and reading starts from the
        earliest available line.
        """
        job = self._jobs.get(job_id)
        if not job:
            payload = self._load_registry(job_id) or {}
            logfile_raw = payload.get("logfile")
            logfile = Path(str(logfile_raw)).resolve() if logfile_raw else None
            if logfile is None or not logfile.exists():
                return {"error": "job not found"}
            # Best-effort: load log file and slice by global offset.
            try:
                lines = logfile.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                lines = []
            total = len(lines)
            start = max(0, int(offset or 0))
            chunk = lines[start : (start + int(limit))] if limit and int(limit) > 0 else lines[start:]
            rc = payload.get("returncode")
            running = bool(payload.get("running"))
            cancelled = bool(payload.get("cancelled"))
            return {
                "id": payload.get("id") or job_id,
                "offset": start,
                "next": start + len(chunk),
                "total_lines": total,
                "running": running,
                "returncode": rc,
                "status": ("running" if running else ("queued" if payload.get("queued") else "completed")),
                "code": (
                    "CANCELLED"
                    if cancelled
                    else (None if rc is None else ("OK" if int(rc or 0) == 0 else "SCRIPT_FAILED"))
                ),
                "kind": payload.get("kind"),
                "meta": payload.get("meta"),
                "queued": bool(payload.get("queued")),
                "cancelled": cancelled,
                "logs": chunk,
            }
        with self._lock:
            buf = job.logs  # deque
            buf_len = len(buf)
            total = job.total_lines
            earliest = total - buf_len  # first global index still in buffer

            truncated = False
            if offset < earliest:
                offset = earliest
                truncated = True

            # Convert global offset to local index within deque.
            local_start = offset - earliest
            if limit > 0:
                chunk = list(buf)[local_start : local_start + limit]
            else:
                chunk = list(buf)[local_start:]

            result: dict = {
                "id": job.id,
                "offset": offset,
                "next": offset + len(chunk),
                "total_lines": total,
                "running": job.running,
                "returncode": job.returncode,
                "status": (
                    "running"
                    if job.running
                    else ("queued" if job.queued else ("completed" if job.returncode is not None else "pending"))
                ),
                "code": (
                    "CANCELLED"
                    if job.cancelled
                    else (None if job.returncode is None else ("OK" if job.returncode == 0 else "SCRIPT_FAILED"))
                ),
                "kind": job.kind,
                "meta": job.meta,
                "logs": chunk,
                "queued": bool(job.queued),
                "cancelled": bool(job.cancelled),
            }
            if truncated:
                result["truncated"] = True
            return result

    def cancel(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "job not found"}
        with self._lock:
            if job.queued and (job.process is None):
                try:
                    self._queue.remove(job_id)
                except ValueError:
                    pass
                job.cancelled = True
                job.queued = False
                job.running = False
                job.returncode = -1
                job.ended_at = _utc_now()
                job.logs.append("[job] cancelled while queued")
                job.total_lines += 1
                self._write_registry(job)
                return {"status": "cancelled"}
            if job.process and job.process.poll() is None:
                try:
                    job.cancelled = True
                    self._terminate_process(job.process, kill=False)
                    job.logs.append('[job] cancelled by user')
                    job.total_lines += 1
                    job.running = False
                    job.ended_at = _utc_now()
                    self._write_registry(job)
                    return {"status": "terminated"}
                except Exception as exc:
                    return {"error": str(exc)}
            return {"status": "not_running"}
