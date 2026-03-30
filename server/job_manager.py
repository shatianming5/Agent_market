from __future__ import annotations
import logging

import subprocess
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

# Maximum number of log lines kept in memory per job.
LOG_RING_SIZE = 5000


@dataclass
class Job:
    id: str
    cmd: List[str]
    cwd: Optional[Path] = None
    env: Optional[dict] = None
    process: Optional[subprocess.Popen] = None
    logs: Deque[str] = field(default_factory=lambda: deque(maxlen=LOG_RING_SIZE))
    total_lines: int = 0  # monotonic counter — survives ring eviction
    returncode: Optional[int] = None
    running: bool = False
    logfile: Optional[Path] = None
    timeout_sec: Optional[int] = None
    kind: Optional[str] = None
    meta: Optional[dict] = None


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None, timeout_sec: Optional[int] = None, kind: Optional[str] = None, meta: Optional[dict] = None) -> str:
        job_id = uuid.uuid4().hex[:12]
        logdir = Path('user_data') / 'job_logs'
        logdir.mkdir(parents=True, exist_ok=True)
        logfile = logdir / f"{job_id}.log"
        job = Job(id=job_id, cmd=cmd, cwd=cwd, env=env, logfile=logfile, timeout_sec=timeout_sec, kind=kind, meta=meta)
        self._jobs[job_id] = job
        self._spawn(job)
        return job_id

    def _spawn(self, job: Job) -> None:
        job.running = True
        proc = subprocess.Popen(
            job.cmd,
            cwd=str(job.cwd) if job.cwd else None,
            env=job.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
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
                                job.process.terminate()
                            except Exception as _exc:
                                logging.getLogger(__name__).debug("Suppressed: %s", _exc)
            threading.Thread(target=_guard, daemon=True).start()

    def status(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "job not found"}
        with self._lock:
            return {
                "id": job.id,
                "cmd": job.cmd,
                "running": job.running,
                "returncode": job.returncode,
                "lines": job.total_lines,
                "status": ("running" if job.running else ("completed" if job.returncode is not None else "pending")),
                "code": (None if job.returncode is None else ("OK" if job.returncode == 0 else "SCRIPT_FAILED")),
                "kind": job.kind,
                "meta": job.meta,
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
            return {"error": "job not found"}
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
                "status": ("running" if job.running else ("completed" if job.returncode is not None else "pending")),
                "code": (None if job.returncode is None else ("OK" if job.returncode == 0 else "SCRIPT_FAILED")),
                "kind": job.kind,
                "meta": job.meta,
                "logs": chunk,
            }
            if truncated:
                result["truncated"] = True
            return result

    def cancel(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "job not found"}
        with self._lock:
            if job.process and job.process.poll() is None:
                try:
                    job.process.terminate()
                    job.logs.append('[job] cancelled by user')
                    job.total_lines += 1
                    job.running = False
                    return {"status": "terminated"}
                except Exception as exc:
                    return {"error": str(exc)}
            return {"status": "not_running"}
