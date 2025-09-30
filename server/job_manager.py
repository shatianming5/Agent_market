from __future__ import annotations

import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Job:
    id: str
    cmd: List[str]
    cwd: Optional[Path] = None
    env: Optional[dict] = None
    process: Optional[subprocess.Popen] = None
    logs: List[str] = field(default_factory=list)
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
                    if f:
                        try:
                            f.write(text + "\n")
                        except Exception:
                            pass
            finally:
                try:
                    f and f.close()
                except Exception:
                    pass
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
                                job.process.terminate()
                            except Exception:
                                pass
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
                "lines": len(job.logs),
                "status": ("running" if job.running else ("completed" if job.returncode is not None else "pending")),
                "code": (None if job.returncode is None else ("OK" if job.returncode == 0 else "SCRIPT_FAILED")),
                "kind": job.kind,
                "meta": job.meta,
            }

    def logs(self, job_id: str, offset: int = 0) -> dict:
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "job not found"}
        with self._lock:
            total = len(job.logs)
            start = max(0, min(offset, total))
            chunk = job.logs[start:]
            return {
                "id": job.id,
                "offset": start,
                "next": start + len(chunk),
                "running": job.running,
                "returncode": job.returncode,
                "status": ("running" if job.running else ("completed" if job.returncode is not None else "pending")),
                "code": (None if job.returncode is None else ("OK" if job.returncode == 0 else "SCRIPT_FAILED")),
                "kind": job.kind,
                "meta": job.meta,
                "logs": chunk,
            }

    def cancel(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "job not found"}
        with self._lock:
            if job.process and job.process.poll() is None:
                try:
                    job.process.terminate()
                    job.logs.append('[job] cancelled by user')
                    job.running = False
                    return {"status": "terminated"}
                except Exception as exc:
                    return {"error": str(exc)}
            return {"status": "not_running"}
