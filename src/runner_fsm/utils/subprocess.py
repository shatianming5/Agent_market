from __future__ import annotations

import subprocess
from pathlib import Path

from ..dtypes import CmdResult

STDIO_TAIL_CHARS = 8000


def tail(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[-n:]


def run_cmd_capture(
    cmd: str,
    cwd: Path,
    *,
    timeout_seconds: int | None = None,
    env: dict[str, str] | None = None,
) -> CmdResult:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            shell=True,
            capture_output=True,
            env=env,
            timeout=timeout_seconds,
        )
        stdout = (p.stdout or b"").decode("utf-8", errors="replace")
        stderr = (p.stderr or b"").decode("utf-8", errors="replace")
        return CmdResult(cmd=cmd, rc=p.returncode, stdout=stdout, stderr=stderr, timed_out=False)
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode("utf-8", errors="replace")
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")
        return CmdResult(cmd=cmd, rc=124, stdout=stdout, stderr=stderr, timed_out=True)
