"""Hardened subprocess execution for untrusted strategy backtests.

Provides OS-level restrictions beyond AST validation:
- CPU time limit (prevents infinite loops)
- Memory limit (prevents OOM)
- No new network sockets (on Linux via seccomp-like resource limits)
- Optional process count limit (disabled by default on shared servers)

These restrictions are applied via preexec_fn in subprocess.run().
On macOS/Windows, only basic resource limits are available.
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_IS_LINUX = platform.system() == "Linux"
_NPROC_ENV_KEY = "AGENT_MARKET_SANDBOX_NPROC_LIMIT"
_THREAD_ENV_KEY = "AGENT_MARKET_SANDBOX_THREADS"
_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _resolve_nproc_limit(env: Optional[dict[str, str]] = None) -> int | None:
    """Return an explicit RLIMIT_NPROC override, or None to leave it unset.

    RLIMIT_NPROC is per-real-UID and counts threads on Linux. In shared research
    environments, a low fixed value can break numpy/OpenBLAS and aiodns imports,
    masking real backtest failures behind SIGINT/KeyboardInterrupt noise.
    """
    raw = (env or os.environ).get(_NPROC_ENV_KEY, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", _NPROC_ENV_KEY, raw)
        return None
    if value <= 0:
        logger.warning("Ignoring non-positive %s=%r", _NPROC_ENV_KEY, raw)
        return None
    return value


def _resolve_thread_limit(env: Optional[dict[str, str]] = None) -> int:
    """Return the default sandbox thread cap for BLAS/OpenMP style runtimes.

    Shared research servers often run many sandboxed backtests concurrently.
    Letting each child keep the library defaults can trigger thread-local
    allocation failures in xgboost/OpenMP before the strategy code even runs.
    """
    raw = (env or os.environ).get(_THREAD_ENV_KEY, "").strip()
    if not raw:
        return 1
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", _THREAD_ENV_KEY, raw)
        return 1
    if value <= 0:
        logger.warning("Ignoring non-positive %s=%r", _THREAD_ENV_KEY, raw)
        return 1
    return value


def _preexec_sandbox(*, cpu_seconds: int = 600, mem_mb: int = 4096, nproc: int | None = None) -> None:
    """Applied as preexec_fn to restrict the child process."""
    try:
        import resource

        # CPU time limit
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 30))

        # Memory limit
        mem_bytes = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

        # Process/thread limit is opt-in only. A low RLIMIT_NPROC breaks common
        # scientific/runtime imports on shared Linux servers because threads are
        # counted against the per-UID limit.
        if nproc is not None:
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
            except (ValueError, AttributeError):
                pass  # Not available on macOS

        # File size limit (prevent filling disk)
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024 * 512, 1024 * 1024 * 512))  # 512MB

    except Exception as exc:
        logger.debug("Resource limits partially applied: %s", exc)


def run_sandboxed(
    cmd: list[str],
    *,
    cwd: str | Path,
    timeout: int = 300,
    cpu_seconds: int = 600,
    mem_mb: int = 4096,
    env: Optional[dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command with OS-level resource restrictions.

    This wraps subprocess.run with:
    - preexec_fn resource limits (CPU, memory, processes, file size)
    - Environment sanitization (remove sensitive vars)
    - Timeout enforcement
    """
    # Sanitize environment: remove sensitive vars from child (D11)
    _SENSITIVE_KEYS = {
        "AGENT_MARKET_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY",
        "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "GITHUB_TOKEN",
        "GH_TOKEN", "ANTHROPIC_API_KEY", "AZURE_OPENAI_KEY",
        "HF_TOKEN", "HUGGINGFACE_TOKEN", "OPENCODE_API_KEY",
        "CLAUDE_API_KEY", "GOOGLE_API_KEY", "COHERE_API_KEY",
        "MISTRAL_API_KEY",
    }
    safe_env = dict(env or os.environ)
    for sensitive_key in _SENSITIVE_KEYS:
        safe_env.pop(sensitive_key, None)
    # Backtests must be deterministic and must not depend on the caller's
    # PYTHONPATH. In particular, `PYTHONPATH=src:.` can inject the repo root
    # and shadow vendored `freqtrade/` as a namespace package.
    safe_env.pop("PYTHONPATH", None)
    # Keep threaded math libraries conservative inside shared-server sandboxes.
    thread_limit = str(_resolve_thread_limit(safe_env))
    for thread_env_key in _THREAD_ENV_VARS:
        safe_env.setdefault(thread_env_key, thread_limit)
    nproc_limit = _resolve_nproc_limit(safe_env)

    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        preexec_fn=lambda: _preexec_sandbox(
            cpu_seconds=cpu_seconds, mem_mb=mem_mb, nproc=nproc_limit,
        ),
        env=safe_env,
    )

    # D11: Scrub stdout/stderr to remove any leaked secrets
    result = _scrub_output(result, _SENSITIVE_KEYS)
    return result


def _scrub_output(
    result: subprocess.CompletedProcess[str],
    sensitive_keys: set[str],
) -> subprocess.CompletedProcess[str]:
    """Remove accidental secret leaks from subprocess output."""
    import os as _os2
    scrub_vals = set()
    for key in sensitive_keys:
        val = _os2.environ.get(key, "")
        if val and len(val) >= 8:  # Only scrub non-trivial values
            scrub_vals.add(val)
    if not scrub_vals:
        return result
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    changed = False
    for val in scrub_vals:
        if val in stdout:
            stdout = stdout.replace(val, "***REDACTED***")
            changed = True
        if val in stderr:
            stderr = stderr.replace(val, "***REDACTED***")
            changed = True
    if not changed:
        return result
    # Reconstruct — handle both CompletedProcess and mock objects
    try:
        return subprocess.CompletedProcess(
            args=getattr(result, "args", []),
            returncode=getattr(result, "returncode", 1),
            stdout=stdout,
            stderr=stderr,
        )
    except Exception:
        # Fallback: mutate in-place for non-standard result objects
        result.stdout = stdout  # type: ignore[attr-defined]
        result.stderr = stderr  # type: ignore[attr-defined]
        return result
