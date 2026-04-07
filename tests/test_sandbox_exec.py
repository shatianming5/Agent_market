from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_resolve_nproc_limit_disabled_by_default(monkeypatch):
    from agent_market.strategy_miner import _sandbox_exec

    monkeypatch.delenv("AGENT_MARKET_SANDBOX_NPROC_LIMIT", raising=False)
    assert _sandbox_exec._resolve_nproc_limit() is None


def test_resolve_nproc_limit_from_env(monkeypatch):
    from agent_market.strategy_miner import _sandbox_exec

    monkeypatch.setenv("AGENT_MARKET_SANDBOX_NPROC_LIMIT", "512")
    assert _sandbox_exec._resolve_nproc_limit() == 512


def test_resolve_thread_limit_defaults_to_one(monkeypatch):
    from agent_market.strategy_miner import _sandbox_exec

    monkeypatch.delenv("AGENT_MARKET_SANDBOX_THREADS", raising=False)
    assert _sandbox_exec._resolve_thread_limit() == 1


def test_resolve_thread_limit_from_env(monkeypatch):
    from agent_market.strategy_miner import _sandbox_exec

    monkeypatch.setenv("AGENT_MARKET_SANDBOX_THREADS", "3")
    assert _sandbox_exec._resolve_thread_limit() == 3


def test_preexec_sandbox_skips_nproc_by_default(monkeypatch):
    from agent_market.strategy_miner import _sandbox_exec

    calls: list[tuple[int, tuple[int, int]]] = []
    fake_resource = SimpleNamespace(
        RLIMIT_CPU=1,
        RLIMIT_AS=2,
        RLIMIT_NPROC=3,
        RLIMIT_FSIZE=4,
        setrlimit=lambda which, value: calls.append((which, value)),
    )
    monkeypatch.setitem(sys.modules, "resource", fake_resource)

    _sandbox_exec._preexec_sandbox(cpu_seconds=60, mem_mb=1024, nproc=None)

    assert (fake_resource.RLIMIT_CPU, (60, 90)) in calls
    assert (fake_resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024)) in calls
    assert (fake_resource.RLIMIT_FSIZE, (1024 * 1024 * 512, 1024 * 1024 * 512)) in calls
    assert all(which != fake_resource.RLIMIT_NPROC for which, _ in calls)


def test_preexec_sandbox_applies_nproc_when_explicit(monkeypatch):
    from agent_market.strategy_miner import _sandbox_exec

    calls: list[tuple[int, tuple[int, int]]] = []
    fake_resource = SimpleNamespace(
        RLIMIT_CPU=1,
        RLIMIT_AS=2,
        RLIMIT_NPROC=3,
        RLIMIT_FSIZE=4,
        setrlimit=lambda which, value: calls.append((which, value)),
    )
    monkeypatch.setitem(sys.modules, "resource", fake_resource)

    _sandbox_exec._preexec_sandbox(cpu_seconds=60, mem_mb=1024, nproc=512)

    assert (fake_resource.RLIMIT_NPROC, (512, 512)) in calls


def test_run_sandboxed_sets_conservative_thread_env(monkeypatch):
    from agent_market.strategy_miner import _sandbox_exec

    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured["env"] = kwargs["env"]
        return SimpleNamespace(args=args[0], returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(_sandbox_exec.subprocess, "run", fake_run)
    monkeypatch.delenv("AGENT_MARKET_SANDBOX_THREADS", raising=False)

    result = _sandbox_exec.run_sandboxed(["echo", "ok"], cwd=".")

    assert result.returncode == 0
    env = captured["env"]
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"


def test_run_sandboxed_strips_pythonpath(monkeypatch):
    from agent_market.strategy_miner import _sandbox_exec

    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured["env"] = kwargs["env"]
        return SimpleNamespace(args=args[0], returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(_sandbox_exec.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "src:.")

    _sandbox_exec.run_sandboxed(["echo", "ok"], cwd=".")

    assert "PYTHONPATH" not in captured["env"]


def test_freqtrade_cli_bootstrap_survives_pythonpath_shadowing():
    try:
        import importlib.util as _importlib_util
        if _importlib_util.find_spec("freqtrade") is None:
            pytest.skip("freqtrade is not installed")
    except Exception:
        pytest.skip("freqtrade availability unknown")

    repo_root = Path(__file__).resolve().parents[1]
    wrapper = repo_root / "scripts" / "freqtrade_cli.py"
    assert wrapper.exists()

    env = dict(os.environ)
    env["PYTHONPATH"] = "src:."
    proc = subprocess.run(
        [sys.executable, str(wrapper), "--version"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (proc.stderr or proc.stdout or "")
    assert "Freqtrade Version" in (proc.stdout or "")
