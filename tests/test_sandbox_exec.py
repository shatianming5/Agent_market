from __future__ import annotations

import sys
from types import SimpleNamespace


def test_resolve_nproc_limit_disabled_by_default(monkeypatch):
    from agent_market.strategy_miner import _sandbox_exec

    monkeypatch.delenv("AGENT_MARKET_SANDBOX_NPROC_LIMIT", raising=False)
    assert _sandbox_exec._resolve_nproc_limit() is None


def test_resolve_nproc_limit_from_env(monkeypatch):
    from agent_market.strategy_miner import _sandbox_exec

    monkeypatch.setenv("AGENT_MARKET_SANDBOX_NPROC_LIMIT", "512")
    assert _sandbox_exec._resolve_nproc_limit() == 512


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
