"""Offline regression coverage for quota, pool, and submit persistence."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent_market.wq_brain import quota_monitor as qm
from agent_market.wq_brain.dtypes import AlphaPoolEntry
from agent_market.wq_brain.errors import StateIntegrityError
from agent_market.wq_brain.pool import AlphaPool


def _entry(alpha_id="A", status="ACTIVE"):
    return AlphaPoolEntry(
        alpha_id=alpha_id, expr="rank(close)", settings_dict={},
        sharpe=1.5, fitness=1.2, returns=0.1, turnover=0.2,
        verified_status=status,
    )


@pytest.fixture
def quota_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    path = qm.quota_path("2026-06-01")
    path.parent.mkdir(parents=True)
    return path


@pytest.mark.parametrize("payload", [
    b'{"counts":', b"\xff", b"[]", b"null", b"{}",
    b'{"counts": null}', b'{"counts": {"submit": -1}}',
    b'{"counts": {"submit": "bad"}}',
])
@pytest.mark.parametrize("operation", ["get", "reserve", "record", "release"])
def test_invalid_quota_blocks_all_access_without_overwriting(quota_file, payload, operation):
    quota_file.write_bytes(payload)
    action = {
        "get": lambda: qm.get_usage("2026-06-01"),
        "reserve": lambda: qm.reserve_action("submit", day="2026-06-01", hard_limit=1),
        "record": lambda: qm.record_action("submit", day="2026-06-01"),
        "release": lambda: qm.release_action("submit", day="2026-06-01"),
    }[operation]
    with pytest.raises(StateIntegrityError, match="state.*quota|quota.*state"):
        action()
    assert quota_file.read_bytes() == payload
    assert not quota_file.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("operation", ["get", "reserve", "record", "release"])
def test_unreadable_quota_does_not_reset_usage(quota_file, monkeypatch, operation):
    qm.record_action("submit", day="2026-06-01", n=7)
    original = quota_file.read_bytes()
    read_text = Path.read_text

    def denied(path, *args, **kwargs):
        if path == quota_file:
            raise PermissionError("read denied")
        return read_text(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", denied)
        action = {
            "get": lambda: qm.get_usage("2026-06-01"),
            "reserve": lambda: qm.reserve_action("submit", day="2026-06-01", hard_limit=7),
            "record": lambda: qm.record_action("submit", day="2026-06-01"),
            "release": lambda: qm.release_action("submit", day="2026-06-01"),
        }[operation]
        with pytest.raises(StateIntegrityError, match="state.*quota|quota.*state"):
            action()
    assert quota_file.read_bytes() == original


def test_dangling_quota_symlink_is_not_a_new_store(quota_file, tmp_path):
    quota_file.symlink_to(tmp_path / "missing.json")
    with pytest.raises(StateIntegrityError, match="state"):
        qm.reserve_action("submit", day="2026-06-01")
    assert quota_file.is_symlink()
    assert not (tmp_path / "missing.json").exists()


@pytest.mark.parametrize("payload", [
    b'[{"alpha_id":"A",', b"\xff", b"{}", b"null", b"[{}]", b"[null]",
])
@pytest.mark.parametrize("operation", ["load", "upsert", "replace"])
def test_invalid_pool_preserves_exact_history(tmp_path, payload, operation):
    path = tmp_path / "pool.json"
    pool = AlphaPool(path)
    pool.add(_entry())
    path.write_bytes(payload)
    action = {
        "load": lambda: AlphaPool(path),
        "upsert": lambda: pool.upsert(_entry("B")),
        "replace": lambda: pool.replace_all([]),
    }[operation]
    with pytest.raises(StateIntegrityError, match="state.*pool|pool.*state"):
        action()
    assert path.read_bytes() == payload
    assert not path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("operation", ["load", "upsert", "replace"])
def test_unreadable_pool_preserves_history(tmp_path, monkeypatch, operation):
    path = tmp_path / "pool.json"
    pool = AlphaPool(path)
    pool.add(_entry())
    original = path.read_bytes()
    read_text = Path.read_text

    def denied(p, *args, **kwargs):
        if p == path:
            raise PermissionError("read denied")
        return read_text(p, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", denied)
        action = {
            "load": lambda: AlphaPool(path),
            "upsert": lambda: pool.upsert(_entry("B")),
            "replace": lambda: pool.replace_all([]),
        }[operation]
        with pytest.raises(StateIntegrityError, match="state"):
            action()
    assert path.read_bytes() == original
    assert [entry.alpha_id for entry in pool] == ["A"]


def test_dangling_pool_symlink_is_not_a_new_store(tmp_path):
    path = tmp_path / "pool.json"
    path.symlink_to(tmp_path / "missing.json")
    with pytest.raises(StateIntegrityError, match="state"):
        AlphaPool(path)
    assert path.is_symlink()


def test_pool_lock_failure_must_not_write_unlocked(tmp_path, monkeypatch):
    import fcntl

    path = tmp_path / "pool.json"
    pool = AlphaPool(path)
    pool.add(_entry())
    original = path.read_bytes()
    monkeypatch.setattr(fcntl, "flock", Mock(side_effect=OSError("lock failed")))
    with pytest.raises(OSError, match="lock failed"):
        pool.upsert(_entry("B"))
    assert path.read_bytes() == original
    assert [entry.alpha_id for entry in pool] == ["A"]


@pytest.fixture
def cli(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "wq_brain.py"
    spec = importlib.util.spec_from_file_location("_wq_integrity_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    monkeypatch.setattr(module, "_ensure_dotenv", lambda: None)
    return module


@pytest.mark.parametrize(("status", "failure"), [
    (status, failure)
    for status in ("ACTIVE", "REJECTED", "UNSUBMITTED")
    for failure in ("fetch", "save", "missing_id", "corrupt_pool")
] + [("ACTIVE", "missing_sharpe")])
def test_submit_recording_failure_reports_partial_without_resubmit(
    cli, tmp_path, monkeypatch, capsys, status, failure,
):
    from agent_market.wq_brain import client
    from agent_market.wq_brain.paths import alpha_pool_path

    response = {"verified_status": status, "rejection_reasons": [{"name": "fitness"}]}
    session = Mock()
    session.submit_alpha.return_value = response
    metrics = SimpleNamespace(
        alpha_id="A", sharpe=1.5, fitness=1.2, returns=0.1, turnover=0.2,
    )
    session.fetch_alpha_metrics.return_value = metrics
    if failure == "fetch":
        session.fetch_alpha_metrics.side_effect = OSError("metrics unavailable")
    elif failure == "save":
        monkeypatch.setattr(AlphaPool, "upsert", Mock(side_effect=OSError("disk full")))
    elif failure == "missing_id":
        metrics.alpha_id = None
    elif failure == "corrupt_pool":
        path = alpha_pool_path("safety")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'[{"alpha_id":"existing",')
    else:
        metrics.sharpe = None
    monkeypatch.setattr(client, "session_from_env", lambda: session)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_submit(argparse.Namespace(
            alpha_id="A", tag="safety", expr="rank(close)", no_pre_check=True,
            verify_after_sec=0,
        ))
    result = json.loads(capsys.readouterr().out)
    assert exc.value.code == 3
    assert result["ok"] is False
    assert result["partial_failure"] is True
    assert result["alpha_id"] == "A"
    assert result["verified_status"] == status
    assert result["wq_response"] == response
    assert result["recorded_to_pool"] is False
    assert result["pool_recording_error"]
    assert "do not resubmit" in result["hint"].lower()
    session.submit_alpha.assert_called_once_with("A", verify_after_sec=0)
    assert qm.get_usage().counts["submit"] == 1
    if failure == "corrupt_pool":
        assert path.read_bytes() == b'[{"alpha_id":"existing",'


@pytest.mark.parametrize(("status", "exit_code"), [
    ("ACTIVE", 0), ("REJECTED", 2), ("UNSUBMITTED", 2),
])
def test_submit_persists_healthy_outcome_and_existing_history(
    cli, monkeypatch, capsys, status, exit_code,
):
    from agent_market.wq_brain import client
    from agent_market.wq_brain.paths import alpha_pool_path

    path = alpha_pool_path("safety")
    pool = AlphaPool(path)
    pool.add(_entry("existing"))
    pool.add(_entry("A", status="UNSUBMITTED"))
    session = Mock()
    session.submit_alpha.return_value = {
        "verified_status": status, "rejection_reasons": [],
    }
    session.fetch_alpha_metrics.return_value = SimpleNamespace(
        alpha_id="A", sharpe=1.5, fitness=1.2, returns=0.1, turnover=0.2,
    )
    monkeypatch.setattr(client, "session_from_env", lambda: session)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_submit(argparse.Namespace(
            alpha_id="A", tag="safety", expr="rank(close)", no_pre_check=True,
            verify_after_sec=0,
        ))
    result = json.loads(capsys.readouterr().out)
    assert exc.value.code == exit_code
    assert result["ok"] is (status == "ACTIVE")
    assert {entry.alpha_id: entry.verified_status for entry in AlphaPool(path)} == {
        "existing": "ACTIVE", "A": status,
    }
    session.submit_alpha.assert_called_once()


def test_cli_corrupt_quota_blocks_remote_submit(cli, monkeypatch, capsys):
    from agent_market.wq_brain import client

    session = Mock()
    monkeypatch.setattr(client, "session_from_env", lambda: session)
    path = qm.quota_path()
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"counts":')
    monkeypatch.setattr("sys.argv", [
        "wq_brain.py", "submit", "A", "--no-pre-check",
    ])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    result = json.loads(capsys.readouterr().out)
    assert exc.value.code != 0
    assert result["ok"] is False
    assert result["error_type"] == "state_integrity"
    session.submit_alpha.assert_not_called()
    assert path.read_bytes() == b'{"counts":'
