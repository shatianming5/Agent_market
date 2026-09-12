"""Exercise destructive workspace operations only beneath pytest temp roots."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


def _load(relative_path, name):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cleaner(tmp_path, monkeypatch):
    module = _load("scripts/clean_workspace.py", "_clean_safety")
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    monkeypatch.setattr(module, "__file__", str(root / "scripts" / "clean_workspace.py"))
    return module, root


@pytest.mark.parametrize("target", [
    ".", "..", "safe/..", "safe/../victim", "../outside", "absolute",
    "C:\\outside", "\\\\server\\share", "link", "link/child",
])
@pytest.mark.parametrize("dry_run", [False, True])
def test_cleaner_validates_all_targets_before_removing_anything(
    cleaner, tmp_path, monkeypatch, target, dry_run,
):
    module, root = cleaner
    safe = root / "safe"
    safe.mkdir()
    (safe / "keep").write_text("evidence")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("outside")
    (root / "link").symlink_to(outside, target_is_directory=True)
    if target == "absolute":
        target = str(outside)
    argv = ["clean_workspace.py", "safe", target]
    if dry_run:
        argv.append("--dry-run")
    monkeypatch.setattr("sys.argv", argv)
    remove = Mock()
    monkeypatch.setattr(module, "rm_rf", remove)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code != 0
    remove.assert_not_called()
    assert (safe / "keep").read_text() == "evidence"
    assert (outside / "keep").read_text() == "outside"


def test_cleaner_reports_actual_deletion_failure(cleaner, monkeypatch, capsys):
    module, root = cleaner
    (root / "safe").mkdir()

    def fail(path, **kwargs):
        if not kwargs.get("ignore_errors", False):
            raise PermissionError("delete denied")

    monkeypatch.setattr(module.shutil, "rmtree", fail)
    monkeypatch.setattr("sys.argv", ["clean_workspace.py", "safe"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "delete denied" in captured.err
    assert "done." not in captured.out
    assert (root / "safe").is_dir()


def test_cleaner_removes_only_requested_descendants(cleaner, monkeypatch):
    module, root = cleaner
    (root / "artifacts").mkdir()
    (root / "artifacts" / "old").write_text("old")
    (root / "keep").write_text("keep")
    monkeypatch.setattr("sys.argv", ["clean_workspace.py", "artifacts", "--keep-dirs"])
    module.main()
    assert list((root / "artifacts").iterdir()) == []
    assert (root / "keep").read_text() == "keep"


def test_cleaner_unlinks_internal_symlink_not_its_target(cleaner, monkeypatch):
    module, root = cleaner
    (root / "keep").mkdir()
    (root / "keep" / "evidence").write_text("keep")
    (root / "link").symlink_to(root / "keep", target_is_directory=True)
    monkeypatch.setattr("sys.argv", ["clean_workspace.py", "link"])
    module.main()
    assert not (root / "link").is_symlink()
    assert (root / "keep" / "evidence").read_text() == "keep"


@pytest.fixture
def creator(tmp_path, monkeypatch):
    module = _load("create_workspace.py", "_create_safety")
    root = tmp_path / "repo"
    (root / "workspace" / "strategies").mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "_download_data", Mock(side_effect=AssertionError("network forbidden")))
    return module, root


@pytest.mark.parametrize("name", [
    ".", "..", "../outside", "nested/name", "nested\\name",
    "absolute", "C:\\outside", "\\\\server\\share", "link",
])
def test_workspace_name_cannot_escape_or_target_root(creator, tmp_path, name):
    module, root = creator
    outside = tmp_path / "outside"
    (root / "link").symlink_to(outside, target_is_directory=True)
    if name == "absolute":
        name = str(outside)
    with pytest.raises((ValueError, SystemExit)):
        module.create_workspace(name)
    assert sorted(p.name for p in root.iterdir()) == ["link", "workspace"]
    assert not outside.exists()


@pytest.mark.parametrize("name", ["my_research", "ws_custom", "trial-1"])
def test_workspace_custom_local_names_are_preserved(creator, name):
    module, root = creator
    ws = module.create_workspace(name)
    assert ws == root / name
    assert (ws / "meta.json").is_file()


def test_workspace_automatic_name_is_unchanged(creator):
    module, root = creator
    (root / "ws_003").mkdir()
    assert module.create_workspace() == root / "ws_004"
