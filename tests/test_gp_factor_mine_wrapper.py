from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_wrapper() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "gp_factor_mine.py"
    spec = importlib.util.spec_from_file_location("_gp_factor_mine_wrapper", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gp_factor_mine_wrapper_targets_v2_script() -> None:
    wrapper = _load_wrapper()

    assert wrapper._V2_PATH.name == "gp_factor_mine_v2.py"
    assert wrapper._V2_PATH.exists()


def test_gp_factor_mine_wrapper_translates_legacy_window_args() -> None:
    wrapper = _load_wrapper()

    translated = wrapper._translate_legacy_args(
        [
            "--tr3-start",
            "2024-01-01",
            "--tr3-end",
            "2025-07-01",
            "--v3-start=2025-07-01",
            "--v3-end=2025-12-01",
            "--n-gen",
            "1",
        ]
    )

    assert translated == [
        "--data-start",
        "2024-01-01",
        "--data-end=2025-12-01",
        "--n-gen",
        "1",
    ]


def test_wrapper_preserves_explicit_legacy_default_translations():
    wrapper = _load_wrapper()
    assert wrapper._translate_legacy_args([
        "--tr3-start=2023-05-15", "--tr3-end=2025-07-01",
        "--v3-start", "2025-07-01", "--v3-end", "2025-12-01",
    ]) == ["--data-start=2023-05-15", "--data-end", "2025-12-01"]


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--tr3-start", "2022-01-01"),
        ("--tr3-end", "2023-01-01"),
        ("--v3-start", "2024-01-01"),
        ("--v3-end", "2025-01-01"),
    ],
)
@pytest.mark.parametrize("equals", [False, True])
def test_wrapper_rejects_custom_splits_instead_of_silently_changing_experiment(flag, value, equals):
    wrapper = _load_wrapper()
    argv = [f"{flag}={value}"] if equals else [flag, value]
    with pytest.raises(ValueError, match="fixed.*windows"):
        wrapper._translate_legacy_args(argv)


def test_wrapper_custom_split_fails_before_loading_miner(monkeypatch, capsys):
    wrapper = _load_wrapper()
    monkeypatch.setattr(wrapper.sys, "argv", ["gp_factor_mine.py", "--tr3-end", "2023-01-01"])
    monkeypatch.setattr(wrapper, "_load_v2", lambda: pytest.fail("custom split must not start mining"))
    assert wrapper.main() == 2
    assert "refusing to change the experiment silently" in capsys.readouterr().err
