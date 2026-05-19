from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


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
            "2024-07-01",
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
