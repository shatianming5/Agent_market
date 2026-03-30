from __future__ import annotations

from pathlib import Path


def test_flow_ext_modules_import_and_basic_helpers() -> None:
    from agent_market.flow_ext import artifacts, steps

    assert isinstance(steps.STEP_ORDER, list)
    for key in ["feature", "expression", "ml", "backtest", "tca"]:
        assert key in steps.STEP_ORDER

    merged = steps.merge_artifacts({"a": "1"}, {"b": "2", "c": None, "d": ""})
    assert merged == {"a": "1", "b": "2"}

    ap = artifacts.artifact_paths("deadbeef")
    assert ap.run_id == "deadbeef"
    assert ap.run_dir.name == "deadbeef"

