from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_flow_ext_package_import_is_lazy() -> None:
    sys.modules.pop("agent_market.flow_ext", None)
    sys.modules.pop("agent_market.flow_ext.artifacts", None)
    sys.modules.pop("agent_market.flow_ext.steps", None)

    importlib.import_module("agent_market.flow_ext")

    assert "agent_market.flow_ext.artifacts" not in sys.modules
    assert "agent_market.flow_ext.steps" not in sys.modules


def test_flow_ext_steps_import_defers_legacy_flow_steps() -> None:
    sys.modules.pop("agent_market.flow_ext.steps", None)
    sys.modules.pop("agent_market.flow_steps", None)

    steps = importlib.import_module("agent_market.flow_ext.steps")

    assert "agent_market.flow_steps" not in sys.modules
    assert isinstance(steps.STEP_ORDER, list)
    assert callable(steps.run_command)
    assert "agent_market.flow_steps" in sys.modules


def test_flow_ext_modules_import_and_basic_helpers() -> None:
    from agent_market.flow_ext.step_dispatch import STEP_HANDLERS
    from agent_market.flow_ext.step_spec import STEP_ORDER
    from agent_market.flow_ext import artifacts, steps

    assert isinstance(steps.STEP_ORDER, list)
    assert steps.STEP_ORDER == STEP_ORDER
    assert list(STEP_HANDLERS) == STEP_ORDER
    for key in ["feature", "expression", "ml", "backtest", "tca"]:
        assert key in steps.STEP_ORDER

    merged = steps.merge_artifacts({"a": "1"}, {"b": "2", "c": None, "d": ""})
    assert merged == {"a": "1", "b": "2"}

    ap = artifacts.artifact_paths("deadbeef")
    assert ap.run_id == "deadbeef"
    assert ap.run_dir.name == "deadbeef"
