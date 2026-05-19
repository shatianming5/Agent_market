"""Tests for agent_flow module: config loading, AgentFlow, helper functions."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# AgentFlowConfig
# ---------------------------------------------------------------------------


def test_agent_flow_config_from_dict():
    from agent_market.agent_flow import AgentFlowConfig

    cfg = AgentFlowConfig.from_dict({
        "feature": {"script": "scripts/feat.py", "args": ["--config", "c.json"]},
        "backtest": {"config": "config.json"},
    })
    assert cfg.feature is not None
    assert cfg.backtest is not None
    assert cfg.capture is None
    assert cfg.ml_training is None


def test_agent_flow_config_from_dict_unknown_keys(caplog):
    from agent_market.agent_flow import AgentFlowConfig

    import logging
    with caplog.at_level(logging.WARNING):
        cfg = AgentFlowConfig.from_dict({"unknown_key": 123, "feature": {}})
    assert cfg.feature == {}
    assert "unknown_key" in caplog.text


def test_agent_flow_config_empty():
    from agent_market.agent_flow import AgentFlowConfig

    cfg = AgentFlowConfig.from_dict({})
    assert cfg.feature is None
    assert cfg.backtest is None
    assert cfg.tca is None


def test_agent_flow_config_all_keys():
    from agent_market.agent_flow import AgentFlowConfig

    data = {
        "capture": {"exchange": "binance"},
        "lob_rebuild": {"depth": 20},
        "feature": {"script": "feat.py"},
        "micro_feature": {"config": "cfg.json"},
        "portfolio": {"pairs": ["BTC/USDT"]},
        "expression": {"script": "expr.py"},
        "factor_compile": {"spec": {}},
        "factor_eval": {"expression": "x+1"},
        "ml_training": {"config": {}},
        "rl_training": {"config": {}},
        "backtest": {"config": "bt.json"},
        "tca": {"html": True},
        "report": {},
        "strategy_miner": {"model": "gpt-4o"},
        "experiment": {"hypothesis": "demo"},
    }
    cfg = AgentFlowConfig.from_dict(data)
    assert cfg.capture == {"exchange": "binance"}
    assert cfg.strategy_miner == {"model": "gpt-4o"}
    assert cfg.experiment == {"hypothesis": "demo"}


# ---------------------------------------------------------------------------
# load_agent_flow_config
# ---------------------------------------------------------------------------


def test_load_agent_flow_config():
    from agent_market.agent_flow import load_agent_flow_config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"feature": {"script": "feat.py"}, "backtest": {"config": "c.json"}}, f)
        f.flush()
        cfg = load_agent_flow_config(Path(f.name))
    assert cfg.feature is not None
    assert cfg.backtest is not None
    Path(f.name).unlink()


def test_load_agent_flow_config_not_found():
    from agent_market.agent_flow import load_agent_flow_config

    with pytest.raises(FileNotFoundError):
        load_agent_flow_config(Path("/nonexistent/config.json"))


def test_agent_flow_script_help_uses_package_cli():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "agent_flow.py"), "--help"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Agent Market end-to-end orchestrator" in proc.stdout
    assert "--log-dir" in proc.stdout


def test_agent_flow_module_help_uses_package_cli():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "agent_market.agent_flow", "--help"],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Agent Market end-to-end orchestrator" in proc.stdout
    assert "--log-dir" in proc.stdout


def test_agent_flow_import_does_not_load_runtime_preflight():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import agent_market.agent_flow; "
                "assert 'agent_market.runtime_preflight' not in sys.modules; "
                "assert 'agent_market.flow_steps' not in sys.modules; "
                "assert 'agent_market.flow_ext.steps' not in sys.modules; "
                "assert 'agent_market.flow_ext.step_dispatch' not in sys.modules; "
                "assert 'agent_market.run_artifacts' not in sys.modules; "
                "print('ok')"
            ),
        ],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert proc.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# helper: _extract_flag_value
# ---------------------------------------------------------------------------


def test_extract_flag_value():
    from agent_market.agent_flow import _extract_flag_value

    assert _extract_flag_value(["--config", "c.json", "--output", "out.json"], "--output") == "out.json"
    assert _extract_flag_value(["--config", "c.json"], "--output") is None
    assert _extract_flag_value(None, "--output") is None
    assert _extract_flag_value("not a list", "--output") is None


def test_extract_flag_value_last_wins():
    from agent_market.agent_flow import _extract_flag_value

    assert _extract_flag_value(["--out", "a.json", "--out", "b.json"], "--out") == "b.json"


# ---------------------------------------------------------------------------
# helper: _write_json_atomic
# ---------------------------------------------------------------------------


def test_write_json_atomic():
    from agent_market.agent_flow import _write_json_atomic

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sub" / "meta.json"
        _write_json_atomic(path, {"run_id": "abc123", "status": "ok"})
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["run_id"] == "abc123"
        # no .tmp file remaining
        assert not list(Path(td).rglob("*.tmp"))


# ---------------------------------------------------------------------------
# helper: _config_snapshot_info
# ---------------------------------------------------------------------------


def test_config_snapshot_info_from_file():
    from agent_market.agent_flow import AgentFlowConfig, _config_snapshot_info

    cfg = AgentFlowConfig()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"feature": {}}, f)
        f.flush()
        info = _config_snapshot_info(cfg, Path(f.name))
    assert info["source"] == "file"
    assert info["sha256"] is not None
    Path(f.name).unlink()


def test_config_snapshot_info_from_object():
    from agent_market.agent_flow import AgentFlowConfig, _config_snapshot_info

    cfg = AgentFlowConfig(feature={"script": "test.py"})
    info = _config_snapshot_info(cfg, None)
    assert info["source"] == "object"
    assert info["sha256"] is not None


def test_config_snapshot_info_missing_file():
    from agent_market.agent_flow import AgentFlowConfig, _config_snapshot_info

    cfg = AgentFlowConfig()
    info = _config_snapshot_info(cfg, Path("/nonexistent/config.json"))
    # Falls back to object snapshot
    assert info["source"] == "object"


# ---------------------------------------------------------------------------
# AgentFlow.STEP_ORDER
# ---------------------------------------------------------------------------


def test_step_order():
    from agent_market.agent_flow import AgentFlow
    from agent_market.flow_ext.step_spec import STEP_ORDER

    assert AgentFlow.STEP_ORDER == STEP_ORDER
    assert "feature" in AgentFlow.STEP_ORDER
    assert "backtest" in AgentFlow.STEP_ORDER
    assert "tca" in AgentFlow.STEP_ORDER
    assert AgentFlow.STEP_ORDER.index("feature") < AgentFlow.STEP_ORDER.index("backtest")
    assert AgentFlow.STEP_ORDER.index("capture") < AgentFlow.STEP_ORDER.index("lob_rebuild")
    assert AgentFlow.STEP_ORDER.index("strategy_miner") < AgentFlow.STEP_ORDER.index("report")


# ---------------------------------------------------------------------------
# AgentFlow.run step filtering
# ---------------------------------------------------------------------------


def test_agent_flow_skips_unconfigured_steps():
    """If a step is requested but not configured, it should be skipped with a warning."""
    from agent_market.agent_flow import AgentFlow, AgentFlowConfig

    cfg = AgentFlowConfig()  # all None
    flow = AgentFlow(cfg)

    with tempfile.TemporaryDirectory() as td:
        with patch("agent_market.agent_flow.paths") as mock_paths:
            mock_paths.REPO_ROOT = Path(td)
            mock_paths.run_meta_latest_path.return_value = Path(td) / "meta_latest.json"
            mock_paths.run_meta_path.return_value = Path(td) / "runs" / "test" / "meta.json"
            mock_paths.models_root.return_value = Path(td) / "models"
            mock_paths.resolve_repo_path.side_effect = lambda x: Path(td) / str(x)
            mock_paths.relpath_under_repo.side_effect = lambda x: str(x)
            mock_paths.relpath_for_meta.side_effect = lambda x: str(x)
            mock_paths.default_feedback_path.return_value = "feedback.json"
            # flow.run should complete without error (no steps configured)
            flow.run(steps=["feature"])


def test_agent_flow_runs_empty_dict_step_config():
    from agent_market.agent_flow import AgentFlow, AgentFlowConfig

    cfg = AgentFlowConfig(report={})
    flow = AgentFlow(cfg)

    with tempfile.TemporaryDirectory() as td:
        with patch("agent_market.agent_flow.paths") as mock_paths:
            mock_paths.REPO_ROOT = Path(td)
            mock_paths.run_meta_latest_path.return_value = Path(td) / "meta_latest.json"
            mock_paths.run_meta_path.return_value = Path(td) / "runs" / "test" / "meta.json"
            mock_paths.models_root.return_value = Path(td) / "models"
            mock_paths.resolve_repo_path.side_effect = lambda x: Path(td) / str(x)
            mock_paths.relpath_under_repo.side_effect = lambda x: str(x)
            mock_paths.relpath_for_meta.side_effect = lambda x: str(x)
            mock_paths.default_feedback_path.return_value = "feedback.json"
            with patch("agent_market.agent_flow.STEP_HANDLERS", {"report": lambda cfg, arts, ctx: None}):
                run_id = flow.run(steps=["report"])
                assert isinstance(run_id, str) and run_id


# ---------------------------------------------------------------------------
# flow_steps unit tests
# ---------------------------------------------------------------------------


def test_resolve_executable_not_found():
    from agent_market.flow_steps import _resolve_executable

    assert _resolve_executable("") is None
    assert _resolve_executable("this_executable_definitely_does_not_exist_xyz") is None


def test_inject_userdir():
    from agent_market.flow_steps import _inject_userdir

    cmd = ["freqtrade", "backtesting", "--config", "c.json"]
    result = _inject_userdir(cmd, "/path/to/user_data")
    assert "--userdir" in result
    assert "/path/to/user_data" in result

    # Should not duplicate
    cmd2 = ["freqtrade", "backtesting", "--userdir", "existing"]
    result2 = _inject_userdir(cmd2, "/path/to/user_data")
    assert result2 == cmd2  # unchanged


def test_inject_userdir_no_backtesting():
    from agent_market.flow_steps import _inject_userdir

    cmd = ["freqtrade", "download-data", "--config", "c.json"]
    result = _inject_userdir(cmd, "/path/to/user_data")
    assert result == cmd  # no backtesting subcommand, unchanged


def test_extract_config_path_from_command():
    from agent_market.flow_steps import _extract_config_path_from_command

    assert _extract_config_path_from_command(["ft", "--config", "c.json"]) == "c.json"
    assert _extract_config_path_from_command(["ft", "-c", "x.json"]) == "x.json"
    assert _extract_config_path_from_command(["ft", "--strategy", "S"]) is None
    assert _extract_config_path_from_command(["ft", "--config"]) is None  # no value after flag
