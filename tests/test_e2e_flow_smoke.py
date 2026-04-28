from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_MISSING_DEPS: list[str] = []
for _mod in ("lightgbm",):
    try:
        __import__(_mod)
    except ModuleNotFoundError:
        _MISSING_DEPS.append(_mod)


@pytest.mark.skipif(bool(_MISSING_DEPS), reason=f"missing optional deps: {_MISSING_DEPS}")
def test_e2e_flow_produces_required_artifacts():
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "agent_flow_kucoin_cpu_nollm_smoke.json"
    script = root / "scripts" / "e2e_smoke_flow.py"

    cmd = [sys.executable, str(script), "--config", str(cfg)]
    subprocess.run(cmd, cwd=str(root), check=True)  # noqa: S603,S607

    # UI "golden path" defaults: ensure /web points to the golden config.
    import server.main as srv  # type: ignore

    client = TestClient(srv.app)
    resp = client.get("/web/index.html")
    assert resp.status_code == 200
    html = resp.text
    assert "configs/agent_flow_kucoin_cpu_nollm.json" in html
    assert "cdn.jsdelivr.net" not in html
    assert "自动布局" in html
    assert 'id="btnTrainBt"' in html
    assert 'id="flowArtifacts"' in html

    # Artifact-check endpoint used by the web UI.
    meta = client.get("/flow/run-meta/latest").json()
    assert meta.get("status") != "error", meta
    assert isinstance(meta.get("run_id"), str) and meta.get("run_id")
    checks = meta.get("checks") or {}
    assert checks.get("feature_output", {}).get("exists") is True
    assert checks.get("expression_output", {}).get("exists") is True
    assert checks.get("feedback_summary", {}).get("exists") is True

    runs = client.get("/flow/runs/list?limit=10").json()
    assert runs.get("items") and isinstance(runs["items"], list)
    assert any(it.get("run_id") == meta["run_id"] for it in runs["items"])


def test_expression_strategy_prefers_lightgbm_summary(monkeypatch, tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    artifacts_root = tmp_path / "artifacts"
    user_root = tmp_path / "user_data"
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(artifacts_root))
    monkeypatch.setenv("AGENT_MARKET_USER_DATA_ROOT", str(user_root))

    strategy_src = root / "user_data" / "strategies" / "ExpressionLongStrategy.py"
    strategy_dst = user_root / "strategies" / "ExpressionLongStrategy.py"
    strategy_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(strategy_src, strategy_dst)

    rl_dir = artifacts_root / "models" / "rl_manual_sanity"
    rl_dir.mkdir(parents=True, exist_ok=True)
    (rl_dir / "training_summary.json").write_text(
        json.dumps({"model": "ppo", "model_path": str(rl_dir / "ppo_trading_env.zip")}),
        encoding="utf-8",
    )

    lgb_dir = artifacts_root / "models" / "lightgbm_real"
    lgb_dir.mkdir(parents=True, exist_ok=True)
    (lgb_dir / "training_summary.json").write_text(
        json.dumps({"model": "lightgbm", "model_path": str(lgb_dir / "lightgbm_model.txt")}),
        encoding="utf-8",
    )

    spec = importlib.util.spec_from_file_location("test_expression_strategy", strategy_dst)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    fake_freqtrade = types.ModuleType("freqtrade")
    fake_strategy = types.ModuleType("freqtrade.strategy")

    class _DummyIStrategy:  # noqa: D401
        """Minimal stub for unit-loading the strategy module."""

    fake_strategy.IStrategy = _DummyIStrategy
    monkeypatch.setitem(sys.modules, "freqtrade", fake_freqtrade)
    monkeypatch.setitem(sys.modules, "freqtrade.strategy", fake_strategy)
    spec.loader.exec_module(module)

    strategy = object.__new__(module.ExpressionLongStrategy)
    resolved = strategy._resolve_model_dir()
    assert resolved == lgb_dir.resolve()
