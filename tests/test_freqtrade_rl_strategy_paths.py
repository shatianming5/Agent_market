from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _import_strategy_module():
    freqtrade_mod = types.ModuleType("freqtrade")
    strategy_mod = types.ModuleType("freqtrade.strategy")

    class _IStrategy:
        def __init__(self, config=None):  # noqa: ANN001
            self.config = config or {}

    strategy_mod.IStrategy = _IStrategy
    sys.modules.setdefault("freqtrade", freqtrade_mod)
    sys.modules["freqtrade.strategy"] = strategy_mod
    sys.modules.pop("workspace.strategies.type_F_ml.freqtrade_rl_strategy", None)
    return importlib.import_module("workspace.strategies.type_F_ml.freqtrade_rl_strategy")


def test_rl_resolve_summary_artifact_falls_back_to_model_dir(tmp_path, monkeypatch) -> None:
    module = _import_strategy_module()
    model_dir = tmp_path / "artifacts" / "models" / "rl_manual_sanity"
    model_dir.mkdir(parents=True, exist_ok=True)
    local_model = model_dir / "ppo_trading_env.zip"
    local_model.write_bytes(b"zip")
    monkeypatch.setattr(module, "_ROOT", tmp_path)

    resolved = module._resolve_summary_artifact(
        "/Users/shatianming/Downloads/Agent_market/artifacts/models/rl_manual_sanity/ppo_trading_env.zip",
        model_dir=model_dir,
        fallback_names=["ppo_trading_env.zip"],
    )

    assert resolved == local_model


def test_rl_load_prefers_rl_dir_and_ignores_sidecar_zip(tmp_path, monkeypatch) -> None:
    module = _import_strategy_module()
    root = tmp_path

    non_rl_dir = root / "artifacts" / "models" / "lightgbm_real"
    non_rl_dir.mkdir(parents=True, exist_ok=True)
    (non_rl_dir / "training_summary.json").write_text(
        json.dumps({"model": "lightgbm", "model_path": "artifacts/models/lightgbm_real/lightgbm_model.txt"}),
        encoding="utf-8",
    )
    (non_rl_dir / "lightgbm_model.txt").write_text("model", encoding="utf-8")

    rl_dir = root / "artifacts" / "models" / "rl_manual_sanity"
    rl_dir.mkdir(parents=True, exist_ok=True)
    (rl_dir / "._ppo_trading_env.zip").write_bytes(b"\x00bad")
    (rl_dir / "ppo_trading_env.zip").write_bytes(b"zip")
    (rl_dir / "feature_snapshot.json").write_text(json.dumps({"features": ["close"]}), encoding="utf-8")
    (rl_dir / "expressions_snapshot.json").write_text("[]", encoding="utf-8")
    (rl_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "model": "ppo",
                "features": ["close"],
                "feature_snapshot": "/Users/shatianming/Downloads/Agent_market/artifacts/models/rl_manual_sanity/feature_snapshot.json",
                "expressions_snapshot": "/Users/shatianming/Downloads/Agent_market/artifacts/models/rl_manual_sanity/expressions_snapshot.json",
                "model_path": "/Users/shatianming/Downloads/Agent_market/artifacts/models/rl_manual_sanity/ppo_trading_env.zip",
            }
        ),
        encoding="utf-8",
    )

    expr_mod = types.ModuleType("agent_market.freqai.expression_engine")
    expr_mod.load_expression_file = lambda path: ["ok", str(path)]
    sys.modules["agent_market.freqai.expression_engine"] = expr_mod

    class _FakePPO:
        @staticmethod
        def load(path):  # noqa: ANN001
            assert Path(path).name == "ppo_trading_env.zip"
            return {"loaded": str(path)}

    sb3_mod = types.ModuleType("stable_baselines3")
    sb3_mod.PPO = _FakePPO
    sys.modules["stable_baselines3"] = sb3_mod

    monkeypatch.setattr(module, "_ROOT", root)

    strategy = module.FreqtradeRLStrategy(config={})
    strategy._load()

    assert strategy._features == ["close"]
    assert strategy._feature_cfg == {"features": ["close"]}
    assert strategy._expression_specs[1] == str(rl_dir / "expressions_snapshot.json")
    assert strategy._model["loaded"].endswith("ppo_trading_env.zip")


def test_rl_sb3_predict_uses_env_state_dims_and_action_mapping() -> None:
    module = _import_strategy_module()

    class _FakeSB3Model:
        policy = object()

        def __init__(self) -> None:
            self.observations = []

        def predict(self, obs, deterministic=True):  # noqa: ANN001
            self.observations.append(np.array(obs, copy=True))
            actions = [1, 0, 2]
            return actions[len(self.observations) - 1], None

    strategy = module.FreqtradeRLStrategy(config={})
    strategy._model = _FakeSB3Model()

    features = np.asarray(
        [
            [10.0, 20.0],
            [11.0, 21.0],
            [12.0, 22.0],
        ],
        dtype=np.float32,
    )
    prices = np.asarray([100.0, 110.0, 105.0], dtype=np.float32)

    preds = strategy._predict_action(features, prices)

    assert preds.tolist() == [1.0, 0.0, -1.0]
    first_obs, second_obs, third_obs = strategy._model.observations
    assert first_obs.tolist() == [10.0, 20.0, 0.0, 0.0]
    assert second_obs.tolist() == pytest.approx([11.0, 21.0, 1.0, 0.1])
    assert third_obs.tolist() == pytest.approx([12.0, 22.0, 1.0, 0.05])
