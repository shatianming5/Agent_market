from __future__ import annotations

import importlib
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np


def _import_strategy_module():
    freqtrade_mod = types.ModuleType("freqtrade")
    strategy_mod = types.ModuleType("freqtrade.strategy")

    class _IStrategy:
        pass

    strategy_mod.IStrategy = _IStrategy
    sys.modules.setdefault("freqtrade", freqtrade_mod)
    sys.modules["freqtrade.strategy"] = strategy_mod
    sys.modules.pop("workspace.strategies.type_F_ml.freqtrade_dl_strategy", None)
    return importlib.import_module("workspace.strategies.type_F_ml.freqtrade_dl_strategy")


def test_dl_resolve_summary_artifact_falls_back_to_model_dir(tmp_path, monkeypatch) -> None:
    module = _import_strategy_module()
    model_dir = tmp_path / "artifacts" / "models" / "pytorch_dl"
    model_dir.mkdir(parents=True, exist_ok=True)
    local_model = model_dir / "model.pkl"
    local_model.write_bytes(b"dummy")
    monkeypatch.setattr(module, "_ROOT", tmp_path)

    resolved = module._resolve_summary_artifact(
        "/Users/shatianming/Downloads/Agent_market/artifacts/models/pytorch_dl/model.pkl",
        model_dir=model_dir,
        fallback_names=["model.pkl"],
    )

    assert resolved == local_model


def test_dl_load_uses_local_artifacts_when_summary_paths_are_stale(tmp_path, monkeypatch) -> None:
    module = _import_strategy_module()
    model_dir = tmp_path / "artifacts" / "models" / "pytorch_dl"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "feature_snapshot.json").write_text(json.dumps({"features": ["close"]}), encoding="utf-8")
    (model_dir / "expressions_snapshot.json").write_text("[]", encoding="utf-8")
    with (model_dir / "model.pkl").open("wb") as handle:
        pickle.dump({"weights": [1.0], "bias": 0.0, "mean": [0.0], "std": [1.0]}, handle)
    (model_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "model": "custom_pickle",
                "features": ["close"],
                "feature_snapshot": "/Users/shatianming/Downloads/Agent_market/artifacts/models/pytorch_dl/feature_snapshot.json",
                "expressions_snapshot": "/Users/shatianming/Downloads/Agent_market/artifacts/models/pytorch_dl/expressions_snapshot.json",
                "model_path": "/Users/shatianming/Downloads/Agent_market/artifacts/models/pytorch_dl/model.pkl",
            }
        ),
        encoding="utf-8",
    )

    expr_mod = types.ModuleType("agent_market.freqai.expression_engine")
    expr_mod.load_expression_file = lambda path: ["ok", str(path)]
    sys.modules["agent_market.freqai.expression_engine"] = expr_mod

    monkeypatch.setattr(module, "_ROOT", tmp_path)

    strategy = module.FreqtradeDLStrategy()
    strategy._load()

    assert strategy._feature_cfg == {"features": ["close"]}
    assert strategy._expression_specs[1] == str(model_dir / "expressions_snapshot.json")
    assert strategy._model["weights"] == [1.0]
    assert strategy._features == ["close"]


def test_dl_load_ignores_appledouble_pt_sidecars(tmp_path, monkeypatch) -> None:
    module = _import_strategy_module()
    model_dir = tmp_path / "artifacts" / "models" / "pytorch_dl"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "._pytorch_mlp.pt").write_bytes(b"\x00bad")
    (model_dir / "pytorch_mlp.pt").write_bytes(b"good")
    (model_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "model": "pytorch_mlp",
                "features": ["close"],
            }
        ),
        encoding="utf-8",
    )

    class _FakeTorch(types.ModuleType):
        def __init__(self):
            super().__init__("torch")

        @staticmethod
        def load(path, map_location="cpu", weights_only=False):  # noqa: ANN001
            assert Path(path).name == "pytorch_mlp.pt"
            return {"weights": np.array([1.0]), "bias": 0.0, "mean": np.array([0.0]), "std": np.array([1.0])}

    sys.modules["torch"] = _FakeTorch()

    monkeypatch.setattr(module, "_ROOT", tmp_path)

    strategy = module.FreqtradeDLStrategy()
    strategy._load()

    assert isinstance(strategy._model, dict)
    assert strategy._model["bias"] == 0.0
