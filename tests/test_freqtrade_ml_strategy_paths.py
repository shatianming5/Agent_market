from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path


def _import_strategy_module():
    freqtrade_mod = types.ModuleType("freqtrade")
    strategy_mod = types.ModuleType("freqtrade.strategy")

    class _IStrategy:  # noqa: D401
        pass

    strategy_mod.IStrategy = _IStrategy
    sys.modules.setdefault("freqtrade", freqtrade_mod)
    sys.modules["freqtrade.strategy"] = strategy_mod
    sys.modules.pop("workspace.strategies.type_F_ml.freqtrade_ml_strategy", None)
    return importlib.import_module("workspace.strategies.type_F_ml.freqtrade_ml_strategy")


def test_resolve_summary_artifact_falls_back_to_model_dir(tmp_path, monkeypatch) -> None:
    module = _import_strategy_module()
    model_dir = tmp_path / "artifacts" / "models" / "lightgbm_real"
    model_dir.mkdir(parents=True, exist_ok=True)
    local_model = model_dir / "lightgbm_model.txt"
    local_model.write_text("dummy", encoding="utf-8")
    monkeypatch.setattr(module, "_ROOT", tmp_path)

    resolved = module._resolve_summary_artifact(
        "/Users/shatianming/Downloads/Agent_market/artifacts/models/lightgbm_real/lightgbm_model.txt",
        model_dir=model_dir,
        fallback_names=["lightgbm_model.txt"],
    )

    assert resolved == local_model


def test_load_model_uses_local_artifacts_when_summary_paths_are_stale(tmp_path, monkeypatch) -> None:
    module = _import_strategy_module()
    model_dir = tmp_path / "artifacts" / "models" / "lightgbm_real"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "lightgbm_model.txt").write_text("model", encoding="utf-8")
    (model_dir / "feature_snapshot.json").write_text(json.dumps({"features": ["close"]}), encoding="utf-8")
    (model_dir / "expressions_snapshot.json").write_text("[]", encoding="utf-8")
    (model_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "model": "lightgbm",
                "features": ["close"],
                "feature_snapshot": "/Users/shatianming/Downloads/Agent_market/artifacts/models/lightgbm_real/feature_snapshot.json",
                "expressions_snapshot": "/Users/shatianming/Downloads/Agent_market/artifacts/models/lightgbm_real/expressions_snapshot.json",
                "model_path": "/Users/shatianming/Downloads/Agent_market/artifacts/models/lightgbm_real/lightgbm_model.txt",
            }
        ),
        encoding="utf-8",
    )

    class _DummyBooster:
        def __init__(self, *, model_file: str):
            self.model_file = model_file

        def predict(self, values):  # noqa: ANN001
            return values[:, 0]

    lightgbm_mod = types.ModuleType("lightgbm")
    lightgbm_mod.Booster = _DummyBooster
    sys.modules["lightgbm"] = lightgbm_mod

    expr_mod = types.ModuleType("agent_market.freqai.expression_engine")
    expr_mod.load_expression_file = lambda path: ["ok", str(path)]
    sys.modules["agent_market.freqai.expression_engine"] = expr_mod

    monkeypatch.setattr(module, "_ROOT", tmp_path)
    monkeypatch.setattr(module.FreqtradeMLStrategy, "_find_model_dir", lambda self: model_dir)

    strategy = module.FreqtradeMLStrategy()
    strategy._load_model()

    assert strategy._feature_cfg == {"features": ["close"]}
    assert strategy._expression_specs[1] == str(model_dir / "expressions_snapshot.json")
    assert strategy._model.model_file == str(model_dir / "lightgbm_model.txt")

