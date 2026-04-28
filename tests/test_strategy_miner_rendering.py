from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np


def test_render_ml_strategy_code_wraps_xgboost_inputs_in_dmatrix(tmp_path, monkeypatch):
    from agent_market.strategy_miner._rendering import _render_ml_strategy_code

    summary_path = tmp_path / "training_summary.json"
    model_path = tmp_path / "xgb_model.json"
    model_path.write_text("{}", encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "model": "xgboost",
                "model_path": str(model_path),
                "features": ["f1", "f2"],
            }
        ),
        encoding="utf-8",
    )

    fake_freqtrade = types.ModuleType("freqtrade")
    fake_strategy = types.ModuleType("freqtrade.strategy")

    class DummyIStrategy:
        pass

    fake_strategy.IStrategy = DummyIStrategy
    monkeypatch.setitem(sys.modules, "freqtrade", fake_freqtrade)
    monkeypatch.setitem(sys.modules, "freqtrade.strategy", fake_strategy)

    captured: dict[str, object] = {}
    fake_xgb = types.ModuleType("xgboost")

    class FakeDMatrix:
        def __init__(self, data):
            captured["dmatrix_type"] = type(data)
            captured["dmatrix_data"] = np.asarray(data)

    class FakeBooster:
        def load_model(self, path):
            captured["model_path"] = path

        def predict(self, matrix):
            captured["predict_arg_type"] = type(matrix)
            return np.array([0.25, -0.1], dtype=np.float32)

    fake_xgb.DMatrix = FakeDMatrix
    fake_xgb.Booster = FakeBooster
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgb)

    source = _render_ml_strategy_code(
        class_name="XgbWrapperTest",
        timeframe="5m",
        summary_path=summary_path,
        enter_threshold=0.01,
        exit_threshold=-0.01,
        minimal_roi={"0": 0.01},
        stoploss=-0.02,
        max_hold_minutes=60,
        startup_candle_count=100,
    )

    ns: dict[str, object] = {}
    exec(source, ns, ns)
    strategy = ns["XgbWrapperTest"]()
    strategy._load_model()

    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    preds = strategy._predict(matrix)

    assert strategy._model_name == "xgboost"
    assert captured["model_path"] == str(model_path)
    assert captured["predict_arg_type"] is FakeDMatrix
    assert captured["dmatrix_type"] is np.ndarray
    np.testing.assert_array_equal(captured["dmatrix_data"], matrix.astype(np.float32))
    np.testing.assert_array_equal(preds, np.array([0.25, -0.1], dtype=np.float32))
