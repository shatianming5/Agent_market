from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
from pandas import DataFrame

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
    sys.path.insert(0, str(_ROOT))

from freqtrade.strategy import IStrategy
from agent_market.freqai.features import apply_configured_features
from agent_market.freqai.model.base import ModelRegistry
import agent_market.freqai.model
from workspace.model_loader import scan_and_register


class MLStrategy_v2(IStrategy):
    timeframe = "5m"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 200

    stoploss = -0.1

    buy_threshold: float = 0.0

    _MODEL_DIR = Path("/Users/shatianming/Downloads/Agent_market/workspace/results/model_auto_dl_v2")
    _REGISTRY_NAME = "auto_dl_v2"
    _PRED_COL = "ml_pred"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._model = None
        self._feature_columns: list[str] | None = None
        self._model_ready: bool = False

    @staticmethod
    def _read_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _extract_feature_columns(training_summary: dict) -> list[str]:
        for key in ("feature_columns", "feature_cols", "features", "feature_names", "columns"):
            v = training_summary.get(key)
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return v

        data = training_summary.get("data")
        if isinstance(data, dict):
            for key in ("feature_columns", "feature_cols", "features", "feature_names", "columns"):
                v = data.get(key)
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    return v

        feats = training_summary.get("features")
        if isinstance(feats, dict):
            v = feats.get("columns") or feats.get("feature_columns") or feats.get("names")
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return v

        raise ValueError("Could not find feature column names in training_summary.json")

    def _ensure_model_loaded(self) -> None:
        if self._model_ready:
            return

        model_dir = self._MODEL_DIR
        training_summary_path = model_dir / "training_summary.json"
        if not training_summary_path.exists():
            raise FileNotFoundError(f"Missing training summary: {training_summary_path}")

        training_summary = self._read_json(training_summary_path)
        self._feature_columns = self._extract_feature_columns(training_summary)

        scan_and_register()

        model_config = {}
        cfg = training_summary.get("config")
        if isinstance(cfg, dict):
            model_config.update(cfg)
        cfg2 = training_summary.get("model_config")
        if isinstance(cfg2, dict):
            model_config.update(cfg2)

        model_config.setdefault("model_dir", str(model_dir))
        model_config.setdefault("registry_name", self._REGISTRY_NAME)

        self._model = ModelRegistry.create(self._REGISTRY_NAME, model_config)
        self._model.load(str(model_dir))

        self._model_ready = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        features_cfg_path = _ROOT / "user_data" / "freqai_features_real.json"
        if not features_cfg_path.exists():
            raise FileNotFoundError(f"Missing feature config: {features_cfg_path}")

        features_cfg = self._read_json(features_cfg_path)
        dataframe = apply_configured_features(dataframe, features_cfg)

        self._ensure_model_loaded()
        assert self._feature_columns is not None
        assert self._model is not None

        missing = [c for c in self._feature_columns if c not in dataframe.columns]
        if missing:
            raise KeyError(
                f"Missing feature columns in dataframe: {missing[:20]}{'...' if len(missing) > 20 else ''}"
            )

        X_df = dataframe[self._feature_columns].copy()
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X = np.nan_to_num(X_df.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        preds = self._model.predict(X)
        preds_arr = np.asarray(preds, dtype=np.float32).reshape(-1)

        if preds_arr.shape[0] != len(dataframe):
            raise ValueError(f"Prediction length mismatch: preds={preds_arr.shape[0]} rows={len(dataframe)}")

        dataframe[self._PRED_COL] = preds_arr
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe
        if self._PRED_COL not in dataframe.columns:
            dataframe[self._PRED_COL] = 0.0

        dataframe.loc[
            (dataframe[self._PRED_COL] > float(self.buy_threshold)),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe
        if self._PRED_COL not in dataframe.columns:
            dataframe[self._PRED_COL] = 0.0

        dataframe.loc[
            (dataframe[self._PRED_COL] < 0.0),
            "exit_long",
        ] = 1
        return dataframe