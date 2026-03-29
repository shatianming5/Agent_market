from __future__ import annotations
import sys, json
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


class MLStrategy_v1(IStrategy):
    timeframe = "1h"
    process_only_new_candles = True
    startup_candle_count = 200

    minimal_roi = {"0": 0.10}
    stoploss = -0.10

    threshold: float = 0.0

    model_dir: Path = Path("/Users/shatianming/Downloads/Agent_market/workspace/results/model_auto_rl_v1")
    registry_name: str = "auto_rl_v1"

    pred_col: str = "ml_pred"

    _feature_cfg: dict | None = None
    _feature_cols: list[str] | None = None
    _model = None
    _model_path: Path | None = None

    def _load_feature_cfg(self) -> dict:
        if self.__class__._feature_cfg is not None:
            return self.__class__._feature_cfg
        cfg_path = _ROOT / "user_data" / "freqai_features_real.json"
        with cfg_path.open("r", encoding="utf-8") as f:
            self.__class__._feature_cfg = json.load(f)
        return self.__class__._feature_cfg

    def _load_training_summary(self) -> dict:
        summary_path = self.model_dir / "training_summary.json"
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _ensure_model_loaded(self) -> None:
        if self.__class__._model is not None and self.__class__._model_path is not None:
            return

        scan_and_register()

        summary = self._load_training_summary()
        feature_cols = summary.get("features") or []
        if not isinstance(feature_cols, list) or not feature_cols:
            raise ValueError("training_summary.json missing non-empty 'features' list")
        self.__class__._feature_cols = [str(c) for c in feature_cols]

        model_path = summary.get("model_path")
        if model_path:
            model_path = Path(str(model_path))
        else:
            model_path = self.model_dir / f"{self.registry_name}.pkl"

        config = {"model_dir": str(self.model_dir)}
        model = ModelRegistry.create(self.registry_name, config)
        model.load(model_path)

        self.__class__._model = model
        self.__class__._model_path = model_path

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        cfg = self._load_feature_cfg()
        dataframe = apply_configured_features(dataframe, cfg)

        self._ensure_model_loaded()
        feature_cols = self.__class__._feature_cols
        model = self.__class__._model
        if feature_cols is None or model is None:
            raise RuntimeError("Model or feature columns not initialized")

        for col in feature_cols:
            if col not in dataframe.columns:
                dataframe[col] = np.nan

        X = dataframe[feature_cols].to_numpy(dtype=np.float32, copy=False)
        preds = model.predict(X)
        preds = np.asarray(preds, dtype=np.float32).reshape(-1)
        if preds.shape[0] != len(dataframe):
            raise RuntimeError(f"predict() length mismatch: got {preds.shape[0]} expected {len(dataframe)}")

        dataframe[self.pred_col] = preds
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        if self.pred_col not in dataframe.columns:
            dataframe["enter_long"] = 0
            return dataframe

        pred = dataframe[self.pred_col].to_numpy(dtype=np.float32, copy=False)
        enter = np.isfinite(pred) & (pred > float(self.threshold))

        dataframe["enter_long"] = 0
        dataframe.loc[enter, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        if self.pred_col not in dataframe.columns:
            dataframe["exit_long"] = 0
            return dataframe

        pred = dataframe[self.pred_col].to_numpy(dtype=np.float32, copy=False)
        exit_ = np.isfinite(pred) & (pred < 0.0)

        dataframe["exit_long"] = 0
        dataframe.loc[exit_, "exit_long"] = 1
        return dataframe