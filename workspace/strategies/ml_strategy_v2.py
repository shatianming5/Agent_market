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
    timeframe = "1h"
    can_short = False
    process_only_new_candles = True

    # Freqtrade requires stoploss to be defined either in config or strategy.
    stoploss = -0.99

    threshold: float = 0.003

    _MODEL_DIR = Path("/Users/shatianming/Downloads/Agent_market/workspace/results/model_auto_dl_v2")
    _REGISTRY_NAME = "auto_dl_v2"

    _feature_cfg: dict | None = None
    _training_summary: dict | None = None
    _feature_cols: list[str] | None = None
    _model = None
    _model_loaded: bool = False

    @staticmethod
    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def _load_feature_cfg(self) -> dict:
        if self._feature_cfg is not None:
            return self._feature_cfg
        cfg_path = _ROOT / "user_data" / "freqai_features_real.json"
        self._feature_cfg = self._read_json(cfg_path)
        return self._feature_cfg

    def _load_training_summary(self) -> dict:
        if self._training_summary is not None:
            return self._training_summary
        summary_path = self._MODEL_DIR / "training_summary.json"
        self._training_summary = self._read_json(summary_path)
        return self._training_summary

    def _ensure_model_loaded(self) -> None:
        if self._model_loaded and self._model is not None and self._feature_cols is not None:
            return

        summary = self._load_training_summary()
        cols = [str(c) for c in (summary.get("features") or []) if str(c).strip()]
        if not cols:
            raise ValueError(f"Model feature list missing in {self._MODEL_DIR / 'training_summary.json'}")
        self._feature_cols = cols

        scan_and_register()

        model_config = {"model_dir": str(self._MODEL_DIR)}
        self._model = ModelRegistry.create(self._REGISTRY_NAME, model_config)

        model_path_raw = summary.get("model_path") or ""
        model_path = (
            Path(str(model_path_raw))
            if str(model_path_raw).strip()
            else (self._MODEL_DIR / f"{self._REGISTRY_NAME}.pkl")
        )
        if model_path.is_dir():
            model_path = model_path / f"{self._REGISTRY_NAME}.pkl"
        if not model_path.exists():
            model_path = self._MODEL_DIR / f"{self._REGISTRY_NAME}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model.load(model_path)
        self._model_loaded = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        self._ensure_model_loaded()

        cfg = self._load_feature_cfg()
        df = apply_configured_features(dataframe, cfg)

        cols = self._feature_cols or []
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan

        X_df = (
            df[cols]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0.0)
        )
        X = X_df.to_numpy(dtype=np.float32, copy=False)

        pred = self._model.predict(X)
        pred = np.asarray(pred, dtype=np.float32).reshape(-1)
        if pred.shape[0] != len(df):
            pred = np.resize(pred, len(df)).astype(np.float32, copy=False)

        df["prediction"] = pred
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty or "prediction" not in dataframe.columns:
            return dataframe
        dataframe.loc[dataframe["prediction"] > float(self.threshold), "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty or "prediction" not in dataframe.columns:
            return dataframe
        dataframe.loc[dataframe["prediction"] < 0.0, "exit_long"] = 1
        return dataframe