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
    interface_version = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count: int = 60

    minimal_roi = {"0": 0.01}
    stoploss = -0.05
    use_exit_signal = True

    threshold: float = 0.0

    _MODEL_DIR = Path("/Users/shatianming/Downloads/Agent_market/workspace/results/model_auto_ml_v1")
    _REGISTRY_NAME = "auto_ml_v1"

    _feature_cfg: dict | None = None
    _training_summary: dict | None = None
    _feature_cols: list[str] | None = None
    _model = None
    _model_loaded: bool = False
    _registry_scanned: bool = False

    @staticmethod
    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8-sig"))

    @classmethod
    def _load_feature_cfg(cls) -> dict:
        if cls._feature_cfg is not None:
            return cls._feature_cfg
        cfg_path = _ROOT / "user_data" / "freqai_features_real.json"
        cls._feature_cfg = cls._read_json(cfg_path)
        return cls._feature_cfg

    @classmethod
    def _load_training_summary(cls) -> dict:
        if cls._training_summary is not None:
            return cls._training_summary
        summary_path = cls._MODEL_DIR / "training_summary.json"
        cls._training_summary = cls._read_json(summary_path)
        return cls._training_summary

    @classmethod
    def _ensure_model_loaded(cls) -> None:
        if cls._model_loaded and cls._model is not None and cls._feature_cols is not None:
            return

        if not cls._registry_scanned:
            scan_and_register()
            cls._registry_scanned = True

        summary = cls._load_training_summary()
        cols = [str(c) for c in (summary.get("features") or []) if str(c).strip()]
        if not cols:
            raise ValueError(f"Model feature list missing in {cls._MODEL_DIR / 'training_summary.json'}")
        cls._feature_cols = cols

        model_config: dict = {"model_dir": str(cls._MODEL_DIR)}
        cls._model = ModelRegistry.create(cls._REGISTRY_NAME, model_config)

        model_path_raw = str(summary.get("model_path") or "").strip()
        model_path = Path(model_path_raw) if model_path_raw else cls._MODEL_DIR

        cls._model.load(model_path)
        cls._model_loaded = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        dataframe["prediction"] = np.nan

        try:
            self._ensure_model_loaded()

            cfg = self._load_feature_cfg()
            df = apply_configured_features(dataframe, cfg)
            if df is None:
                df = dataframe

            cols = self._feature_cols or []
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan

            matrix_df = (
                df[cols]
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .ffill()
                .bfill()
                .fillna(0.0)
            )
            X = matrix_df.to_numpy(dtype=np.float32, copy=False)

            pred = self._model.predict(X)
            pred = np.asarray(pred, dtype=np.float32).reshape(-1)
            if pred.shape[0] != len(df):
                dataframe["prediction"] = np.nan
                return dataframe

            dataframe["prediction"] = pred
            return dataframe
        except Exception:
            return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe
        if "prediction" not in dataframe.columns:
            dataframe["enter_long"] = 0
            return dataframe
        cond = (dataframe["volume"] > 0) & (dataframe["prediction"] > float(self.threshold))
        dataframe.loc[cond, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe
        if "prediction" not in dataframe.columns:
            dataframe["exit_long"] = 0
            return dataframe
        cond = (dataframe["volume"] > 0) & (dataframe["prediction"] < 0.0)
        dataframe.loc[cond, "exit_long"] = 1
        return dataframe