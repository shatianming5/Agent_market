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


class MLStrategy_v3(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False

    minimal_roi = {"0": 0.10}
    stoploss = -0.10
    trailing_stop = False

    process_only_new_candles = True
    startup_candle_count = 200

    # Prediction threshold for entries
    threshold: float = 0.0

    # Internal cache
    _ml_model = None
    _ml_feature_cols: list[str] | None = None
    _ml_ready: bool = False
    _ml_error: str | None = None

    MODEL_DIR = Path("/Users/shatianming/Downloads/Agent_market/workspace/results/model_auto_ml_v3")
    REGISTRY_NAME = "auto_ml_v3"

    @staticmethod
    def _extract_feature_columns(summary: dict) -> list[str]:
        for key in ("feature_columns", "features", "training_features", "used_features", "feature_names", "columns"):
            val = summary.get(key)
            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                return val

        # Common nested shapes
        for key in ("data", "train", "training", "meta", "metadata"):
            nest = summary.get(key)
            if isinstance(nest, dict):
                cols = MLStrategy_v3._extract_feature_columns(nest)
                if cols:
                    return cols

        raise KeyError(
            "Unable to find feature column names in training_summary.json. "
            "Expected keys like 'feature_columns' or 'features'."
        )

    def _ensure_model_loaded(self) -> None:
        if self._ml_ready:
            return

        try:
            model_dir = self.MODEL_DIR
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")

            feature_cfg_path = _ROOT / "user_data" / "freqai_features_real.json"
            if not feature_cfg_path.exists():
                raise FileNotFoundError(f"Feature config not found: {feature_cfg_path}")

            training_summary_path = model_dir / "training_summary.json"
            if not training_summary_path.exists():
                raise FileNotFoundError(f"training_summary.json not found: {training_summary_path}")

            cfg = json.loads(feature_cfg_path.read_text(encoding="utf-8"))
            summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
            feature_cols = self._extract_feature_columns(summary)

            # Ensure custom model classes are registered before create()
            scan_and_register()

            config = {
                "feature_config_path": str(feature_cfg_path),
                "model_dir": str(model_dir),
                "training_summary": summary,
                "feature_columns": feature_cols,
            }
            model = ModelRegistry.create(self.REGISTRY_NAME, config)
            model.load(str(model_dir))

            self._ml_model = model
            self._ml_feature_cols = feature_cols
            self._ml_ready = True
            self._ml_error = None
        except Exception as e:
            self._ml_model = None
            self._ml_feature_cols = None
            self._ml_ready = False
            self._ml_error = f"{type(e).__name__}: {e}"

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Always provide the prediction column (safe default)
        if dataframe.empty:
            dataframe["prediction"] = np.nan
            return dataframe

        self._ensure_model_loaded()

        try:
            feature_cfg_path = _ROOT / "user_data" / "freqai_features_real.json"
            cfg = json.loads(feature_cfg_path.read_text(encoding="utf-8"))

            # Apply configured features
            dataframe = apply_configured_features(dataframe, cfg)

            if not self._ml_ready or self._ml_model is None or not self._ml_feature_cols:
                dataframe["prediction"] = 0.0
                return dataframe

            # Build feature matrix in training feature order
            missing = [c for c in self._ml_feature_cols if c not in dataframe.columns]
            if missing:
                raise KeyError(f"Missing required feature columns in dataframe: {missing[:20]}")

            X = dataframe.loc[:, self._ml_feature_cols].to_numpy(dtype=np.float32, copy=False)
            # Replace NaN/inf to keep model inference stable
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            preds = self._ml_model.predict(X)
            preds = np.asarray(preds, dtype=np.float32).reshape(-1)
            if len(preds) != len(dataframe):
                raise ValueError(f"Prediction length mismatch: got {len(preds)} expected {len(dataframe)}")

            dataframe["prediction"] = preds
            return dataframe
        except Exception:
            dataframe["prediction"] = 0.0
            return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe

        if "prediction" not in dataframe.columns:
            dataframe["prediction"] = 0.0

        cond = dataframe["prediction"] > float(self.threshold)
        dataframe.loc[cond, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe

        if "prediction" not in dataframe.columns:
            dataframe["prediction"] = 0.0

        cond = dataframe["prediction"] < 0.0
        dataframe.loc[cond, "exit_long"] = 1
        return dataframe