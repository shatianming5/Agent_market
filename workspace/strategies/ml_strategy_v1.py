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
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count: int = 200

    minimal_roi = {"0": 0.10}
    stoploss = -0.10
    use_exit_signal = True

    threshold: float = 0.0

    model_dir: Path = Path("/Users/shatianming/Downloads/Agent_market/workspace/results/model_auto_ml_v1")
    registry_name: str = "auto_ml_v1"

    _did_register: bool = False
    _feature_cfg: dict | None = None
    _training_summary: dict | None = None
    _model: object | None = None
    _feature_columns: list[str] | None = None

    @staticmethod
    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8-sig"))

    @classmethod
    def _ensure_registered(cls) -> None:
        if cls._did_register:
            return
        scan_and_register()
        cls._did_register = True

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
        summary_path = cls.model_dir / "training_summary.json"
        cls._training_summary = cls._read_json(summary_path)
        return cls._training_summary

    @classmethod
    def _extract_feature_columns(cls, summary: dict) -> list[str]:
        def _extract(d: dict) -> list[str] | None:
            for key in ("feature_columns", "features", "training_features", "used_features", "feature_names", "columns"):
                v = d.get(key)
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    cols = [c for c in (s.strip() for s in v) if c]
                    if cols:
                        return cols
            feats = d.get("features")
            if isinstance(feats, dict):
                for key in ("columns", "feature_columns", "feature_names", "names"):
                    v = feats.get(key)
                    if isinstance(v, list) and all(isinstance(x, str) for x in v):
                        cols = [c for c in (s.strip() for s in v) if c]
                        if cols:
                            return cols
            for key in ("data", "train", "training", "meta", "metadata"):
                nest = d.get(key)
                if isinstance(nest, dict):
                    got = _extract(nest)
                    if got:
                        return got
            return None

        cols = _extract(summary)
        if not cols:
            raise KeyError("Could not find feature column names in training_summary.json")
        return cols

    @classmethod
    def _resolve_model_path(cls, summary: dict) -> Path:
        model_path = summary.get("model_path")
        if model_path:
            p = Path(str(model_path))
            if p.exists():
                return p

        if cls.model_dir.is_file():
            return cls.model_dir

        for candidate in ("auto_ml_v1.pkl", "model.pkl", "model.joblib", "model.onnx"):
            p = cls.model_dir / candidate
            if p.exists():
                return p

        return cls.model_dir

    @classmethod
    def _ensure_model_loaded(cls) -> tuple[object, list[str], dict]:
        if cls._model is not None and cls._feature_columns is not None and cls._training_summary is not None:
            return cls._model, cls._feature_columns, cls._training_summary

        cls._ensure_registered()
        summary = cls._load_training_summary()
        cols = cls._extract_feature_columns(summary)

        config = {
            "model_dir": str(cls.model_dir),
            "training_summary": summary,
            "feature_columns": cols,
            "registry_name": cls.registry_name,
        }

        model = ModelRegistry.create(cls.registry_name, config)
        model_path = cls._resolve_model_path(summary)

        # Some adapters historically implemented `load` as a @classmethod returning a new instance
        # (instead of mutating `self`). Support both.
        loaded = model.load(model_path)
        if loaded is not None:
            model = loaded

        cls._model = model
        cls._feature_columns = cols
        cls._training_summary = summary
        return model, cols, summary

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None:
            return dataframe

        if dataframe.empty:
            dataframe["predictions"] = np.nan
            return dataframe

        cfg = self._load_feature_cfg()
        dataframe = apply_configured_features(dataframe, cfg)

        model, cols, _summary = self._ensure_model_loaded()

        missing = [c for c in cols if c not in dataframe.columns]
        if missing:
            for c in missing:
                dataframe[c] = 0.0

        X_df = dataframe.loc[:, cols].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        X = np.nan_to_num(X_df.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        preds = model.predict(X)
        preds = np.asarray(preds, dtype=np.float32).reshape(-1)

        if preds.shape[0] != len(dataframe):
            raise RuntimeError(f"predict() returned wrong length: {preds.shape[0]} vs {len(dataframe)}")

        dataframe["predictions"] = preds
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe
        if "predictions" not in dataframe.columns:
            dataframe["predictions"] = 0.0

        cond = (dataframe["volume"] > 0) & (dataframe["predictions"] > float(self.threshold))
        dataframe.loc[cond, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe
        if "predictions" not in dataframe.columns:
            dataframe["predictions"] = 0.0

        cond = (dataframe["volume"] > 0) & (dataframe["predictions"] < 0.0)
        dataframe.loc[cond, "exit_long"] = 1
        return dataframe