from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pandas import DataFrame

def _inject_project_paths() -> Path:
    here = Path(__file__).resolve()
    root: Path | None = None
    for parent in here.parents:
        if (parent / "src" / "agent_market").exists():
            root = parent
            break
    if root is None:
        # Fallback to the original heuristic
        root = here.parents[2] if len(here.parents) >= 3 else here.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_ROOT = _inject_project_paths()

from freqtrade.strategy import IStrategy  # noqa: E402

from agent_market.freqai.features import apply_configured_features  # noqa: E402
from agent_market.freqai.model.base import ModelRegistry  # noqa: E402
import agent_market.freqai.model  # noqa: F401, E402
from workspace.model_loader import scan_and_register  # noqa: E402


class MLStrategy_v2(IStrategy):
    timeframe = "5m"
    startup_candle_count = 200

    process_only_new_candles = True
    use_exit_signal = True
    ignore_roi_if_entry_signal = False

    minimal_roi = {"0": 0.0}
    stoploss = -0.99

    prediction_threshold: float = 0.0

    _MODEL_DIR = Path("/Users/shatianming/Downloads/Agent_market/workspace/results/model_auto_ml_v2")
    _REGISTRY_NAME = "auto_ml_v2"
    _FEATURE_CFG_PATH = _ROOT / "user_data" / "freqai_features_real.json"
    _PRED_COL = "ml_pred"

    _model: Any = None
    _feature_cols: list[str] | None = None
    _feature_cfg: dict | None = None
    _expressions_path: Path | None = None
    _expression_specs: list[Any] | None = None

    @staticmethod
    def _read_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)

    @staticmethod
    def _resolve_under_root(path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (_ROOT / p).resolve()

    @classmethod
    def _extract_feature_cols(cls, training_summary: dict) -> list[str]:
        for key in ("feature_columns", "features", "feature_cols"):
            cols = training_summary.get(key)
            if isinstance(cols, list) and cols and all(isinstance(c, str) for c in cols):
                return cols

        data = training_summary.get("data")
        if isinstance(data, dict):
            cols = data.get("feature_columns") or data.get("features")
            if isinstance(cols, list) and cols and all(isinstance(c, str) for c in cols):
                return cols

        raise ValueError(
            "training_summary.json missing feature column list (expected key: "
            "feature_columns/features/feature_cols)"
        )

    @classmethod
    def _ensure_resources_loaded(cls) -> None:
        if cls._model is not None and cls._feature_cols is not None and cls._feature_cfg is not None:
            return

        if not cls._MODEL_DIR.exists():
            raise FileNotFoundError(f"Model directory not found: {cls._MODEL_DIR}")

        training_summary_path = cls._MODEL_DIR / "training_summary.json"
        if not training_summary_path.exists():
            raise FileNotFoundError(f"training_summary.json not found: {training_summary_path}")

        training_summary = cls._read_json(training_summary_path)
        cls._feature_cols = cls._extract_feature_cols(training_summary)

        # Load the exact feature config used during training when available
        if cls._feature_cfg is None:
            feat_path_val = training_summary.get("feature_snapshot") or training_summary.get("feature_file")
            if feat_path_val:
                feat_path = cls._resolve_under_root(str(feat_path_val))
            else:
                feat_path = cls._FEATURE_CFG_PATH

            if not feat_path.exists():
                raise FileNotFoundError(f"Feature config not found: {feat_path}")
            cls._feature_cfg = cls._read_json(feat_path)

        # Load expression snapshot (needed for f001.. factors) when available
        expr_path_val = training_summary.get("expressions_snapshot") or training_summary.get("expressions_file")
        if expr_path_val:
            cls._expressions_path = cls._resolve_under_root(str(expr_path_val))
        else:
            cls._expressions_path = None

        scan_and_register()
        model = ModelRegistry.create(cls._REGISTRY_NAME, training_summary)
        model.load(str(cls._MODEL_DIR))
        cls._model = model

    @classmethod
    def _apply_expressions_if_needed(cls, df: DataFrame) -> DataFrame:
        if cls._expressions_path is None:
            return df
        if not cls._expressions_path.exists():
            return df

        if cls._expression_specs is None:
            from agent_market.freqai.expression_engine import load_expression_file  # noqa: WPS433

            cls._expression_specs = load_expression_file(cls._expressions_path)

        if not cls._expression_specs:
            return df

        from agent_market.freqai.expression_engine import apply_expressions  # noqa: WPS433

        df, _added = apply_expressions(df, cls._expression_specs, on_error="raise")
        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self._ensure_resources_loaded()
        df = dataframe

        df = apply_configured_features(df, self._feature_cfg)
        df = self._apply_expressions_if_needed(df)

        missing = [c for c in self._feature_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing feature columns in dataframe: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        X = df[self._feature_cols].to_numpy(dtype=np.float32, copy=False)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        preds = self._model.predict(X)
        preds = np.asarray(preds, dtype=np.float32).reshape(-1)
        if len(preds) != len(df):
            raise ValueError(f"Prediction length ({len(preds)}) != dataframe length ({len(df)})")

        df[self._PRED_COL] = preds
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, "enter_long"] = 0
        if self._PRED_COL in dataframe.columns:
            dataframe.loc[
                dataframe[self._PRED_COL] > float(self.prediction_threshold),
                "enter_long",
            ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, "exit_long"] = 0
        if self._PRED_COL in dataframe.columns:
            dataframe.loc[dataframe[self._PRED_COL] < 0.0, "exit_long"] = 1
        return dataframe