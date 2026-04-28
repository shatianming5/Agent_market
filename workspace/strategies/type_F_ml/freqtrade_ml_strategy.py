"""Freqtrade ML Strategy — loads a trained model and uses predictions for trading.

Works with: LightGBM, XGBoost, Ridge, or any BaseModelAdapter.
Reads model from training_summary.json → loads features → predicts → trades.

Config required in freqtrade config JSON:
  No extra config needed — reads from artifacts/models/<model_dir>/

Usage:
  freqtrade backtesting --strategy FreqtradeMLStrategy --strategy-path workspace/strategies/type_F_ml
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pandas import DataFrame

# Project path injection
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]
for p in [str(_ROOT / "src"), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from freqtrade.strategy import IStrategy


def _resolve_summary_artifact(
    raw_path: Optional[str],
    *,
    model_dir: Path,
    fallback_names: Optional[List[str]] = None,
) -> Optional[Path]:
    """Resolve summary artifact paths across machines.

    Training summaries often store absolute paths from the machine that created the
    model. When the model directory is copied to a different host, prefer local
    siblings in `model_dir` before failing on the stale absolute path.
    """
    candidates: List[Path] = []
    if raw_path:
        raw = Path(str(raw_path))
        if raw.is_absolute():
            candidates.append(model_dir / raw.name)
            candidates.append(raw)
        else:
            candidates.append(_ROOT / raw)
            candidates.append(model_dir / raw.name)
    for fallback in fallback_names or []:
        fallback_path = Path(str(fallback))
        candidates.append(fallback_path if fallback_path.is_absolute() else (model_dir / fallback_path))

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate

    return candidates[0] if candidates else None


class FreqtradeMLStrategy(IStrategy):
    """Loads a pre-trained ML model and trades based on its predictions."""

    timeframe = "1h"
    can_short = False
    minimal_roi = {"0": 0.10, "240": 0.03, "720": -1}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 60

    # ML parameters
    ml_enter_threshold = 0.003
    ml_exit_threshold = 0.0

    # Model discovery: picks the most recently trained model
    _model = None
    _model_features: Optional[List[str]] = None
    _feature_cfg: Optional[Dict[str, Any]] = None
    _expression_specs = None
    _scaler = None
    _loaded = False

    def _find_model_dir(self) -> Path:
        """Find the most recently modified model directory."""
        models_root = _ROOT / "artifacts" / "models"
        candidates = [
            d for d in models_root.iterdir()
            if d.is_dir() and (d / "training_summary.json").exists()
        ]
        if not candidates:
            raise FileNotFoundError(f"No trained models in {models_root}")
        return max(candidates, key=lambda d: (d / "training_summary.json").stat().st_mtime)

    def _load_model(self):
        """Load model, features, and expressions."""
        if self._loaded:
            return

        model_dir = self._find_model_dir()
        summary_path = model_dir / "training_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        data_section = summary.get("data") if isinstance(summary.get("data"), dict) else {}
        summary_timeframe = str(data_section.get("timeframe") or "").strip()
        if summary_timeframe:
            self.timeframe = summary_timeframe

        # Load feature config
        feature_path = _resolve_summary_artifact(
            summary.get("feature_snapshot") or summary.get("feature_file"),
            model_dir=model_dir,
            fallback_names=["feature_snapshot.json"],
        )
        if feature_path and feature_path.exists():
            self._feature_cfg = json.loads(feature_path.read_text(encoding="utf-8-sig"))

        if self._feature_cfg is None:
            for candidate in [_ROOT / "user_data" / "freqai_features_real.json",
                              _ROOT / "user_data" / "freqai_features.json"]:
                if candidate.exists():
                    self._feature_cfg = json.loads(candidate.read_text(encoding="utf-8-sig"))
                    break

        # Load expressions
        expr_path = _resolve_summary_artifact(
            summary.get("expressions_snapshot") or summary.get("expressions_file"),
            model_dir=model_dir,
            fallback_names=["expressions_snapshot.json"],
        )
        if expr_path and expr_path.exists():
            from agent_market.freqai.expression_engine import load_expression_file
            self._expression_specs = load_expression_file(expr_path)

        scaler_path = model_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as handle:
                scaler_payload = pickle.load(handle)
            self._scaler = scaler_payload.get("scaler")

        # Load model
        self._model_features = [str(c) for c in (summary.get("features") or []) if str(c).strip()]
        model_path = _resolve_summary_artifact(
            summary.get("model_path"),
            model_dir=model_dir,
            fallback_names=["lightgbm_model.txt", "model.pkl", "model.pt", "model.json"],
        )
        if model_path is None:
            raise FileNotFoundError("Model file path missing in training summary")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_name = summary.get("model", "lightgbm")
        if model_name == "lightgbm":
            import lightgbm as lgb
            self._model = lgb.Booster(model_file=str(model_path))
        elif model_name in ("xgboost", "xgb"):
            import xgboost as xgb
            self._model = xgb.Booster()
            self._model.load_model(str(model_path))
        elif model_name == "pytorch_mlp":
            import torch
            checkpoint = torch.load(model_path, map_location="cpu")
            from agent_market.freqai.model.torch_models import FeedForwardNet
            cfg = checkpoint.get("config", {})
            model = FeedForwardNet(
                checkpoint.get("input_dim", len(self._model_features)),
                cfg.get("hidden_dims", [64, 32]),
                cfg.get("dropout", 0.0),
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self._model = model
        else:
            # Generic pickle model
            import pickle
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)

        self._loaded = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Run prediction through the loaded model."""
        if hasattr(self._model, "predict"):
            # LightGBM Booster, sklearn, etc.
            return self._model.predict(X)
        elif hasattr(self._model, "forward"):
            # PyTorch
            import torch
            with torch.no_grad():
                tensor = torch.from_numpy(X.astype(np.float32))
                return self._model(tensor).cpu().numpy()
        else:
            raise RuntimeError(f"Unknown model type: {type(self._model)}")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self._load_model()

        # Apply feature engineering
        if self._feature_cfg:
            from agent_market.freqai.features import apply_configured_features
            dataframe = apply_configured_features(dataframe, self._feature_cfg)

        # Apply expressions
        if self._expression_specs:
            from agent_market.freqai.expression_engine import apply_expressions
            dataframe, _ = apply_expressions(dataframe, self._expression_specs, on_error="raise")

        # Ensure all model features exist
        for col in self._model_features:
            if col not in dataframe.columns:
                dataframe[col] = 0.0

        # Build feature matrix and predict
        matrix = (
            dataframe[self._model_features]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .fillna(0.0)
        )
        matrix_values = matrix.to_numpy(dtype=np.float32)
        if self._scaler is not None:
            matrix_values = self._scaler.transform(matrix_values)
        preds = self._predict(matrix_values)
        dataframe["ml_pred"] = preds.reshape(-1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (dataframe["ml_pred"] > self.ml_enter_threshold),
            ["enter_long", "enter_tag"],
        ] = (1, "ml_long")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (dataframe["ml_pred"] < self.ml_exit_threshold),
            "exit_long",
        ] = 1
        return dataframe
