"""Freqtrade RL Strategy — uses a trained RL model for trading decisions.

Loads RL model (Q-table, DQN, etc) from workspace/models/ or artifacts/models/
and uses it to make buy/sell decisions based on market state.

Since most RL models output discrete actions (buy/sell/hold), this maps:
  action=1 → enter_long
  action=0 → hold
  action=-1 or 2 → exit_long
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pandas import DataFrame

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]
for p in [str(_ROOT / "src"), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from freqtrade.strategy import IStrategy


class FreqtradeRLStrategy(IStrategy):
    """RL-based trading strategy for freqtrade."""

    timeframe = "1h"
    can_short = False
    minimal_roi = {"0": 0.10, "240": 0.03}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.025
    use_exit_signal = True
    startup_candle_count = 60
    process_only_new_candles = True

    _model = None
    _features: Optional[List[str]] = None
    _feature_cfg = None
    _expression_specs = None
    _loaded = False

    def _load(self):
        if self._loaded:
            return

        models_root = _ROOT / "artifacts" / "models"
        model_path = None
        summary = None

        # Find RL model (look for rl_ prefixed dirs or any model)
        for d in sorted(models_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            summary_f = d / "training_summary.json"
            if not summary_f.exists():
                continue
            s = json.loads(summary_f.read_text())
            mp = Path(s.get("model_path", ""))
            if not mp.is_absolute():
                mp = _ROOT / mp
            if mp.exists():
                model_path = mp
                summary = s
                break

        if model_path is None or summary is None:
            raise FileNotFoundError("No RL model found")

        self._features = [str(c) for c in (summary.get("features") or []) if str(c).strip()]

        for candidate in [_ROOT / "user_data" / "freqai_features_real.json"]:
            if candidate.exists():
                self._feature_cfg = json.loads(candidate.read_text(encoding="utf-8-sig"))
                break

        expr_file = summary.get("expressions_snapshot") or summary.get("expressions_file")
        if expr_file:
            ep = Path(expr_file) if Path(expr_file).is_absolute() else _ROOT / expr_file
            if ep.exists():
                from agent_market.freqai.expression_engine import load_expression_file
                self._expression_specs = load_expression_file(ep)

        # Load model
        suffix = model_path.suffix.lower()
        if suffix == ".pkl":
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
        elif suffix in (".pt", ".pth"):
            import torch
            self._model = torch.load(model_path, map_location="cpu", weights_only=False)
        elif suffix == ".zip":
            # Stable-Baselines3 model
            try:
                from stable_baselines3 import PPO
                self._model = PPO.load(str(model_path))
            except ImportError:
                raise ImportError("stable_baselines3 required for .zip RL models")
        else:
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)

        self._loaded = True

    def _predict_action(self, X: np.ndarray) -> np.ndarray:
        """Predict RL action for each row. Returns array of floats."""
        if hasattr(self._model, "predict"):
            # Sklearn-like or custom model with predict()
            preds = self._model.predict(X)
            return np.array(preds, dtype=float).reshape(-1)
        elif isinstance(self._model, dict):
            # Q-table or weight-based model
            if "weights" in self._model:
                w = self._model["weights"]
                b = self._model.get("bias", 0)
                mean = self._model.get("mean", np.zeros(X.shape[1]))
                std = self._model.get("std", np.ones(X.shape[1]))
                X_norm = (X - mean) / (std + 1e-8)
                return (X_norm @ w + b).reshape(-1)
        elif hasattr(self._model, "forward"):
            import torch
            with torch.no_grad():
                t = torch.from_numpy(X.astype(np.float32))
                return self._model(t).cpu().numpy().reshape(-1)

        # Fallback: treat as regression model output
        return np.zeros(len(X))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self._load()
        if self._feature_cfg:
            from agent_market.freqai.features import apply_configured_features
            dataframe = apply_configured_features(dataframe, self._feature_cfg)
        if self._expression_specs:
            from agent_market.freqai.expression_engine import apply_expressions
            dataframe, _ = apply_expressions(dataframe, self._expression_specs, on_error="raise")
        for col in self._features:
            if col not in dataframe.columns:
                dataframe[col] = 0.0
        matrix = dataframe[self._features].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        dataframe["rl_pred"] = self._predict_action(matrix.to_numpy(dtype=np.float32))
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RL pred > 0 means "buy signal" (regression output) or action=1
        dataframe.loc[(dataframe["volume"] > 0) & (dataframe["rl_pred"] > 0.003),
                      ["enter_long", "enter_tag"]] = (1, "rl_long")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(dataframe["volume"] > 0) & (dataframe["rl_pred"] < 0.0),
                      "exit_long"] = 1
        return dataframe
