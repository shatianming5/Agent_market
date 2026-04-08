"""Freqtrade DL Strategy — loads a trained PyTorch model for predictions.

Uses the same feature pipeline as FreqtradeMLStrategy but specifically
loads PyTorch .pt models (MLP, LSTM, GRU, etc).
"""
from __future__ import annotations

import json
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


def _resolve_summary_artifact(
    raw_path: Optional[str],
    *,
    model_dir: Path,
    fallback_names: Optional[List[str]] = None,
) -> Optional[Path]:
    """Resolve copied training artifacts when summaries contain stale absolute paths."""
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


def _iter_model_candidates(directory: Path, *suffixes: str) -> List[Path]:
    items: List[Path] = []
    for suffix in suffixes:
        items.extend(
            path for path in directory.glob(f"*{suffix}")
            if path.is_file() and not path.name.startswith("._")
        )
    return sorted(items)


class FreqtradeDLStrategy(IStrategy):
    """PyTorch Deep Learning strategy for freqtrade."""

    timeframe = "1h"
    can_short = False
    minimal_roi = {"0": 0.08, "240": 0.03}
    stoploss = -0.04
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    use_exit_signal = True
    startup_candle_count = 60
    process_only_new_candles = True

    dl_enter_threshold = 0.002
    dl_exit_threshold = 0.0

    _model = None
    _features: Optional[List[str]] = None
    _feature_cfg = None
    _expression_specs = None
    _loaded = False

    def _load(self):
        if self._loaded:
            return

        # Find PyTorch model (look for .pt or .pkl in model dirs)
        models_root = _ROOT / "artifacts" / "models"
        model_path = None
        summary = None
        model_dir = None

        for d in sorted(models_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            # Look for pytorch files
            for ext in [".pt", ".pth"]:
                candidates = _iter_model_candidates(d, ext)
                if candidates:
                    model_path = candidates[0]
                    model_dir = d
                    summary_f = d / "training_summary.json"
                    if summary_f.exists():
                        summary = json.loads(summary_f.read_text())
                    break
            if model_path:
                break

        # If no .pt found, fall back to any model with training_summary
        if model_path is None:
            for d in sorted(models_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if (d / "training_summary.json").exists():
                    summary = json.loads((d / "training_summary.json").read_text())
                    model_dir = d
                    mp = _resolve_summary_artifact(
                        summary.get("model_path"),
                        model_dir=d,
                        fallback_names=["pytorch_mlp.pt", "model.pt", "model.pth", "model.pkl"],
                    )
                    if mp is None:
                        continue
                    if mp.exists():
                        model_path = mp
                        break

        if model_path is None or summary is None or model_dir is None:
            raise FileNotFoundError("No DL model found in artifacts/models/")

        self._features = [str(c) for c in (summary.get("features") or []) if str(c).strip()]

        # Load features config
        for candidate in [
            _resolve_summary_artifact(
                summary.get("feature_snapshot") or summary.get("feature_file"),
                model_dir=model_dir,
                fallback_names=["feature_snapshot.json"],
            ),
            _ROOT / "user_data" / "freqai_features_real.json",
            _ROOT / "user_data" / "freqai_features.json",
        ]:
            if candidate is None:
                continue
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

        # Load model
        suffix = model_path.suffix.lower()
        if suffix in (".pt", ".pth"):
            import torch
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                from agent_market.freqai.model.torch_models import FeedForwardNet
                cfg = checkpoint.get("config", {})
                net = FeedForwardNet(
                    checkpoint.get("input_dim", len(self._features)),
                    cfg.get("hidden_dims", [64, 32]),
                    cfg.get("dropout", 0.0),
                )
                net.load_state_dict(checkpoint["state_dict"])
                net.eval()
                self._model = net
            else:
                self._model = checkpoint
        else:
            # Pickle fallback
            import pickle
            with open(model_path, "rb") as f:
                obj = pickle.load(f)
            if hasattr(obj, "predict"):
                self._model = obj
            elif isinstance(obj, dict) and "weights" in obj:
                # Ridge-like model
                self._model = obj
            else:
                raise ValueError(f"Cannot load model from {model_path}")

        self._loaded = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self._model, "forward"):
            import torch
            with torch.no_grad():
                t = torch.from_numpy(X.astype(np.float32))
                return self._model(t).cpu().numpy().reshape(-1)
        elif hasattr(self._model, "predict"):
            return np.array(self._model.predict(X)).reshape(-1)
        elif isinstance(self._model, dict) and "weights" in self._model:
            w = self._model["weights"]
            b = self._model.get("bias", 0)
            mean = self._model.get("mean", np.zeros(X.shape[1]))
            std = self._model.get("std", np.ones(X.shape[1]))
            X_norm = (X - mean) / (std + 1e-8)
            return (X_norm @ w + b).reshape(-1)
        else:
            raise RuntimeError(f"Unknown model type: {type(self._model)}")

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
        dataframe["dl_pred"] = self._predict(matrix.to_numpy(dtype=np.float32))
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(dataframe["volume"] > 0) & (dataframe["dl_pred"] > self.dl_enter_threshold),
                      ["enter_long", "enter_tag"]] = (1, "dl_long")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(dataframe["volume"] > 0) & (dataframe["dl_pred"] < self.dl_exit_threshold),
                      "exit_long"] = 1
        return dataframe
