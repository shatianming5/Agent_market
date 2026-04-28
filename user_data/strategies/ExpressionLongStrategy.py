from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy


def _inject_project_paths() -> Path:
    here = Path(__file__).resolve()
    root = None
    for parent in here.parents:
        if (parent / "src" / "agent_market").exists():
            root = parent
            break
    if root is None:
        root = here.parents[2]
    src = root / "src"
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(root))
    return root


PROJECT_ROOT = _inject_project_paths()


def _paths():
    from agent_market import paths as _am_paths

    return _am_paths


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_under_root(path: str) -> Path:
    return _paths().resolve_repo_path(path)


class ExpressionLongStrategy(IStrategy):
    """
    Minimal strategy for step-by-step debugging:
    - Reuses this repo's `freqai_features_real.json` engineered features.
    - Loads the LightGBM model trained by `scripts/agent_flow.py --steps ml`.
    """

    timeframe = "1h"
    minimal_roi = {"0": 0.1, "240": -1}
    stoploss = -0.05
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 60
    can_short = False

    ml_enter_threshold = 0.003
    ml_exit_threshold = 0.0
    rl_long_prob_threshold = 0.55
    rl_short_prob_exit_threshold = 0.55

    _feature_cfg: Optional[Dict[str, Any]] = None
    _model: Any = None
    _model_features: Optional[List[str]] = None
    _rl_signals: Dict[str, DataFrame] = {}
    _training_summary: Optional[Dict[str, Any]] = None
    _expressions_file: Optional[Path] = None
    _expression_specs: Optional[List[Any]] = None

    @staticmethod
    def _is_lightgbm_summary(summary_path: Path) -> bool:
        try:
            payload = _read_json(summary_path)
        except Exception:
            return False
        model_name = str(payload.get("model") or "").strip().lower()
        model_path = str(payload.get("model_path") or "").strip().lower()
        if model_name and model_name != "lightgbm":
            return False
        return model_path.endswith(".txt") or "lightgbm" in model_path

    def _resolve_model_dir(self) -> Path:
        """Resolve model directory: env MODEL_DIR > latest LightGBM summary > lightgbm_real."""
        import os
        env_dir = os.environ.get("AGENT_MODEL_DIR")
        if env_dir:
            p = _resolve_under_root(env_dir)
            summary_path = p / "training_summary.json"
            if summary_path.exists() and self._is_lightgbm_summary(summary_path):
                return p

        # Pick the most recently modified LightGBM model dir under the active models root.
        models_root = _paths().models_root()
        if models_root.exists():
            candidates = [
                d for d in models_root.iterdir()
                if d.is_dir()
                and (d / "training_summary.json").exists()
                and self._is_lightgbm_summary(d / "training_summary.json")
            ]
            if candidates:
                return max(candidates, key=lambda d: (d / "training_summary.json").stat().st_mtime)

        return _paths().models_root() / "lightgbm_real"

    def _load_training_summary(self) -> Optional[Dict[str, Any]]:
        if self._training_summary is not None:
            return self._training_summary
        summary_path = self._resolve_model_dir() / "training_summary.json"
        if not summary_path.exists():
            return None
        try:
            self._training_summary = _read_json(summary_path)
        except Exception:
            return None
        return self._training_summary

    def _load_feature_cfg(self) -> Dict[str, Any]:
        if self._feature_cfg is not None:
            return self._feature_cfg

        summary = self._load_training_summary() or {}
        snapshot = summary.get("feature_snapshot") or summary.get("feature_file")
        path: Path
        if snapshot:
            candidate = _resolve_under_root(str(snapshot))
            if candidate.exists():
                path = candidate
            else:
                path = _paths().user_data_root() / "freqai_features_real.json"
        else:
            path = _paths().user_data_root() / "freqai_features_real.json"
        if not path.exists():
            path = _paths().user_data_root() / "freqai_features.json"
        self._feature_cfg = _read_json(path)
        return self._feature_cfg

    def _load_model(self) -> Tuple[Any, List[str]]:
        if self._model is not None and self._model_features is not None:
            return self._model, self._model_features

        summary_path = self._resolve_model_dir() / "training_summary.json"
        summary = self._load_training_summary()
        if summary is None:
            summary = _read_json(summary_path)
        model_path = _resolve_under_root(str(summary.get("model_path") or ""))
        features = [str(col) for col in (summary.get("features") or []) if str(col).strip()]
        expr_file = summary.get("expressions_snapshot") or summary.get("expressions_file")
        if expr_file:
            self._expressions_file = _resolve_under_root(str(expr_file))
        if not model_path.exists():
            raise FileNotFoundError(f"LightGBM model not found: {model_path}")
        if not features:
            raise ValueError(f"Model feature list missing in {summary_path}")

        import lightgbm as lgb  # type: ignore

        self._model = lgb.Booster(model_file=str(model_path))
        self._model_features = features
        return self._model, features

    def _apply_expressions_if_needed(self, df: DataFrame) -> DataFrame:
        if self._expressions_file is None:
            return df
        if not self._expressions_file.exists():
            return df
        if self._expression_specs is None:
            from agent_market.freqai.expression_engine import load_expression_file  # noqa: WPS433

            self._expression_specs = load_expression_file(self._expressions_file)
        if not self._expression_specs:
            return df
        from agent_market.freqai.expression_engine import apply_expressions  # noqa: WPS433

        df, _cols = apply_expressions(df, self._expression_specs, on_error="raise")
        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        from agent_market.freqai.features import apply_configured_features  # noqa: WPS433

        feature_cfg = self._load_feature_cfg()
        df = apply_configured_features(dataframe, feature_cfg)

        model, cols = self._load_model()
        df = self._apply_expressions_if_needed(df)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {', '.join(missing[:10])}")

        matrix = (
            df[cols]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0.0)
        )
        df["ml_pred"] = model.predict(matrix.to_numpy(dtype=np.float32))

        try:
            pair = metadata.get("pair") if isinstance(metadata, dict) else None
            exchange = str(feature_cfg.get("exchange") or "unknown")
            if pair:
                sig = self._rl_signals.get(pair)
                if sig is None:
                    sanitized = str(pair).replace("/", "_")
                    sig_path = (
                        _paths().artifacts_root()
                        / "signals"
                        / "rl_real"
                        / exchange
                        / f"{sanitized}-{self.timeframe}.feather"
                    )
                    if sig_path.exists():
                        import pandas as pd  # noqa: WPS433

                        sig = pd.read_feather(sig_path)
                        sig["date"] = pd.to_datetime(sig["date"], utc=True)
                        self._rl_signals[pair] = sig
                if sig is not None:
                    df = df.merge(sig, on="date", how="left")
        except Exception:
            # If RL signals are missing, strategy falls back to ML-only gating.
            pass
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        cond = (df["volume"] > 0) & (df["ml_pred"] > float(self.ml_enter_threshold))
        if "rl_action" in df.columns:
            cond &= df["rl_action"] == 1
        elif "rl_long_prob" in df.columns:
            cond &= df["rl_long_prob"] > float(self.rl_long_prob_threshold)
        df.loc[cond, ["enter_long", "enter_tag"]] = (1, "ml_rl_long")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        cond = (df["volume"] > 0) & (df["ml_pred"] < float(self.ml_exit_threshold))
        if "rl_action" in df.columns:
            cond |= (df["volume"] > 0) & (df["rl_action"] != 1)
        elif "rl_short_prob" in df.columns:
            cond |= (df["volume"] > 0) & (
                df["rl_short_prob"] > float(self.rl_short_prob_exit_threshold)
            )
        df.loc[cond, "exit_long"] = 1
        return df
