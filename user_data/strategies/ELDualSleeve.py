"""ELDualSleeve — v7 (80%) + R5 XSRanker (20%) parallel sleeve strategy.

Env vars:
  AGENT_MODEL_DIR     — v7 entry model (primary, 80% stake weight)
  AGENT_R5_MODEL_DIR  — R5 sleeve model (secondary, 20% stake weight)
  AGENT_EXIT_MODEL_DIR — optional short-horizon exit model (v7 pipeline)

Entry priority:
  1. v7 signals entry  → enter_tag="v7_long", stake=80% of proposed
  2. v7 neutral AND R5 signals → enter_tag="r5_long", stake=20% of proposed
  3. Both neutral → no entry

Exit: same ATR-trailing stoploss as ELExitATRLSClsLong.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy, stoploss_from_absolute


def _inject_project_paths() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "agent_market").exists():
            return parent
    return here.parents[2]


PROJECT_ROOT = _inject_project_paths()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def _paths():
    from agent_market import paths as _p
    return _p


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve(path: str) -> Path:
    return _paths().resolve_repo_path(path)


# Stake fractions per source
_STAKE_FRAC = {"v7_long": 0.80, "r5_long": 0.20}


class ELDualSleeve(IStrategy):
    timeframe = "1h"
    minimal_roi = {"0": 0.20, "2880": -1}
    stoploss = -0.10
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 60
    can_short = False

    enter_conf: float = 0.55
    exit_conf: float = 0.45
    atr_multiplier: float = 1.5
    profit_activation: float = 0.12

    # --- v7 primary model ---
    _v7_model: Any = None
    _v7_features: Optional[List[str]] = None
    _v7_summary: Optional[Dict[str, Any]] = None
    _v7_expr_file: Optional[Path] = None
    _v7_expr_specs: Optional[List[Any]] = None

    # --- R5 sleeve model ---
    _r5_model: Any = None
    _r5_features: Optional[List[str]] = None
    _r5_expr_file: Optional[Path] = None
    _r5_expr_specs: Optional[List[Any]] = None

    # --- optional exit model ---
    _exit_model: Any = None
    _exit_model_features: Optional[List[str]] = None

    _feature_cfg: Optional[Dict[str, Any]] = None
    _atr_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _instantiate_model(model_name: str, model_path: Path):
        if model_name == "xgboost":
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(str(model_path))
            class _XGBWrap:
                def __init__(self, b): self.b = b
                def predict(self, X):
                    import xgboost as xgb
                    return self.b.predict(xgb.DMatrix(X))
            return _XGBWrap(booster)
        if model_name in ("stacked", "ridge_classifier"):
            import pickle
            with Path(model_path).open("rb") as f:
                clf = pickle.load(f)
            class _SKWrap:
                def __init__(self, m): self.m = m
                def predict(self, X):
                    if hasattr(self.m, "predict_proba"):
                        return self.m.predict_proba(X)
                    return self.m.predict(X)
            return _SKWrap(clf)
        import lightgbm as lgb
        return lgb.Booster(model_file=str(model_path))

    def _load_model_from_dir(self, model_dir: Path) -> Tuple[Any, List[str], Optional[Path]]:
        """Load model + features + expression file from a model directory."""
        sp = model_dir / "training_summary.json"
        summary = _read_json(sp)
        model_path = _resolve(str(summary.get("model_path") or ""))
        features = [str(c) for c in (summary.get("features") or []) if str(c).strip()]
        expr_file_str = summary.get("expressions_snapshot") or summary.get("expressions_file")
        expr_file = _resolve(str(expr_file_str)) if expr_file_str else None
        model = self._instantiate_model(
            str(summary.get("model") or "lightgbm").lower(), model_path
        )
        return model, features, expr_file

    def _load_v7(self) -> Tuple[Any, List[str]]:
        if self._v7_model is not None:
            return self._v7_model, self._v7_features  # type: ignore[return-value]
        env = os.environ.get("AGENT_MODEL_DIR")
        if env:
            model_dir = _resolve(env)
        else:
            models_root = _paths().models_root()
            candidates = [
                d for d in models_root.iterdir()
                if d.is_dir() and (d / "training_summary.json").exists()
            ] if models_root.exists() else []
            model_dir = max(candidates,
                            key=lambda d: (d / "training_summary.json").stat().st_mtime,
                            default=models_root / "lightgbm_real")
        model, features, expr_file = self._load_model_from_dir(model_dir)
        self._v7_model, self._v7_features, self._v7_expr_file = model, features, expr_file
        if self._v7_summary is None:
            self._v7_summary = _read_json(model_dir / "training_summary.json")
        return model, features

    def _load_r5(self) -> Optional[Tuple[Any, List[str]]]:
        if self._r5_model is not None:
            return self._r5_model, self._r5_features  # type: ignore[return-value]
        env = os.environ.get("AGENT_R5_MODEL_DIR")
        if not env:
            return None
        model_dir = _resolve(env)
        sp = model_dir / "training_summary.json"
        if not sp.exists():
            return None
        try:
            model, features, expr_file = self._load_model_from_dir(model_dir)
            self._r5_model, self._r5_features, self._r5_expr_file = model, features, expr_file
            return model, features
        except Exception as exc:
            print(f"[DualSleeve] R5 model load failed: {exc}", file=sys.stderr)
            return None

    def _load_exit_model(self) -> Optional[Tuple[Any, List[str]]]:
        if self._exit_model is not None:
            return self._exit_model, self._exit_model_features or []
        env = os.environ.get("AGENT_EXIT_MODEL_DIR")
        if not env:
            return None
        exit_dir = _resolve(env)
        sp = exit_dir / "training_summary.json"
        if not sp.exists():
            return None
        summary = _read_json(sp)
        mpath = _resolve(str(summary.get("model_path") or ""))
        if not mpath.exists():
            return None
        features = [str(c) for c in (summary.get("features") or []) if str(c).strip()]
        try:
            self._exit_model = self._instantiate_model(
                str(summary.get("model") or "lightgbm").lower(), mpath)
            self._exit_model_features = features
            return self._exit_model, features
        except Exception:
            return None

    def _load_feature_cfg(self) -> Dict[str, Any]:
        if self._feature_cfg is not None:
            return self._feature_cfg
        summary = self._v7_summary or {}
        snapshot = summary.get("feature_snapshot") or summary.get("feature_file")
        if snapshot:
            cand = _resolve(str(snapshot))
            path = cand if cand.exists() else _paths().user_data_root() / "freqai_features_real.json"
        else:
            path = _paths().user_data_root() / "freqai_features_real.json"
        if not path.exists():
            path = _paths().user_data_root() / "freqai_features.json"
        self._feature_cfg = _read_json(path)
        return self._feature_cfg

    # ------------------------------------------------------------------
    # Feature engineering helpers (same as ELExitATRLSClsLong)
    # ------------------------------------------------------------------
    def _apply_expressions(self, df: DataFrame, expr_file: Optional[Path],
                            expr_specs_attr: str) -> DataFrame:
        if expr_file is None or not expr_file.exists():
            return df
        specs = getattr(self, expr_specs_attr)
        if specs is None:
            from agent_market.freqai.expression_engine import load_expression_file
            specs = load_expression_file(expr_file)
            setattr(self, expr_specs_attr, specs)
        if not specs:
            return df
        from agent_market.freqai.expression_engine import apply_expressions
        df, _ = apply_expressions(df, specs, on_error="raise")
        return df

    def _add_mtf4h_features(self, df: DataFrame, pair: str) -> DataFrame:
        import pandas as pd
        from agent_market.freqai.features import apply_configured_features
        pair_sanitized = pair.split(":")[0].replace("/", "_")
        path_4h = _paths().user_data_root() / "data" / "kucoin" / f"{pair_sanitized}-4h.feather"
        if not path_4h.exists():
            return df
        try:
            df_4h = pd.read_feather(path_4h)
        except Exception:
            return df
        df_4h["date"] = pd.to_datetime(df_4h["date"], utc=True)
        df_4h = df_4h.sort_values("date").reset_index(drop=True)
        MTF_CFG = {"features": [
            {"name": "rsi_14", "type": "rsi", "period": 14},
            {"name": "adx_14", "type": "adx", "period": 14},
            {"name": "atr_norm_14", "type": "atr_norm", "period": 14},
            {"name": "ema_pct_12", "type": "ema_pct", "period": 12},
            {"name": "ema_pct_48", "type": "ema_pct", "period": 48},
            {"name": "cmf_20", "type": "cmf", "period": 20},
            {"name": "plus_di_14", "type": "plus_di", "period": 14},
            {"name": "minus_di_14", "type": "minus_di", "period": 14},
            {"name": "return_zscore_24", "type": "return_zscore", "period": 24},
            {"name": "realized_vol_24", "type": "realized_vol", "period": 24},
            {"name": "return_skew_48", "type": "return_skew", "period": 48},
            {"name": "donchian_width_48", "type": "donchian_width", "period": 48},
        ]}
        df_4h = apply_configured_features(df_4h, MTF_CFG)
        feat_cols = [f["name"] for f in MTF_CFG["features"]]
        df_4h["__close_time__"] = df_4h["date"] + pd.Timedelta(hours=4)
        rename = {c: f"mtf4h_{c}" for c in feat_cols}
        mtf_df = (df_4h[["__close_time__"] + feat_cols]
                  .rename(columns=rename)
                  .rename(columns={"__close_time__": "date"})
                  .sort_values("date"))
        df = df.sort_values("date").reset_index(drop=True)
        merged = pd.merge_asof(df, mtf_df, on="date", direction="backward")
        return merged

    def _add_xs_and_funding(self, df: DataFrame, pair: str) -> DataFrame:
        import pandas as pd
        pair_sanitized = pair.split(":")[0].replace("/", "_")
        path = _paths().user_data_root() / "data" / "kucoin" / f"{pair_sanitized}-1h.feather"
        if not path.exists():
            return df
        try:
            ref = pd.read_feather(path)
        except Exception:
            return df
        ref["date"] = pd.to_datetime(ref["date"], utc=True)
        extra_cols = [c for c in ref.columns if
                      c.startswith("xs_") or c.startswith("funding_") or c.startswith("micro_")]
        if not extra_cols:
            return df
        ref_small = ref[["date"] + extra_cols].sort_values("date").reset_index(drop=True)
        df = df.sort_values("date").reset_index(drop=True)
        drop = [c for c in extra_cols if c in df.columns]
        if drop:
            df = df.drop(columns=drop)
        merged = pd.merge_asof(df, ref_small, on="date", direction="backward")
        for c in extra_cols:
            merged[c] = merged[c].fillna(0)
        return merged

    @staticmethod
    def _predict_proba(model, features: List[str], df: DataFrame) -> Optional[np.ndarray]:
        missing = [c for c in features if c not in df.columns]
        if missing:
            return None
        mat = (df[features].astype(float)
               .replace([np.inf, -np.inf], np.nan)
               .ffill().fillna(0.0))
        probs = np.asarray(model.predict(mat.to_numpy(dtype=np.float32)))
        if probs.ndim != 2 or probs.shape[1] != 3:
            return None
        return probs

    # ------------------------------------------------------------------
    # Strategy hooks
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        from agent_market.freqai.features import apply_configured_features

        # Ensure v7 summary is loaded for feature_cfg resolution
        self._load_v7()
        feature_cfg = self._load_feature_cfg()
        df = apply_configured_features(dataframe, feature_cfg)

        pair = metadata.get("pair") if isinstance(metadata, dict) else None
        if pair:
            df = self._add_mtf4h_features(df, pair)
            df = self._add_xs_and_funding(df, pair)

        # --- v7 model predictions ---
        v7_model, v7_feats = self._load_v7()
        df = self._apply_expressions(df, self._v7_expr_file, "_v7_expr_specs")
        probs_v7 = self._predict_proba(v7_model, v7_feats, df)
        if probs_v7 is not None:
            df["p_down"] = probs_v7[:, 0]
            df["p_flat"] = probs_v7[:, 1]
            df["p_up"] = probs_v7[:, 2]
        else:
            df["p_down"] = 0.0; df["p_flat"] = 1.0; df["p_up"] = 0.0

        # --- R5 sleeve model predictions ---
        r5_loaded = self._load_r5()
        if r5_loaded is not None:
            r5_model, r5_feats = r5_loaded
            df = self._apply_expressions(df, self._r5_expr_file, "_r5_expr_specs")
            probs_r5 = self._predict_proba(r5_model, r5_feats, df)
            if probs_r5 is not None:
                df["p_r5_down"] = probs_r5[:, 0]
                df["p_r5_up"] = probs_r5[:, 2]
            else:
                df["p_r5_down"] = 0.0; df["p_r5_up"] = 0.0
        else:
            df["p_r5_down"] = 0.0; df["p_r5_up"] = 0.0

        # --- optional exit model ---
        exit_loaded = self._load_exit_model()
        if exit_loaded is not None:
            ex_model, ex_cols = exit_loaded
            ex_mat_ok = all(c in df.columns for c in ex_cols)
            if ex_mat_ok:
                ex_mat = (df[ex_cols].astype(float)
                          .replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0))
                ex_probs = np.asarray(ex_model.predict(ex_mat.to_numpy(dtype=np.float32)))
                if ex_probs.ndim == 2 and ex_probs.shape[1] == 3:
                    df["p_exit_down"] = ex_probs[:, 0]
                    df["p_exit_up"] = ex_probs[:, 2]
                else:
                    df["p_exit_down"] = 0.0; df["p_exit_up"] = 0.0
            else:
                df["p_exit_down"] = 0.0; df["p_exit_up"] = 0.0
        else:
            df["p_exit_down"] = np.nan; df["p_exit_up"] = np.nan

        h, l, c = df["high"], df["low"], df["close"]
        tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
        df["atr"] = tr.ewm(span=27, adjust=False).mean()
        if pair and len(df) > 0:
            atr_last = float(df["atr"].iloc[-1])
            if not np.isnan(atr_last):
                self._atr_cache[pair] = atr_last

        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
        return df

    def custom_stake_amount(self, pair: str, current_time: Any, current_rate: float,
                             proposed_stake: float, min_stake: Optional[float],
                             max_stake: float, leverage: float, entry_tag: Optional[str],
                             side: str, **kwargs: Any) -> float:
        frac = _STAKE_FRAC.get(entry_tag or "", 1.0)
        stake = proposed_stake * frac
        if min_stake is not None:
            stake = max(stake, float(min_stake))
        return min(stake, float(max_stake))

    def custom_stoploss(self, pair: str, trade: Any, current_time: Any,
                         current_rate: float, current_profit: float,
                         after_fill: bool, **kwargs: Any) -> float:
        if current_profit > float(self.profit_activation):
            atr = self._atr_cache.get(pair, 0.0)
            if atr > 0:
                trail_price = current_rate - float(self.atr_multiplier) * atr
                sl = stoploss_from_absolute(trail_price, current_rate, is_short=False)
                return max(sl, -0.02)
        return self.stoploss

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        conf = float(self.enter_conf)
        uptrend = df["ema50"] > df["ema200"]
        vol_ok = df["volume"] > 0

        # Primary: v7 long signal
        v7_long = vol_ok & uptrend & (df["p_up"] > conf)
        # Secondary: R5 only when v7 is neutral (p_up <= enter_conf)
        r5_long = vol_ok & uptrend & (df["p_r5_up"] > conf) & ~v7_long

        df.loc[v7_long, ["enter_long", "enter_tag"]] = (1, "v7_long")
        df.loc[r5_long, ["enter_long", "enter_tag"]] = (1, "r5_long")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_c = float(self.exit_conf)
        # Always use v7 primary model for exit — preserves its exit timing.
        # R5 is only used for additional entries, not for exit control.
        long_exit = df["p_up"] < exit_c
        if "p_exit_down" in df.columns and df["p_exit_down"].notna().any():
            long_exit = long_exit | (df["p_exit_down"] > 0.45)
        df.loc[(df["volume"] > 0) & long_exit, "exit_long"] = 1
        return df
