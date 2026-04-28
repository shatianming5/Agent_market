from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy, stoploss_from_absolute


def _inject_project_paths() -> Path:
    here = Path(__file__).resolve()
    root = None
    for parent in here.parents:
        if (parent / "src" / "agent_market").exists():
            root = parent
            break
    if root is None:
        root = here.parents[2]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    return root


PROJECT_ROOT = _inject_project_paths()


def _paths():
    from agent_market import paths as _p
    return _p


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve(path: str) -> Path:
    return _paths().resolve_repo_path(path)


class ELExitATRLSClsLong(IStrategy):
    """Long + Short strategy using a 3-way classifier (down/flat/up).

    Entry long : p_up   > enter_conf  AND  EMA50 > EMA200
    Entry short: p_down > enter_conf  AND  EMA50 < EMA200
    Exit  long : p_up   < exit_conf
    Exit  short: p_down < exit_conf
    """

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

    _feature_cfg: Optional[Dict[str, Any]] = None
    _model: Any = None
    _model_features: Optional[List[str]] = None
    _training_summary: Optional[Dict[str, Any]] = None
    _expressions_file: Optional[Path] = None
    _expression_specs: Optional[List[Any]] = None
    _atr_cache: Dict[str, float] = {}
    # --- optional short-horizon exit model (M1) ---
    _exit_model: Any = None
    _exit_model_features: Optional[List[str]] = None
    # --- optional PCA transformer ---
    _pca_data: Optional[Dict[str, Any]] = None
    _pca_loaded: bool = False

    @staticmethod
    def _is_lightgbm_summary(p: Path) -> bool:
        try:
            d = _read_json(p)
        except Exception:
            return False
        name = str(d.get("model") or "").lower()
        mpath = str(d.get("model_path") or "").lower()
        if name and name != "lightgbm":
            return False
        return mpath.endswith(".txt") or "lightgbm" in mpath

    def _resolve_model_dir(self) -> Path:
        import os
        env_dir = os.environ.get("AGENT_MODEL_DIR")
        if env_dir:
            p = _resolve(env_dir)
            sp = p / "training_summary.json"
            # Honor AGENT_MODEL_DIR for any model type with a valid summary.
            # Supported: lightgbm, xgboost, stacked (and any future adapter).
            if sp.exists():
                return p
        models_root = _paths().models_root()
        if models_root.exists():
            candidates = [
                d for d in models_root.iterdir()
                if d.is_dir() and (d / "training_summary.json").exists()
                and self._is_lightgbm_summary(d / "training_summary.json")
            ]
            if candidates:
                return max(candidates, key=lambda d: (d / "training_summary.json").stat().st_mtime)
        return _paths().models_root() / "lightgbm_real"

    def _load_feature_cfg(self) -> Dict[str, Any]:
        if self._feature_cfg is not None:
            return self._feature_cfg
        summary = self._load_training_summary() or {}
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

    def _load_training_summary(self) -> Optional[Dict[str, Any]]:
        if self._training_summary is not None:
            return self._training_summary
        sp = self._resolve_model_dir() / "training_summary.json"
        if not sp.exists():
            return None
        try:
            self._training_summary = _read_json(sp)
        except Exception:
            return None
        return self._training_summary

    def _load_model(self) -> Tuple[Any, List[str]]:
        if self._model is not None:
            return self._model, self._model_features  # type: ignore[return-value]
        summary = self._load_training_summary() or _read_json(
            self._resolve_model_dir() / "training_summary.json"
        )
        model_path = _resolve(str(summary.get("model_path") or ""))
        features = [str(c) for c in (summary.get("features") or []) if str(c).strip()]
        expr_file = summary.get("expressions_snapshot") or summary.get("expressions_file")
        if expr_file:
            self._expressions_file = _resolve(str(expr_file))
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self._model = self._instantiate_model(str(summary.get("model") or "lightgbm").lower(), model_path)
        self._model_features = features
        return self._model, features

    def _load_pca(self) -> Optional[Dict[str, Any]]:
        """Load PCA transformer from model dir if pca.pkl exists. Returns None if absent."""
        if self._pca_loaded:
            return self._pca_data
        self._pca_loaded = True
        pca_path = self._resolve_model_dir() / 'pca.pkl'
        if not pca_path.exists():
            self._pca_data = None
            return None
        try:
            import pickle
            with open(pca_path, 'rb') as f:
                self._pca_data = pickle.load(f)
        except Exception:
            self._pca_data = None
        return self._pca_data

    def _load_exit_model(self) -> Optional[Tuple[Any, List[str]]]:
        """Load optional exit model from AGENT_EXIT_MODEL_DIR. Returns None if not set."""
        import os
        if self._exit_model is not None:
            return self._exit_model, self._exit_model_features or []
        exit_env = os.environ.get("AGENT_EXIT_MODEL_DIR")
        if not exit_env:
            return None
        exit_dir = _resolve(exit_env)
        sp = exit_dir / "training_summary.json"
        if not sp.exists():
            return None
        summary = _read_json(sp)
        mpath = _resolve(str(summary.get("model_path") or ""))
        if not mpath.exists():
            return None
        features = [str(c) for c in (summary.get("features") or []) if str(c).strip()]
        try:
            self._exit_model = self._instantiate_model(str(summary.get("model") or "lightgbm").lower(), mpath)
            self._exit_model_features = features
            return self._exit_model, features
        except Exception:
            return None

    @staticmethod
    def _instantiate_model(model_name: str, model_path: Path):
        """Load a trained model file and wrap to unified .predict(X)→(N, K) API."""
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

    def _apply_expressions(self, df: DataFrame) -> DataFrame:
        if self._expressions_file is None or not self._expressions_file.exists():
            return df
        if self._expression_specs is None:
            from agent_market.freqai.expression_engine import load_expression_file
            self._expression_specs = load_expression_file(self._expressions_file)
        if not self._expression_specs:
            return df
        from agent_market.freqai.expression_engine import apply_expressions
        df, _ = apply_expressions(df, self._expression_specs, on_error="raise")
        return df

    def _add_mtf4h_features(self, df: DataFrame, pair: str) -> DataFrame:
        """Causally merge 4h-derived features (mtf4h_*) onto 1h dataframe."""
        import pandas as pd
        from agent_market.freqai.features import apply_configured_features
        # Strip :USDT suffix (futures notation) to match spot KuCoin data naming
        pair_clean = pair.split(":")[0] if pair else ""
        pair_sanitized = pair_clean.replace("/", "_") if pair_clean else ""
        path_4h = _paths().user_data_root() / "data" / "kucoin" / f"{pair_sanitized}-4h.feather"
        if not path_4h.exists():
            import sys as _sys
            print(f"[MTF] 4h file not found: {path_4h}", file=_sys.stderr)
            return df
        try:
            df_4h = pd.read_feather(path_4h)
        except Exception:
            return df
        df_4h["date"] = pd.to_datetime(df_4h["date"], utc=True)
        df_4h = df_4h.sort_values("date").reset_index(drop=True)
        # Reduced feature set matching scripts/merge_mtf_features.py
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
        """Merge cross-sectional ranks + funding rate features from training feather.

        Rationale: training pipeline reads feather directly (all cols), but freqtrade
        backtest only sees OHLCV. Re-load the training feather for this pair and
        merge xs_* / funding_* columns via merge_asof.
        """
        import pandas as pd
        pair_clean = pair.split(":")[0] if pair else ""
        pair_sanitized = pair_clean.replace("/", "_") if pair_clean else ""
        path = _paths().user_data_root() / "data" / "kucoin" / f"{pair_sanitized}-1h.feather"
        if not path.exists(): return df
        try:
            ref = pd.read_feather(path)
        except Exception:
            return df
        ref["date"] = pd.to_datetime(ref["date"], utc=True)
        extra_cols = [c for c in ref.columns if c.startswith("xs_") or c.startswith("funding_") or c.startswith("micro_")]
        if not extra_cols: return df
        ref_small = ref[["date"] + extra_cols].sort_values("date").reset_index(drop=True)
        df = df.sort_values("date").reset_index(drop=True)
        # Drop existing if re-run
        drop = [c for c in extra_cols if c in df.columns]
        if drop: df = df.drop(columns=drop)
        merged = pd.merge_asof(df, ref_small, on="date", direction="backward")
        for c in extra_cols:
            merged[c] = merged[c].fillna(0)
        return merged

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        from agent_market.freqai.features import apply_configured_features
        feature_cfg = self._load_feature_cfg()
        df = apply_configured_features(dataframe, feature_cfg)

        # Merge multi-timeframe 4h features (required by v4 expressions)
        pair = metadata.get("pair") if isinstance(metadata, dict) else None
        if pair:
            df = self._add_mtf4h_features(df, pair)
            df = self._add_xs_and_funding(df, pair)

        model, cols = self._load_model()
        df = self._apply_expressions(df)
        pca_data = self._load_pca()
        if pca_data is not None:
            _expr_cols = pca_data['expr_cols']
            _pca = pca_data['pca']
            _pca_names = pca_data['pca_col_names']
            if all(c in df.columns for c in _expr_cols):
                _expr_mat = (
                    df[_expr_cols].astype(float)
                    .replace([np.inf, -np.inf], np.nan)
                    .ffill().fillna(0.0)
                    .to_numpy()
                )
                _pca_result = _pca.transform(_expr_mat)
                for _j, _pname in enumerate(_pca_names):
                    df[_pname] = _pca_result[:, _j]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing[:5]}")
        mat = (
            df[cols].astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .ffill().fillna(0.0)
        )
        probs = model.predict(mat.to_numpy(dtype=np.float32))
        # probs shape: (N, 3) — columns are [down, flat, up]
        probs = np.asarray(probs)
        if probs.ndim != 2 or probs.shape[1] != 3:
            df["p_down"] = 0.0; df["p_flat"] = 1.0; df["p_up"] = 0.0
        else:
            df["p_down"] = probs[:, 0]
            df["p_flat"] = probs[:, 1]
            df["p_up"] = probs[:, 2]

        # M1: optional short-horizon exit model — reuses entry features, predicts
        # P(down in <exit_label_period> bars) which drives faster exits.
        exit_loaded = self._load_exit_model()
        if exit_loaded is not None:
            ex_model, ex_cols = exit_loaded
            missing_e = [c for c in ex_cols if c not in df.columns]
            if not missing_e:
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
        atr = tr.ewm(span=27, adjust=False).mean()
        df["atr"] = atr

        pair = metadata.get("pair") if isinstance(metadata, dict) else None
        if pair and len(atr) > 0 and not np.isnan(float(atr.iloc[-1])):
            self._atr_cache[pair] = float(atr.iloc[-1])

        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

        return df

    def custom_stake_amount(self, pair: str, current_time: Any, current_rate: float,
                             proposed_stake: float, min_stake: Optional[float],
                             max_stake: float, leverage: float, entry_tag: Optional[str],
                             side: str, **kwargs: Any) -> float:
        """M2: scale stake by model confidence.

        Lookup the last row of the pair's dataframe to read p_up / p_down at the
        entry timestamp. stake_multiplier = 0.5 + 2 * (max(p_up, p_down) - exit_conf)
        clamped to [0.5, 2.0] — so 50% conviction gets baseline, 90%+ gets 2x.
        """
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is None or df.empty:
                return proposed_stake
            last = df.iloc[-1]
            conf = float(max(float(last.get("p_up", 0.0)), float(last.get("p_down", 0.0))))
            exit_c = float(self.exit_conf)
            mult = 0.5 + 2.0 * (conf - exit_c)
            mult = max(0.5, min(2.0, mult))
            out = proposed_stake * mult
            if min_stake is not None:
                out = max(out, float(min_stake))
            out = min(out, float(max_stake))
            return out
        except Exception:
            return proposed_stake

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: Any,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float:
        if current_profit > float(self.profit_activation):
            atr = self._atr_cache.get(pair, 0.0)
            if atr > 0:
                is_short = getattr(trade, "is_short", False)
                if is_short:
                    trail_price = current_rate + float(self.atr_multiplier) * atr
                else:
                    trail_price = current_rate - float(self.atr_multiplier) * atr
                sl = stoploss_from_absolute(trail_price, current_rate, is_short=is_short)
                return max(sl, -0.02)
        return self.stoploss

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        uptrend = df["ema50"] > df["ema200"]
        downtrend = df["ema50"] < df["ema200"]
        conf = float(self.enter_conf)

        long_cond = (df["volume"] > 0) & (df["p_up"] > conf) & uptrend
        short_cond = (df["volume"] > 0) & (df["p_down"] > conf) & downtrend

        df.loc[long_cond, ["enter_long", "enter_tag"]] = (1, "cls_long")
        df.loc[short_cond, ["enter_short", "enter_tag"]] = (1, "cls_short")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_c = float(self.exit_conf)
        # Entry-model based exit (existing behavior)
        long_exit = (df["p_up"] < exit_c)
        short_exit = (df["p_down"] < exit_c)
        # M1: short-horizon exit model adds a second trigger if loaded
        if "p_exit_down" in df.columns and df["p_exit_down"].notna().any():
            long_exit = long_exit | (df["p_exit_down"] > 0.45)
            short_exit = short_exit | (df["p_exit_up"] > 0.45)
        df.loc[(df["volume"] > 0) & long_exit, "exit_long"] = 1
        df.loc[(df["volume"] > 0) & short_exit, "exit_short"] = 1
        return df
