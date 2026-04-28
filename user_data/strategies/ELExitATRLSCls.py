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


class ELExitATRLSCls(IStrategy):
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
    can_short = True

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

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        import os

        raw = os.environ.get(name)
        if raw in (None, ""):
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    @staticmethod
    def _env_bool(name: str, default: bool = True) -> bool:
        import os

        raw = os.environ.get(name)
        if raw in (None, ""):
            return bool(default)
        return str(raw).strip().lower() not in {"0", "false", "no", "off"}

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

    def _add_ohlcv_micro_features(self, df: DataFrame) -> DataFrame:
        """Compute OHLCV-derived micro columns used by FactorLab expressions."""
        import pandas as pd

        close = df["close"].astype("float64")
        open_ = df["open"].astype("float64")
        high = df["high"].astype("float64")
        low = df["low"].astype("float64")
        volume = df["volume"].astype("float64")

        eps = 1e-12
        ret_1 = close.pct_change()
        logret_1 = np.log(close + eps).diff()

        cols = {
            "ret_1": ret_1,
            "logret_1": logret_1,
            "range_pct": (high - low) / (close + eps),
            "body_pct": (close - open_) / (open_ + eps),
            "wick_up_pct": (high - np.maximum(open_, close)) / (close + eps),
            "wick_down_pct": (np.minimum(open_, close) - low) / (close + eps),
        }
        for w in (12, 24, 72):
            cols[f"rv_{w}"] = logret_1.rolling(w).std(ddof=0)
            cols[f"vol_z_{w}"] = (
                volume - volume.rolling(w).mean()
            ) / (volume.rolling(w).std(ddof=0) + eps)
            cols[f"amihud_{w}"] = (ret_1.abs() / (volume + eps)).rolling(w).mean()

        drop = [c for c in cols if c in df.columns]
        if drop:
            df = df.drop(columns=drop)
        out = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        return out.replace([np.inf, -np.inf], np.nan)

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
        """Merge cross-sectional, pair-relative, funding, and micro features from training feather.

        Rationale: training pipeline reads feather directly (all cols), but freqtrade
        backtest only sees OHLCV. Re-load the training feather for this pair and
        merge engineered columns via merge_asof.
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
        extra_cols = [
            c for c in ref.columns
            if c.startswith("xs_")
            or c.startswith("pair_")
            or c.startswith("funding_")
            or c.startswith("micro_")
        ]
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

    def _add_btc_regime_features(self, df: DataFrame, pair: str) -> DataFrame:
        """Add BTC trend columns for optional market-regime entry filtering."""
        import pandas as pd

        pair_clean = pair.split(":")[0] if pair else ""
        if pair_clean == "BTC/USDT":
            btc = df[["date", "close"]].copy()
        else:
            path = (
                _paths().user_data_root()
                / "data"
                / "okx"
                / "futures"
                / "BTC_USDT_USDT-1h-futures.feather"
            )
            if not path.exists():
                return df
            try:
                btc = pd.read_feather(path, columns=["date", "close"])
            except Exception:
                return df
            btc["date"] = pd.to_datetime(btc["date"], utc=True)
        btc = btc.sort_values("date").reset_index(drop=True)
        btc["btc_ema50"] = btc["close"].ewm(span=50, adjust=False).mean()
        btc["btc_ema200"] = btc["close"].ewm(span=200, adjust=False).mean()
        btc = btc[["date", "btc_ema50", "btc_ema200"]]
        return pd.merge_asof(df.sort_values("date"), btc, on="date", direction="backward")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        from agent_market.freqai.features import apply_configured_features
        feature_cfg = self._load_feature_cfg()
        df = apply_configured_features(dataframe, feature_cfg)
        df = self._add_ohlcv_micro_features(df)

        # Merge multi-timeframe 4h features (required by v4 expressions)
        pair = metadata.get("pair") if isinstance(metadata, dict) else None
        if pair:
            df = self._add_mtf4h_features(df, pair)
            df = self._add_xs_and_funding(df, pair)
            df = self._add_btc_regime_features(df, pair)

        model, cols = self._load_model()
        df = self._apply_expressions(df)
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
            exit_c = self._env_float("AGENT_EXIT_CONF", float(self.exit_conf))
            base_stake = self._env_float("AGENT_BASE_STAKE", float(proposed_stake))
            mult = 0.5 + 2.0 * (conf - exit_c)
            mult = max(0.5, min(2.0, mult))
            out = base_stake * mult
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

    def leverage(
        self,
        pair: str,
        current_time: Any,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs: Any,
    ) -> float:
        lev = self._env_float("AGENT_LEVERAGE", float(proposed_leverage or 1.0))
        return max(1.0, min(float(lev), float(max_leverage)))

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        uptrend = df["ema50"] > df["ema200"]
        downtrend = df["ema50"] < df["ema200"]
        conf = self._env_float("AGENT_ENTER_CONF", float(self.enter_conf))
        use_trend_filter = self._env_bool("AGENT_TREND_FILTER", True)
        use_btc_trend_filter = self._env_bool("AGENT_BTC_TREND_FILTER", False)
        min_prob_gap = self._env_float("AGENT_MIN_PROB_GAP", 0.0)
        side_mode = __import__("os").environ.get("AGENT_SIDE_MODE", "both").strip().lower()

        long_cond = (df["volume"] > 0) & (df["p_up"] > conf)
        short_cond = (df["volume"] > 0) & (df["p_down"] > conf)
        if min_prob_gap > 0:
            long_cond = long_cond & ((df["p_up"] - df["p_down"]) > min_prob_gap)
            short_cond = short_cond & ((df["p_down"] - df["p_up"]) > min_prob_gap)
        if use_trend_filter:
            long_cond = long_cond & uptrend
            short_cond = short_cond & downtrend
        if use_btc_trend_filter and {"btc_ema50", "btc_ema200"}.issubset(df.columns):
            btc_uptrend = df["btc_ema50"] > df["btc_ema200"]
            long_cond = long_cond & btc_uptrend
            short_cond = short_cond & ~btc_uptrend
        if side_mode == "long":
            short_cond = short_cond & False
        elif side_mode == "short":
            long_cond = long_cond & False

        df.loc[long_cond, ["enter_long", "enter_tag"]] = (1, "cls_long")
        df.loc[short_cond, ["enter_short", "enter_tag"]] = (1, "cls_short")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_c = self._env_float("AGENT_EXIT_CONF", float(self.exit_conf))
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
