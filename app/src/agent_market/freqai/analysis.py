from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from .features import apply_configured_features


ALLOWED_FUNCS = {
    "z": lambda s: (_ensure_series(s) - _ensure_series(s).mean())
    / (_ensure_series(s).std(ddof=0) + 1e-9),
    "abs": np.abs,
    "sign": np.sign,
    "clip": np.clip,
    "shift": lambda s, n=1: _ensure_series(s).shift(int(n)),
    "roll_mean": lambda s, w=3: _ensure_series(s).rolling(int(w)).mean(),
    "roll_std": lambda s, w=3: _ensure_series(s).rolling(int(w)).std(ddof=0),
    "pct_change": lambda s, n=1: _ensure_series(s).pct_change(int(n)),
    "ema": lambda s, span=5: _ensure_series(s).ewm(span=int(span), adjust=False).mean(),
    "rolling_max": lambda s, w=5: _ensure_series(s).rolling(int(w)).max(),
    "rolling_min": lambda s, w=5: _ensure_series(s).rolling(int(w)).min(),
    "rolling": lambda s, w=5: _ensure_series(s).rolling(int(w)).mean(),
    "log1p": lambda s: np.log1p(_ensure_series(s)),
    "tanh": lambda s: np.tanh(_ensure_series(s)),
}


BASE_COLUMNS = {
    "date",
    "pair",
    "time",
    "open_time",
    "close_time",
}


@dataclass(slots=True)
class FactorIC:
    name: str
    ic_pearson: float
    ic_spearman: float
    samples: int
    mean: float
    std: float
    positive_ratio: Optional[float] = None
    ic_std: Optional[float] = None
    ic_ir: Optional[float] = None

    @property
    def coverage(self) -> float:
        return float(self.samples)


def _ensure_series(data) -> pd.Series:
    if isinstance(data, pd.Series):
        return data
    raise TypeError("Expressions must evaluate to pandas.Series")


def compute_information_coefficient(
    dataframe: pd.DataFrame,
    factors: Sequence[str],
    returns: pd.Series,
    window: int = 20,
) -> List[FactorIC]:
    results: List[FactorIC] = []
    for name in factors:
        if name not in dataframe.columns:
            continue
        feature = dataframe[name].astype(float)
        mask = feature.replace([np.inf, -np.inf], np.nan).notna() & returns.notna()
        if mask.sum() < 8:
            continue
        x = feature[mask]
        y = returns[mask]
        pearson = x.corr(y)
        spearman = x.rank().corr(y.rank())
        rolling_ic = (
            x.rolling(window).corr(y)
            .dropna()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(rolling_ic) == 0:
            ic_std = None
            positive_ratio = None
            ic_ir = None
        else:
            ic_std = float(rolling_ic.std(ddof=0))
            positive_ratio = float((rolling_ic > 0).mean())
            ic_ir = float(rolling_ic.mean() / ic_std) if ic_std and ic_std > 0 else None
        results.append(
            FactorIC(
                name=name,
                ic_pearson=float(pearson) if pearson is not None and math.isfinite(pearson) else 0.0,
                ic_spearman=float(spearman) if spearman is not None and math.isfinite(spearman) else 0.0,
                samples=int(mask.sum()),
                mean=float(x.mean()),
                std=float(x.std(ddof=0)),
                positive_ratio=positive_ratio,
                ic_std=ic_std,
                ic_ir=ic_ir,
            )
        )
    return results


def factor_correlation_matrix(dataframe: pd.DataFrame, factors: Sequence[str]) -> pd.DataFrame:
    matrix = (
        dataframe[list(factors)]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(axis=0, how="any")
    )
    if matrix.empty:
        return pd.DataFrame()
    return matrix.corr()


def factor_quantile_stats(
    signal: pd.Series,
    forward_returns: pd.Series,
    quantiles: int = 5,
) -> Dict[str, float]:
    data = pd.DataFrame({"signal": signal, "returns": forward_returns}).replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return {}
    data["bucket"] = pd.qcut(data["signal"], quantiles, labels=False, duplicates="drop")
    grouped = data.groupby("bucket")["returns"].mean()
    if grouped.empty:
        return {}
    top = grouped.iloc[-1]
    bottom = grouped.iloc[0]
    spread = float(top - bottom)
    t_stat = float(spread / (data["returns"].std(ddof=1) / np.sqrt(len(data)))) if len(data) > 1 else None
    return {
        "quantiles": int(grouped.size),
        "samples": int(len(data)),
        "top_mean": float(top),
        "bottom_mean": float(bottom),
        "spread": spread,
        "t_stat": t_stat if t_stat is not None and math.isfinite(t_stat) else None,
    }


def evaluate_expressions(
    dataframe: pd.DataFrame,
    expressions: Iterable[Dict],
) -> Dict[str, pd.Series]:
    local_dict: Dict[str, pd.Series] = {col: dataframe[col] for col in dataframe.columns}
    results: Dict[str, pd.Series] = {}
    for idx, expr in enumerate(expressions):
        expression = expr.get("expression")
        if not expression:
            continue
        name = expr.get("name") or f"expr_{idx}"
        try:
            value = eval(expression, {"__builtins__": {}}, {**local_dict, **ALLOWED_FUNCS})
        except Exception:
            continue
        series = _ensure_series(value).astype(float).replace([np.inf, -np.inf], np.nan)
        results[name] = series
        local_dict[name] = series
    return results


if TYPE_CHECKING:
    from agent_market.freqai.context import FeatureContext


def build_feature_dataframe(
    raw_dataframe: pd.DataFrame,
    feature_cfg: Dict,
    context: "FeatureContext | None" = None,
    include_corr_pairs: bool = False,
    base_pair: Optional[str] = None,
) -> pd.DataFrame:
    df = raw_dataframe.copy()
    df = apply_configured_features(df, feature_cfg)
    if include_corr_pairs and context and context.settings:
        cfg_pairs = feature_cfg.get("include_corr_pairlist")
        raw_params = context.settings.raw.get("freqai", {}).get("feature_parameters", {})
        corr_pairs = cfg_pairs or raw_params.get("include_corr_pairlist", [])
        primary_pair = base_pair
        if not primary_pair and context.settings.pairs:
            primary_pair = context.settings.pairs[0]
        df = _merge_corr_pair_features(
            df,
            feature_cfg,
            context.settings.data_dir,
            context.timeframe,
            corr_pairs,
            primary_pair,
        )
        if feature_cfg.get("feature_combos"):
            combo_cfg = {
                "features": [],
                "feature_combos": feature_cfg.get("feature_combos", []),
            }
            df = apply_configured_features(df, combo_cfg)
        lstm_dir_cfg = (
            feature_cfg.get("lstm_prediction_dir")
            or raw_params.get("lstm_prediction_dir")
        )
        if lstm_dir_cfg and primary_pair:
            df = _merge_lstm_predictions(
                df,
                Path(lstm_dir_cfg),
                primary_pair,
                context.timeframe,
            )
    return df


def _merge_corr_pair_features(
    base_df: pd.DataFrame,
    feature_cfg: Dict,
    data_dir: Path,
    timeframe: str,
    corr_pairs: Sequence[str],
    base_pair: Optional[str],
) -> pd.DataFrame:
    if not corr_pairs:
        return base_df

    base = base_df.sort_values("date").reset_index(drop=True)
    base["date"] = pd.to_datetime(base["date"])
    base_sanitized = base_pair.replace("/", "_") if base_pair else None

    for pair in corr_pairs:
        sanitized = pair.replace("/", "_")
        if base_sanitized and sanitized == base_sanitized:
            continue
        file_path = data_dir / f"{sanitized}-{timeframe}.feather"
        if not file_path.exists():
            continue
        df_other = pd.read_feather(file_path)
        if "date" not in df_other.columns:
            continue
        df_other["date"] = pd.to_datetime(df_other["date"])
        df_other = df_other.sort_values("date")
        df_other = apply_configured_features(df_other, feature_cfg)
        feature_cols = [col for col in df_other.columns if col not in BASE_COLUMNS]
        if not feature_cols:
            continue
        rename_map = {
            col: f"{col}_{sanitized}_{timeframe}"
            for col in feature_cols
        }
        subset = df_other[["date"] + feature_cols].rename(columns=rename_map)
        base = base.merge(subset, on="date", how="left")
    return base


def _merge_lstm_predictions(
    base_df: pd.DataFrame,
    prediction_dir: Path,
    base_pair: str,
    timeframe: str,
) -> pd.DataFrame:
    file_path = prediction_dir / f"{base_pair.replace('/', '_')}-{timeframe}.csv"
    if not file_path.exists():
        return base_df
    try:
        pred_df = pd.read_csv(file_path)
    except Exception:
        return base_df
    if "date" not in pred_df.columns:
        return base_df
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    pred_df = pred_df.sort_values("date").drop_duplicates(subset="date", keep="last")
    base_tz = getattr(base_df["date"].dt, "tz", None)
    pred_tz = getattr(pred_df["date"].dt, "tz", None)
    if base_tz is not None:
        if pred_tz is None:
            pred_df["date"] = pred_df["date"].dt.tz_localize(base_tz)
        else:
            pred_df["date"] = pred_df["date"].dt.tz_convert(base_tz)
    else:
        if pred_tz is not None:
            pred_df["date"] = pred_df["date"].dt.tz_localize(None)
    sanitized = base_pair.replace("/", "_")
    rename_map = {}
    if "prediction" in pred_df.columns:
        rename_map["prediction"] = f"lstm_pred_{sanitized}"
    if "actual" in pred_df.columns:
        rename_map["actual"] = f"lstm_actual_{sanitized}"
    pred_df = pred_df.rename(columns=rename_map)
    cols = ["date"] + [col for col in pred_df.columns if col != "date"]
    return base_df.merge(pred_df[cols], on="date", how="left")
