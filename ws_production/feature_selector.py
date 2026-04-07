"""Feature Selector — reduces overfitting by selecting robust features.

Methods: mutual information, correlation filter, stability (rolling IC).

Usage:
    from workspace.feature_selector import select_features
    selected = select_features(df, target_col="future_return", max_features=15)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _mutual_info_scores(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Compute mutual information between each feature and target."""
    try:
        from sklearn.feature_selection import mutual_info_regression
        mi = mutual_info_regression(
            X.fillna(0).replace([np.inf, -np.inf], 0).values,
            y.fillna(0).values,
            random_state=42,
            n_neighbors=5,
        )
        return pd.Series(mi, index=X.columns)
    except ImportError:
        # Fallback: absolute correlation
        return X.corrwith(y).abs()


def _correlation_filter(X: pd.DataFrame, threshold: float = 0.9) -> List[str]:
    """Remove highly correlated features (keep first of each correlated pair)."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return [c for c in X.columns if c not in to_drop]


def _rolling_ic_stability(X: pd.DataFrame, y: pd.Series, window: int = 200) -> pd.Series:
    """Compute rolling IC stability (mean IC / std IC) for each feature."""
    stability = {}
    for col in X.columns:
        rolling_corr = X[col].rolling(window).corr(y)
        ic_mean = rolling_corr.mean()
        ic_std = rolling_corr.std()
        stability[col] = abs(ic_mean) / (ic_std + 1e-10)
    return pd.Series(stability)


def select_features(
    df: pd.DataFrame,
    target_col: str = "future_return",
    *,
    max_features: int = 15,
    corr_threshold: float = 0.85,
    min_mi_quantile: float = 0.25,
    exclude_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Select top features using multi-criteria ranking.

    Steps:
    1. Remove constant/near-constant columns
    2. Remove highly correlated features (keep more informative one)
    3. Rank by mutual information with target
    4. Rank by rolling IC stability
    5. Combined ranking → top N

    Returns:
        {
            "selected": [col_names],
            "scores": {col: {mi, stability, combined_rank}},
            "removed_correlated": [col_names],
            "removed_constant": [col_names],
        }
    """
    if exclude_cols is None:
        exclude_cols = ["date", "open", "high", "low", "close", "volume", "pair"]

    # Also exclude non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols_base = [c for c in df.columns if c not in exclude_cols and c != target_col]
    feature_cols_base = [c for c in feature_cols_base if c in numeric_cols]

    feature_cols = feature_cols_base
    if target_col not in df.columns:
        return {"selected": feature_cols[:max_features], "scores": {}, "error": "target not found"}

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Clean
    X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    y = y.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    # Step 1: Remove constant columns
    std = X.std()
    constant_cols = list(std[std < 1e-10].index)
    X = X.drop(columns=constant_cols)

    if X.empty:
        return {"selected": [], "scores": {}, "removed_constant": constant_cols}

    # Step 2: Correlation filter
    surviving = _correlation_filter(X, threshold=corr_threshold)
    removed_corr = [c for c in X.columns if c not in surviving]
    X = X[surviving]

    if X.empty or len(X.columns) == 0:
        return {"selected": [], "scores": {}, "removed_correlated": removed_corr}

    # Step 3: Mutual information
    mi_scores = _mutual_info_scores(X, y)
    mi_rank = mi_scores.rank(ascending=False)

    # Step 4: Stability
    stability_scores = _rolling_ic_stability(X, y, window=min(200, len(df) // 3))
    stability_rank = stability_scores.rank(ascending=False)

    # Step 5: Combined ranking (lower = better)
    combined_rank = 0.6 * mi_rank + 0.4 * stability_rank
    combined_rank = combined_rank.sort_values()

    selected = list(combined_rank.head(max_features).index)

    scores = {}
    for col in selected:
        scores[col] = {
            "mutual_info": float(mi_scores.get(col, 0)),
            "stability": float(stability_scores.get(col, 0)),
            "combined_rank": float(combined_rank.get(col, 999)),
        }

    return {
        "selected": selected,
        "scores": scores,
        "removed_constant": constant_cols,
        "removed_correlated": removed_corr,
        "total_candidates": len(feature_cols),
        "after_filter": len(X.columns),
        "final_count": len(selected),
    }


def build_feature_matrix(
    pairs: List[str],
    *,
    exchange: str = "kucoin",
    timeframe: str = "1h",
    label_period: int = 12,
    max_features: int = 15,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Load data, compute features, select best ones, return clean matrix.

    Returns (df_with_features, selected_columns, selection_report).
    """
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from agent_market.freqai.features import apply_configured_features
    from agent_market.freqai.training.labels import future_return

    feature_cfg_path = ROOT / "user_data" / "freqai_features_real.json"
    import json
    feature_cfg = json.loads(feature_cfg_path.read_text(encoding="utf-8-sig"))

    frames = []
    for pair in pairs:
        slug = pair.replace("/", "_")
        data_path = ROOT / "user_data" / "data" / exchange / f"{slug}-{timeframe}.feather"
        if not data_path.exists():
            continue
        df = pd.read_feather(data_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = apply_configured_features(df, feature_cfg)
        df["future_return"] = future_return(df["close"], label_period)
        df["pair"] = pair
        frames.append(df)

    if not frames:
        return pd.DataFrame(), [], {"error": "no data"}

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["future_return"])

    report = select_features(combined, "future_return", max_features=max_features)
    return combined, report["selected"], report


__all__ = ["select_features", "build_feature_matrix"]
