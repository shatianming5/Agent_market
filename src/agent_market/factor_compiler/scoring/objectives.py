from __future__ import annotations

import numpy as np
import pandas as pd


def _align(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    return pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()


def information_coefficient(factor: pd.Series, target: pd.Series) -> float:
    aligned = _align(factor, target)
    if len(aligned) < 2:
        return float("nan")
    return float(aligned["x"].corr(aligned["y"]))


def rank_information_coefficient(factor: pd.Series, target: pd.Series) -> float:
    aligned = _align(factor, target)
    if len(aligned) < 2:
        return float("nan")
    fx = aligned["x"].rank(method="average")
    ty = aligned["y"].rank(method="average")
    return float(fx.corr(ty))

