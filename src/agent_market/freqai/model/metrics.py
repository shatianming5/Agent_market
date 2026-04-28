"""Shared model evaluation metrics."""
from __future__ import annotations

import numpy as np


def rmse(preds: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((preds - target) ** 2)))


def multiclass_metrics(probs: np.ndarray, target: np.ndarray) -> dict:
    """Return {acc, logloss, up_prec, dn_prec} for a 3-class classifier (0=dn, 1=flat, 2=up)."""
    probs = np.asarray(probs)
    target = np.asarray(target, dtype=int)
    eps = 1e-12
    if probs.ndim == 1:
        preds = (probs > 0.5).astype(int)
        return {"acc": float((preds == target).mean())}
    pred_class = probs.argmax(axis=1)
    acc = float((pred_class == target).mean())
    # log-loss
    n = len(target)
    logloss = -float(np.mean(np.log(np.clip(probs[np.arange(n), target], eps, 1.0))))
    # Directional precision (how often "up predicted" was truly up)
    up_mask = pred_class == 2
    dn_mask = pred_class == 0
    up_prec = float((target[up_mask] == 2).mean()) if up_mask.any() else 0.0
    dn_prec = float((target[dn_mask] == 0).mean()) if dn_mask.any() else 0.0
    return {
        "accuracy": acc,
        "logloss": logloss,
        "up_precision": up_prec,
        "dn_precision": dn_prec,
        "up_pred_rate": float(up_mask.mean()),
        "dn_pred_rate": float(dn_mask.mean()),
    }
