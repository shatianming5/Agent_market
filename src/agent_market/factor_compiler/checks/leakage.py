from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..api_models import ExprArg, ExprNode
from .types import CheckResult, fail, ok


def _literal_int(arg: ExprArg) -> Optional[int]:
    if isinstance(arg, bool) or arg is None:
        return None
    if isinstance(arg, int):
        return int(arg)
    if isinstance(arg, float) and float(arg).is_integer():
        return int(arg)
    if isinstance(arg, ExprNode) and str(arg.op) in {"pos", "neg", "const"}:
        if not arg.args:
            return None
        inner = _literal_int(arg.args[0])
        if inner is None:
            return None
        if str(arg.op) == "neg":
            return -inner
        return inner
    return None


def _walk(expr: ExprNode) -> list[ExprNode]:
    out: list[ExprNode] = []
    stack: list[ExprNode] = [expr]
    while stack:
        node = stack.pop()
        out.append(node)
        for child in node.args or []:
            if isinstance(child, ExprNode):
                stack.append(child)
    return out


def check_no_negative_shift(expr: ExprNode) -> CheckResult:
    for node in _walk(expr):
        op = str(node.op)
        if op not in {"shift", "lag"}:
            continue
        args = list(node.args or [])
        if len(args) < 2:
            continue
        n = _literal_int(args[1])
        if n is not None and n < 0:
            return fail(
                "leakage_shift_test",
                code="LEAKAGE_NEGATIVE_SHIFT",
                message=f"Negative shift detected: {op}(..., {n})",
                details={"op": op, "n": int(n)},
            )
    return ok("leakage_shift_test", message="no_negative_shift")


def _corr_abs(x: pd.Series, y: pd.Series) -> float:
    aligned = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    return float(abs(aligned["x"].corr(aligned["y"])))


def check_permutation_leakage(
    factor: pd.Series,
    target: pd.Series,
    *,
    n_perm: int = 50,
    seed: int = 42,
    corr_threshold: float = 0.98,
    p_threshold: float = 0.05,
) -> CheckResult:
    """
    Minimal leakage sanity check using a permutation test.

    Heuristic: if abs(corr(factor, target)) is *extremely* high and statistically unlikely under
    permutation, flag as suspicious (could indicate label leakage or a bug).
    """

    base = _corr_abs(factor, target)
    if not np.isfinite(base):
        return ok(
            "leakage_permutation_test",
            message="skipped_empty_or_nan",
            details={"corr_abs": None, "n_perm": int(n_perm)},
        )

    rng = np.random.default_rng(int(seed))
    aligned = pd.concat([factor.rename("x"), target.rename("y")], axis=1).dropna()
    x = aligned["x"]
    y = aligned["y"]
    perms: list[float] = []
    y_values = y.to_numpy()
    for _ in range(int(n_perm)):
        perm_y = pd.Series(rng.permutation(y_values), index=y.index)
        perms.append(_corr_abs(x, perm_y))

    perms_arr = np.asarray(perms, dtype=float)
    p_value = float((np.sum(perms_arr >= float(base)) + 1) / (len(perms_arr) + 1))
    details: Dict[str, Any] = {
        "corr_abs": float(base),
        "p_value": p_value,
        "n_perm": int(n_perm),
        "corr_threshold": float(corr_threshold),
        "p_threshold": float(p_threshold),
    }

    if float(base) >= float(corr_threshold) and p_value <= float(p_threshold):
        return fail(
            "leakage_permutation_test",
            code="LEAKAGE_PERMUTATION_SUSPECT",
            message=f"Suspiciously high corr_abs={base:.4f} (p={p_value:.4g})",
            details=details,
        )
    return ok("leakage_permutation_test", message="ok", details=details)


def check_shift_test(
    factor: pd.Series,
    target: pd.Series,
    *,
    shift: int = 1,
    corr_threshold: float = 0.98,
) -> CheckResult:
    """
    Shift test (plan.md 3.5.4):

    If shifting the factor by +1 still yields extremely high correlation with the target,
    it could indicate leakage or a degenerate evaluation setup.
    """

    s = int(shift)
    if s <= 0:
        return ok("leakage_shift_test", message="skipped_invalid_shift", details={"shift": int(shift)})

    base = _corr_abs(factor, target)
    shifted = _corr_abs(factor.shift(s), target)
    details = {"shift": int(s), "corr_abs": _safe_float(base), "corr_abs_shifted": _safe_float(shifted), "corr_threshold": float(corr_threshold)}

    if np.isfinite(base) and np.isfinite(shifted) and float(base) >= float(corr_threshold) and float(shifted) >= float(corr_threshold):
        return fail(
            "leakage_shift_test",
            code="LEAKAGE_SHIFT_SUSPECT",
            message=f"Shift test suspicious: corr_abs={base:.4f} shifted={shifted:.4f}",
            details=details,
        )
    return ok("leakage_shift_test", message="ok", details=details)


def _safe_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def check_label_leakage_signature(
    factor: pd.Series,
    target: pd.Series,
    *,
    max_lag: int = 5,
    corr_threshold: float = 0.98,
    spike_ratio: float = 0.5,
) -> CheckResult:
    """
    Label leakage signature (plan.md 3.5.4):

    Compute correlations across small lags and flag an abnormal spike at lag=0.
    """

    k = int(max_lag)
    if k <= 0:
        return ok("leakage_signature", message="skipped_invalid_max_lag", details={"max_lag": int(max_lag)})

    corrs: list[dict[str, Any]] = []
    for lag in range(-k, k + 1):
        c = _corr_abs(factor.shift(lag), target)
        corrs.append({"lag": int(lag), "corr_abs": _safe_float(c)})

    valid = [(row["lag"], float(row["corr_abs"])) for row in corrs if row.get("corr_abs") is not None]
    if not valid:
        return ok("leakage_signature", message="skipped_empty_or_nan", details={"corrs": corrs})

    valid_sorted = sorted(valid, key=lambda x: x[1], reverse=True)
    (best_lag, best_corr) = valid_sorted[0]
    second_corr = valid_sorted[1][1] if len(valid_sorted) > 1 else 0.0
    details = {
        "max_lag": int(k),
        "best_lag": int(best_lag),
        "best_corr_abs": float(best_corr),
        "second_best_corr_abs": float(second_corr),
        "corr_threshold": float(corr_threshold),
        "spike_ratio": float(spike_ratio),
        "corrs": corrs,
    }

    if int(best_lag) == 0 and float(best_corr) >= float(corr_threshold) and float(second_corr) <= float(best_corr) * float(spike_ratio):
        return fail(
            "leakage_signature",
            code="LEAKAGE_SIGNATURE_SPIKE",
            message=f"0-lag spike detected: corr_abs={best_corr:.4f} second={second_corr:.4f}",
            details=details,
        )
    return ok("leakage_signature", message="ok", details=details)
