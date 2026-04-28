from __future__ import annotations

from typing import Any, Optional


def _is_level_sequence(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return True
    try:  # pragma: no cover - optional dependency
        import numpy as np  # noqa: PLC0415

        if isinstance(value, np.ndarray):
            return True
    except Exception:
        pass
    return False


def _ensure_pandas() -> Any:
    try:
        import pandas as pd  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Missing dependency for microstructure features: {exc}") from exc
    return pd


def _safe_list_n(value: Any, n: int) -> Optional[float]:
    try:
        if not _is_level_sequence(value):
            return None
        if n < 0 or n >= len(value):
            return None
        v = value[n]
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def compute_microprice(lob: Any, *, eps: float = 1e-12) -> Any:
    """Compute microprice from top-of-book."""
    pd = _ensure_pandas()
    bid1 = lob.get("bid1")
    ask1 = lob.get("ask1")
    bid_sz1 = lob.get("bid_sz").apply(lambda xs: _safe_list_n(xs, 0))
    ask_sz1 = lob.get("ask_sz").apply(lambda xs: _safe_list_n(xs, 0))

    denom = (bid_sz1.fillna(0.0) + ask_sz1.fillna(0.0)) + float(eps)
    num = (bid1.astype("float64") * ask_sz1.astype("float64")) + (
        ask1.astype("float64") * bid_sz1.astype("float64")
    )
    out = num / denom
    out = out.where((bid1.notna()) & (ask1.notna()) & (bid_sz1.notna()) & (ask_sz1.notna()))
    return out


def _depth_from_sizes(sizes: Any, levels: int) -> float:
    total = 0.0
    if not _is_level_sequence(sizes):
        return total
    for x in sizes[: int(levels)]:
        try:
            if x is None:
                continue
            v = float(x)
            if not (v == v):  # NaN
                continue
            if v <= 0:
                continue
            total += v
        except Exception:
            continue
    return float(total)


def compute_depth_bid_levels(lob: Any, *, levels: int) -> Any:
    return lob.get("bid_sz").apply(lambda xs: _depth_from_sizes(xs, levels=levels))


def compute_depth_ask_levels(lob: Any, *, levels: int) -> Any:
    return lob.get("ask_sz").apply(lambda xs: _depth_from_sizes(xs, levels=levels))


def compute_imbalance_from_depth(depth_bid: Any, depth_ask: Any, *, eps: float = 1e-12) -> Any:
    denom = depth_bid.astype("float64") + depth_ask.astype("float64") + float(eps)
    return (depth_bid.astype("float64") - depth_ask.astype("float64")) / denom


def _slope_size_distance(
    px_levels: Any,
    sz_levels: Any,
    mid: Any,
    *,
    levels: int,
    eps: float = 1e-12,
) -> Optional[float]:
    try:
        if not _is_level_sequence(px_levels) or not _is_level_sequence(sz_levels):
            return None
        m = float(mid)
    except Exception:
        return None

    xs: list[float] = []
    ys: list[float] = []
    n = max(1, int(levels))
    for i in range(min(n, len(px_levels), len(sz_levels))):
        try:
            px = px_levels[i]
            sz = sz_levels[i]
            if px is None or sz is None:
                continue
            dist = abs(float(m) - float(px)) / (abs(float(m)) + float(eps))
            size = float(sz)
        except Exception:
            continue
        if not (dist == dist) or not (size == size):  # NaN
            continue
        if not (size > 0):
            continue
        xs.append(float(dist))
        ys.append(float(size))

    if len(xs) < 2:
        return None
    x_mean = float(sum(xs) / len(xs))
    y_mean = float(sum(ys) / len(ys))
    denom = float(sum((x - x_mean) ** 2 for x in xs))
    if denom <= float(eps):
        return None
    numer = float(sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)))
    return float(numer / (denom + float(eps)))


def compute_slope_bid_levels(lob: Any, *, levels: int, eps: float = 1e-12) -> Any:
    """Compute best-effort bid book slope (size vs relative distance to mid)."""
    pd = _ensure_pandas()
    bid_px = lob.get("bid_px")
    bid_sz = lob.get("bid_sz")
    mid = lob.get("mid")
    if bid_px is None or bid_sz is None or mid is None:
        return pd.Series([None] * int(lob.shape[0]), index=lob.index)

    out = [_slope_size_distance(bpx, bsz, m, levels=int(levels), eps=float(eps)) for bpx, bsz, m in zip(bid_px, bid_sz, mid)]
    return pd.Series(out, index=lob.index)


def _convexity_from_sizes(sizes: Any, *, levels: int, eps: float = 1e-12) -> Optional[float]:
    try:
        if not _is_level_sequence(sizes):
            return None
        L = int(levels)
    except Exception:
        return None
    if L <= 0:
        return None
    n = min(L, len(sizes))
    if n <= 0:
        return None
    num = 0.0
    den = 0.0
    for i in range(n):
        try:
            sz = sizes[i]
            if sz is None:
                continue
            v = float(sz)
        except Exception:
            continue
        if not (v == v):  # NaN
            continue
        if not (v > 0):
            continue
        w = float((i + 1) / float(L))
        num += float(v) * (w * w)
        den += float(v)
    if den <= float(eps):
        return None
    return float(num / (den + float(eps)))


def compute_convexity_levels(lob: Any, *, levels: int, eps: float = 1e-12) -> Any:
    """Compute a simple L2 convexity proxy from size distribution across levels."""
    pd = _ensure_pandas()
    bid_sz = lob.get("bid_sz")
    ask_sz = lob.get("ask_sz")
    if bid_sz is None and ask_sz is None:
        return pd.Series([None] * int(lob.shape[0]), index=lob.index)

    out: list[Optional[float]] = []
    rows = int(lob.shape[0])
    for i in range(rows):
        b = None
        a = None
        if bid_sz is not None:
            try:
                b = _convexity_from_sizes(bid_sz.iloc[i], levels=int(levels), eps=float(eps))
            except Exception:
                b = None
        if ask_sz is not None:
            try:
                a = _convexity_from_sizes(ask_sz.iloc[i], levels=int(levels), eps=float(eps))
            except Exception:
                a = None
        if b is None and a is None:
            out.append(None)
        elif b is None:
            out.append(a)
        elif a is None:
            out.append(b)
        else:
            out.append(float((float(b) + float(a)) / 2.0))

    return pd.Series(out, index=lob.index)


__all__ = [
    "compute_depth_ask_levels",
    "compute_depth_bid_levels",
    "compute_imbalance_from_depth",
    "compute_microprice",
    "compute_slope_bid_levels",
    "compute_convexity_levels",
]
