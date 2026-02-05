from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


def _ensure_pandas() -> Any:
    try:
        import pandas as pd  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Missing dependency for microstructure features: {exc}") from exc
    return pd


@dataclass(frozen=True, slots=True)
class FeatureContext:
    lob: Any
    trades: Optional[Any]
    aligned_trades: Optional[Any]


ComputeFn = Callable[[FeatureContext], Any]


@dataclass(frozen=True, slots=True)
class FeatureDef:
    name: str
    source: str
    kind: str
    params: Dict[str, Any]
    compute: ComputeFn


class FeatureRegistry:
    def __init__(self) -> None:
        self._defs: Dict[str, FeatureDef] = {}

    def register(self, feature: FeatureDef) -> None:
        name = str(feature.name).strip()
        if not name:
            raise ValueError("feature.name must not be empty")
        if name in self._defs:
            raise ValueError(f"Duplicate feature: {name}")
        self._defs[name] = feature

    def get(self, name: str) -> FeatureDef:
        try:
            return self._defs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown feature: {name}") from exc

    def list(self) -> List[FeatureDef]:
        return [self._defs[k] for k in sorted(self._defs)]

    def manifest(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": d.name,
                "source": d.source,
                "kind": d.kind,
                "params": dict(d.params),
            }
            for d in self.list()
        ]


def build_default_microstructure_registry(
    *,
    depth_levels: int = 20,
    windows_sec: Sequence[int] = (10,),
) -> FeatureRegistry:
    """Build the minimal microstructure feature registry (LOB + trades).

    Registers (at least) the features required by Missing-013:
      - mid/spread/rel_spread/microprice
      - depth_bid_L/depth_ask_L/imbalance_L (as depth_bid_{L}, ...)
      - trade_sign
      - vwap_w/ofi_w/arrival_intensity_w (as vwap_{w}, ...)
      - rv_w (realized volatility on mid)
    """
    from agent_market.microstructure.features.core_features import (  # noqa: WPS433
        compute_convexity_levels,
        compute_depth_ask_levels,
        compute_depth_bid_levels,
        compute_imbalance_from_depth,
        compute_microprice,
        compute_slope_bid_levels,
    )
    from agent_market.microstructure.features.volatility_features import (  # noqa: WPS433
        realized_volatility,
    )

    pd = _ensure_pandas()
    import numpy as np  # noqa: PLC0415
    eps = 1e-12

    levels = int(depth_levels)
    if levels <= 0:
        levels = 20
    levels = max(1, min(levels, 200))

    windows = []
    for w in windows_sec:
        try:
            iw = int(w)
        except Exception:
            continue
        if iw > 0:
            windows.append(iw)
    if not windows:
        windows = [10]

    reg = FeatureRegistry()

    def _col(name: str) -> ComputeFn:
        return lambda ctx: ctx.lob.get(name)

    reg.register(
        FeatureDef(
            name="mid",
            source="lob_state",
            kind="mid",
            params={},
            compute=_col("mid"),
        )
    )
    reg.register(
        FeatureDef(
            name="spread",
            source="lob_state",
            kind="spread",
            params={},
            compute=_col("spread"),
        )
    )
    reg.register(
        FeatureDef(
            name="rel_spread",
            source="lob_state",
            kind="rel_spread",
            params={},
            compute=_col("rel_spread"),
        )
    )

    reg.register(
        FeatureDef(
            name="microprice",
            source="lob_state",
            kind="microprice",
            params={},
            compute=lambda ctx: compute_microprice(ctx.lob, eps=eps),
        )
    )

    bid_name = f"depth_bid_{levels}"
    ask_name = f"depth_ask_{levels}"
    imb_name = f"imbalance_{levels}"
    slope_bid_name = f"slope_bid_{levels}"
    convex_name = f"convexity_{levels}"

    reg.register(
        FeatureDef(
            name=bid_name,
            source="lob_state",
            kind="depth_bid",
            params={"levels": levels},
            compute=lambda ctx: compute_depth_bid_levels(ctx.lob, levels=levels),
        )
    )
    reg.register(
        FeatureDef(
            name=ask_name,
            source="lob_state",
            kind="depth_ask",
            params={"levels": levels},
            compute=lambda ctx: compute_depth_ask_levels(ctx.lob, levels=levels),
        )
    )
    reg.register(
        FeatureDef(
            name=imb_name,
            source="lob_state",
            kind="imbalance",
            params={"levels": levels},
            compute=lambda ctx: compute_imbalance_from_depth(
                compute_depth_bid_levels(ctx.lob, levels=levels),
                compute_depth_ask_levels(ctx.lob, levels=levels),
                eps=eps,
            ),
        )
    )

    reg.register(
        FeatureDef(
            name=slope_bid_name,
            source="lob_state",
            kind="slope_bid",
            params={"levels": levels},
            compute=lambda ctx: compute_slope_bid_levels(ctx.lob, levels=levels, eps=eps),
        )
    )

    reg.register(
        FeatureDef(
            name=convex_name,
            source="lob_state",
            kind="convexity",
            params={"levels": levels},
            compute=lambda ctx: compute_convexity_levels(ctx.lob, levels=levels, eps=eps),
        )
    )

    def _aligned_or_zero(ctx: FeatureContext, name: str) -> Any:
        if ctx.aligned_trades is None:
            return pd.Series([0.0] * int(ctx.lob.shape[0]), index=ctx.lob.index)
        col = ctx.aligned_trades.get(name)
        if col is None:
            return pd.Series([0.0] * int(ctx.lob.shape[0]), index=ctx.lob.index)
        return pd.Series(col, index=ctx.lob.index).astype("float64").fillna(0.0)

    def _expected_slippage_proxy(ctx: FeatureContext) -> Any:
        mid = ctx.lob.get("mid").astype("float64")
        spread = ctx.lob.get("spread").astype("float64")
        spread_bps = 10000.0 * spread / (mid.abs() + float(eps))

        depth_bid = compute_depth_bid_levels(ctx.lob, levels=levels).astype("float64")
        depth_ask = compute_depth_ask_levels(ctx.lob, levels=levels).astype("float64")
        imb = compute_imbalance_from_depth(depth_bid, depth_ask, eps=eps).abs().fillna(0.0)

        w = int(windows[0])
        intensity = _aligned_or_zero(ctx, f"arrival_intensity_{w}")
        return spread_bps.fillna(0.0) * (1.0 + 0.5 * imb) * (1.0 + 0.1 * np.log1p(intensity * float(w)))

    def _fill_prob_proxy(ctx: FeatureContext) -> Any:
        mid = ctx.lob.get("mid").astype("float64")
        spread = ctx.lob.get("spread").astype("float64")
        spread_bps = 10000.0 * spread / (mid.abs() + float(eps))

        depth_bid = compute_depth_bid_levels(ctx.lob, levels=levels).astype("float64")
        depth_ask = compute_depth_ask_levels(ctx.lob, levels=levels).astype("float64")
        depth_total = (depth_bid.fillna(0.0) + depth_ask.fillna(0.0)).clip(lower=0.0)
        imb = compute_imbalance_from_depth(depth_bid, depth_ask, eps=eps).abs().fillna(0.0)

        w = int(windows[0])
        intensity = _aligned_or_zero(ctx, f"arrival_intensity_{w}")
        score = (
            -0.25 * spread_bps.fillna(0.0)
            + 0.02 * np.log1p(depth_total)
            + 0.2 * np.log1p(intensity * float(w))
            + 0.5 * imb
        )
        prob = 1.0 / (1.0 + np.exp(-score.astype("float64")))
        return prob.clip(lower=0.0, upper=1.0)

    def _toxicity_proxy(ctx: FeatureContext) -> Any:
        mid = ctx.lob.get("mid").astype("float64")
        ret_abs = mid.pct_change().abs().fillna(0.0)

        w = int(windows[0])
        ofi = _aligned_or_zero(ctx, f"ofi_{w}").abs().fillna(0.0)
        return (ofi * ret_abs * 10000.0).astype("float64")

    reg.register(
        FeatureDef(
            name="expected_slippage_proxy",
            source="lob_state+trades",
            kind="expected_slippage_proxy",
            params={"levels": levels, "window_sec": int(windows[0])},
            compute=_expected_slippage_proxy,
        )
    )
    reg.register(
        FeatureDef(
            name="fill_prob_proxy",
            source="lob_state+trades",
            kind="fill_prob_proxy",
            params={"levels": levels, "window_sec": int(windows[0])},
            compute=_fill_prob_proxy,
        )
    )
    reg.register(
        FeatureDef(
            name="toxicity_proxy",
            source="lob_state+trades",
            kind="toxicity_proxy",
            params={"levels": levels, "window_sec": int(windows[0])},
            compute=_toxicity_proxy,
        )
    )

    def _aligned(name: str) -> ComputeFn:
        def _fn(ctx: FeatureContext) -> Any:
            if ctx.aligned_trades is None:
                return pd.Series([None] * int(ctx.lob.shape[0]))
            return ctx.aligned_trades.get(name)

        return _fn

    reg.register(
        FeatureDef(
            name="trade_sign",
            source="trades",
            kind="trade_sign",
            params={},
            compute=_aligned("trade_sign"),
        )
    )

    for w in windows:
        reg.register(
            FeatureDef(
                name=f"rv_{w}",
                source="lob_state",
                kind="realized_vol",
                params={"window": int(w), "basis": "mid_pct_change"},
                compute=lambda ctx, _w=int(w): realized_volatility(ctx.lob.get("mid"), window=_w),
            )
        )
        reg.register(
            FeatureDef(
                name=f"vwap_{w}",
                source="trades",
                kind="vwap",
                params={"window_sec": int(w)},
                compute=_aligned(f"vwap_{w}"),
            )
        )
        reg.register(
            FeatureDef(
                name=f"ofi_{w}",
                source="trades",
                kind="ofi",
                params={"window_sec": int(w)},
                compute=_aligned(f"ofi_{w}"),
            )
        )
        reg.register(
            FeatureDef(
                name=f"arrival_intensity_{w}",
                source="trades",
                kind="arrival_intensity",
                params={"window_sec": int(w)},
                compute=_aligned(f"arrival_intensity_{w}"),
            )
        )
        reg.register(
            FeatureDef(
                name=f"buy_vol_{w}",
                source="trades",
                kind="buy_vol",
                params={"window_sec": int(w)},
                compute=_aligned(f"buy_vol_{w}"),
            )
        )
        reg.register(
            FeatureDef(
                name=f"sell_vol_{w}",
                source="trades",
                kind="sell_vol",
                params={"window_sec": int(w)},
                compute=_aligned(f"sell_vol_{w}"),
            )
        )

    return reg


def compute_registry_features(*, ctx: FeatureContext, registry: FeatureRegistry) -> Any:
    """Compute all features in registry and return a dataframe."""
    pd = _ensure_pandas()
    out = ctx.lob[["ts", "symbol", "event"]].copy()
    for feat in registry.list():
        out[feat.name] = feat.compute(ctx)
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    return out


__all__ = [
    "ComputeFn",
    "FeatureContext",
    "FeatureDef",
    "FeatureRegistry",
    "build_default_microstructure_registry",
    "compute_registry_features",
]
