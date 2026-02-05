from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ArtifactRef(BaseModel):
    name: str
    path: str


class TCASource(BaseModel):
    kind: Literal["freqtrade_backtest"] = "freqtrade_backtest"
    backtest_zip: str
    strategy: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class TCAQuantiles(BaseModel):
    p0: Optional[float] = None
    p5: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    p95: Optional[float] = None
    p100: Optional[float] = None


class TCASummary(BaseModel):
    trades: int
    win_rate: Optional[float] = None
    profit_abs_total: Optional[float] = None
    profit_ratio_total: Optional[float] = None
    fees_total: Optional[float] = None

    # v1 placeholders (filled in Phase 3 when LOB/mid data exists)
    slippage_bps_est: Optional[float] = None


class TCATimeRange(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None


class TCAStrategyMeta(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class TCAMeta(BaseModel):
    run_id: str
    generated_at: str
    exchange: Optional[str] = None
    market: Optional[str] = None  # "perp|spot" (best-effort)
    symbols: List[str] = Field(default_factory=list)
    time_range: TCATimeRange = Field(default_factory=TCATimeRange)
    timeframe: Optional[str] = None
    data_sources: List[str] = Field(default_factory=list)
    strategy: TCAStrategyMeta


class TCAOrder(BaseModel):
    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    type: Literal["MKT", "LMT"]
    qty: float
    limit_px: Optional[float] = None
    submit_ts: Optional[str] = None
    cancel_ts: Optional[str] = None


class TCAFill(BaseModel):
    order_id: str
    fill_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: float
    price: float
    ts: str
    fee: Optional[float] = None
    liquidity: Optional[Literal["TAKER", "MAKER"]] = None


class TCABenchmark(BaseModel):
    definition: str
    value_series_ref: Optional[str] = None


class TCABenchmarks(BaseModel):
    arrival_mid: TCABenchmark
    vwap: TCABenchmark


class TCACostValue(BaseModel):
    bps: Optional[float] = None
    quote_ccy: Optional[float] = None


class TCAImplementationShortfallComponents(BaseModel):
    spread: TCACostValue = Field(default_factory=TCACostValue)
    delay: TCACostValue = Field(default_factory=TCACostValue)
    market_impact: TCACostValue = Field(default_factory=TCACostValue)
    fees: TCACostValue = Field(default_factory=TCACostValue)


class TCAImplementationShortfall(BaseModel):
    total: TCACostValue = Field(default_factory=TCACostValue)
    by_component: TCAImplementationShortfallComponents = Field(
        default_factory=TCAImplementationShortfallComponents
    )


class TCASlippageDistribution(BaseModel):
    p50_bps: Optional[float] = None
    p90_bps: Optional[float] = None
    p99_bps: Optional[float] = None


class TCAFillQuality(BaseModel):
    fill_rate: Optional[float] = None
    avg_fill_latency_ms: Optional[float] = None
    cancel_rate: Optional[float] = None


class TCACosts(BaseModel):
    implementation_shortfall: TCAImplementationShortfall = Field(
        default_factory=TCAImplementationShortfall
    )
    slippage_distribution: TCASlippageDistribution = Field(
        default_factory=TCASlippageDistribution
    )
    fill: TCAFillQuality = Field(default_factory=TCAFillQuality)


class TCARegime(BaseModel):
    vol_bucket: Optional[str] = None
    liquidity_bucket: Optional[str] = None


class TCAPlotRef(BaseModel):
    name: str
    path: str


class TCADiagnostics(BaseModel):
    regime: TCARegime = Field(default_factory=TCARegime)
    notes: List[str] = Field(default_factory=list)
    plots: List[TCAPlotRef] = Field(default_factory=list)
    participation: Optional[Dict[str, Any]] = None


class TCAReport(BaseModel):
    # plan.md v1 keys (required positions)
    schema_version: str = "1.0"
    meta: TCAMeta
    orders: List[TCAOrder] = Field(default_factory=list)
    fills: List[TCAFill] = Field(default_factory=list)
    benchmarks: TCABenchmarks
    costs: TCACosts = Field(default_factory=TCACosts)
    diagnostics: TCADiagnostics = Field(default_factory=TCADiagnostics)

    # backward-compatible (Phase 1 / pre-v1) keys
    version: int = 1
    run_id: str
    generated_at: str
    source: TCASource
    summary: TCASummary
    distributions: Dict[str, TCAQuantiles] = Field(default_factory=dict)
    per_pair: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: List[ArtifactRef] = Field(default_factory=list)


__all__ = [
    "ArtifactRef",
    "TCABenchmark",
    "TCABenchmarks",
    "TCACosts",
    "TCACostValue",
    "TCADiagnostics",
    "TCAFill",
    "TCAFillQuality",
    "TCAImplementationShortfall",
    "TCAImplementationShortfallComponents",
    "TCAMeta",
    "TCAOrder",
    "TCAPlotRef",
    "TCAQuantiles",
    "TCARegime",
    "TCAReport",
    "TCASlippageDistribution",
    "TCASource",
    "TCAStrategyMeta",
    "TCASummary",
    "TCATimeRange",
]
