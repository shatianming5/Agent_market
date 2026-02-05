from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ExpressionReq(BaseModel):
    config: str = Field(..., description="Path to freqtrade JSON config")
    feature_file: str = Field(..., description="Path to feature json")
    output: str = Field("user_data/freqai_expressions.json")
    timeframe: str = Field("4h")
    llm_model: str = Field("gpt-3.5-turbo")
    llm_count: int = 20
    llm_loops: int = 1
    llm_timeout: float = 60
    llm_api_key: Optional[str] = None
    feedback: Optional[str] = None
    feedback_top: int = 0


class BacktestReq(BaseModel):
    config: str = Field(...)
    strategy: str = Field("ExpressionLongStrategy")
    strategy_path: str = Field("user_data/strategies")
    timerange: str = Field("20210101-20211231")
    freqaimodel: str = Field("LightGBMRegressor")
    export: bool = True
    export_filename: str = Field("user_data/backtest_results/latest_trades_multi")


class FlowReq(BaseModel):
    config: str = Field(..., description="Path to agent_flow JSON config")
    steps: Optional[object] = Field(
        None,
        description="Either space separated string or list: feature portfolio expression ml rl backtest",
    )


class FeatureReq(BaseModel):
    config: str = Field(...)
    output: str = Field("user_data/freqai_features.json")
    timeframe: str = Field("4h")
    pairs: Optional[str] = Field(
        None, description="Comma or space separated pairs, e.g. 'BTC/USDT ETH/USDT'"
    )


class HyperoptReq(BaseModel):
    config: str = Field(...)  # freqtrade config
    strategy: str = Field("ExpressionLongStrategy")
    strategy_path: str = Field("user_data/strategies")
    timerange: str = Field("20210101-20210430")
    spaces: str = Field("buy sell protection")
    hyperopt_loss: str = Field("SharpeHyperOptLoss")
    epochs: int = Field(20)
    freqaimodel: Optional[str] = Field("LightGBMRegressor")
    job_workers: int = Field(-1)


class RLTrainReq(BaseModel):
    config: str = Field(..., description="Path to RL training JSON config (train_ppo.json)")


class TrainReq(BaseModel):
    config: Optional[str] = Field(
        None, description="Path to ML training JSON config (train_*.json)"
    )
    config_obj: Optional[dict] = Field(None, description="Inline ML training config (JSON object)")


__all__ = [
    "BacktestReq",
    "CaptureReq",
    "ExpressionReq",
    "FactorCompileReq",
    "FactorEvalReq",
    "FeatureReq",
    "FlowReq",
    "HyperoptReq",
    "LobRebuildReq",
    "MicroFeatureReq",
    "RLTrainReq",
    "TCAReq",
    "TrainReq",
]


class CaptureReq(BaseModel):
    exchange: str = Field("kucoin", description="Exchange name (currently only supports kucoin)")
    symbols: str = Field("BTC-USDT", description="Comma/space separated symbols, e.g. BTC-USDT ETH-USDT")
    channels: str = Field("match,level2", description="Comma/space separated channels: match, level2")
    duration_sec: int = Field(60, description="Capture duration in seconds (live mode)")
    out_dir: Optional[str] = Field(None, description="Optional session output directory")
    fixture: Optional[str] = Field(None, description="Optional offline fixture (jsonl) to replay")


class LobRebuildReq(BaseModel):
    exchange: str = Field("kucoin", description="Exchange name (currently only supports kucoin)")
    capture_dir: str = Field(..., description="Capture session dir containing level2.ndjson.gz")
    snapshot: str = Field(..., description="Snapshot JSON with bids/asks + sequence (fixture)")
    symbol: str = Field("BTC-USDT", description="Symbol, e.g. BTC-USDT")
    depth: int = Field(20, description="Top-N depth to persist for bid/ask arrays")
    out_dir: Optional[str] = Field(
        None, description="Optional output directory (default: artifacts/lob_rebuild/<id>)"
    )


class MicroFeatureReq(BaseModel):
    config: str = Field(..., description="Path to freqtrade JSON config")
    run_id: Optional[str] = Field(None, description="Optional run_id (default: generated)")
    out_dir: Optional[str] = Field(
        None,
        description="Optional output dir (default: artifacts/runs/<run_id>/micro_feature)",
    )
    timeframe: Optional[str] = Field(None, description="Optional timeframe override (e.g. 1h)")
    timerange: Optional[str] = Field(None, description="Optional timerange YYYYMMDD-YYYYMMDD")
    data_dir: Optional[str] = Field(None, description="Optional data dir root override (default: from config)")
    pairs: Optional[List[str]] = Field(None, description="Optional pair override list")


class TCAReq(BaseModel):
    run_id: Optional[str] = Field(
        None,
        description="Optional run_id (default: latest Flow run_id if available)",
    )
    backtest_zip: Optional[str] = Field(
        None, description="Optional path to backtest-result-*.zip (default: latest)"
    )
    results_dir: str = Field("user_data/backtest_results", description="Dir to locate latest backtest zip")
    out: Optional[str] = Field(None, description="Optional output JSON path override")
    html: bool = Field(False, description="Whether to also write a minimal HTML report")


class FactorCompileReq(BaseModel):
    run_id: Optional[str] = Field(
        None,
        description="Optional run_id to write artifacts under artifacts/runs/<run_id>/factor_compile",
    )
    out_dir: Optional[str] = Field(
        None, description="Optional output dir (default: artifacts/factor_compile/<id>)"
    )
    spec: Optional[dict] = Field(None, description="Inline FactorSpec JSON object")
    spec_path: Optional[str] = Field(None, description="Path to FactorSpec JSON file")


class FactorEvalReq(BaseModel):
    run_id: Optional[str] = Field(
        None,
        description="Optional run_id to reuse compiled_expression from factor_compile and write factor_eval artifacts under artifacts/runs/<run_id>/factor_eval",
    )
    out_dir: Optional[str] = Field(
        None, description="Optional output dir (default: artifacts/factor_eval/<id>)"
    )
    expression: Optional[str] = Field(None, description="Inline ExpressionEngine expression string")
    expression_path: Optional[str] = Field(None, description="Path to expression text file")
    rows: int = Field(256, description="Rows for minimal offline eval")
    seed: int = Field(7, description="Seed for minimal offline eval")
