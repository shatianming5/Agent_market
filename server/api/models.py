from __future__ import annotations

from typing import Optional

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
    "ExpressionReq",
    "FeatureReq",
    "FlowReq",
    "HyperoptReq",
    "RLTrainReq",
    "TrainReq",
]
