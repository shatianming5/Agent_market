import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import freqtrade.scripts.freqai_expression_agent as expr
from agent_market.freqai.context import resolve_feature_context
from agent_market.freqai.rl.env import TradingEnv
from agent_market.freqai.rl.trainer import RLTrainer
from agent_market.freqai.training.inference import AuxModelPredictor
from agent_market.freqai.training.pipeline import FeatureDatasetBuilder, TrainingPipeline

def _sample_dataframe(size: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=size, freq="h")
    close = np.linspace(100, 110, size)
    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "date": dates,
            "close": close,
            "high": close + 1,
            "low": close - 1,
            "volume": np.full(size, 1000.0),
            "feat_test": np.sin(np.linspace(0, 3.14, size)),
        }
    )
    return df


def test_generate_llm_expressions_with_mock(monkeypatch):
    df = _sample_dataframe()
    feature_cols = ["feat_test"]
    feature_cfg = {
        "pairs": ["BTC/USDT"],
        "exchange": "binanceus",
        "features": [
            {
                "name": "feat_test",
                "type": "sma_pct",
                "period": 5,
            }
        ],
    }
    combos: list[dict] = []

    fake_response = {
        "expressions": [
            {
                "name": "llm_test",
                "expression": "z(feat_test)",
                "description": "test factor",
                "reason": "captures oscillation",
                "category": "momentum",
            }
        ]
    }

    def _fake_request(prompt: str, config):  # noqa: ANN001
        return json.dumps(fake_response), {"prompt_tokens": 10, "completion_tokens": 5}

    monkeypatch.setattr(expr.llm_utils, "request_completion", _fake_request)

    llm_args = SimpleNamespace(
        llm_count=1,
        llm_base_url="https://mock",
        llm_api_key="dummy",
        llm_model="gpt-3.5-turbo",
        llm_temperature=0.2,
        llm_max_tokens=128,
        llm_retries=1,
        llm_timeout=10,
    )

    scores = expr.generate_llm_expressions(
        df,
        feature_cols,
        feature_cfg,
        combos,
        timeframe="1h",
        label_period=3,
        score_method="correlation",
        complexity_penalty=0.01,
        backtest_weight=0.0,
        periods_per_year=8760,
        stability_windows=2,
        stability_min_samples=5,
        llm_options=llm_args,
    )

    assert scores, "LLM expressions should produce at least one score"
    assert scores[0]["origin"] == "llm"
    assert scores[0]["expression"] == "z(feat_test)"


def test_freqai_settings_validate_dataset(tmp_path):
    from agent_market.config import FreqAISettings

    data_root = tmp_path / "data"
    exchange_dir = data_root / "binanceus"
    exchange_dir.mkdir(parents=True)

    df = _sample_dataframe()
    (exchange_dir / "BTC_USDT-1h.feather").write_bytes(b"")
    # Overwrite with real feather content
    df[["date", "close", "high", "low", "volume"]].to_feather(
        exchange_dir / "BTC_USDT-1h.feather"
    )

    config_data = {
        "datadir": str(data_root),
        "exchange": {
            "name": "binanceus",
            "pair_whitelist": ["BTC/USDT"],
        },
        "freqai": {
            "feature_parameters": {
                "include_timeframes": ["1h"],
                "label_period_candles": 12,
            },
            "train_period_days": 45,
            "backtest_period_days": 15,
        },
    }
    config_path = tmp_path / "config_freqai.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    settings = FreqAISettings.from_file(config_path)
    assert settings.timeframe == "1h"
    assert settings.data_dir == exchange_dir

    # Should not raise
    settings.validate_dataset()

    # Missing??????
    (exchange_dir / "BTC_USDT-1h.feather").unlink()
    with pytest.raises(FileNotFoundError):
        settings.validate_dataset()



def test_expression_agent_resolve_context(tmp_path):
    data_root = tmp_path / "data"
    exchange_dir = data_root / "binanceus"
    exchange_dir.mkdir(parents=True)
    (exchange_dir / "BTC_USDT-1h.feather").write_bytes(b"placeholder")

    config_data = {
        "datadir": str(data_root),
        "exchange": {
            "name": "binanceus",
            "pair_whitelist": ["BTC/USDT"],
        },
        "freqai": {
            "feature_parameters": {
                "include_timeframes": ["1h"],
                "label_period_candles": 12,
            },
            "train_period_days": 45,
            "backtest_period_days": 15,
        },
    }
    config_path = tmp_path / "config_freqai.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    feature_payload = {
        "label_period_candles": 12,
        "timeframe": "1h",
        "exchange": "binanceus",
        "pairs": ["BTC/USDT"],
        "features": [
            {
                "name": "feat_demo",
                "type": "sma_pct",
                "period": 5,
            }
        ],
    }
    feature_path = config_path.parent / expr.FEATURE_FILE.name
    feature_path.write_text(json.dumps(feature_payload), encoding="utf-8")

    context = resolve_feature_context(expr.FEATURE_FILE, None, config_path)

    assert context.feature_file == feature_path
    assert context.timeframe == "1h"
    assert context.settings is not None
    assert context.settings.data_dir == exchange_dir




def test_generate_llm_expressions_no_candidates(monkeypatch):
    df = _sample_dataframe()
    feature_cols = ["feat_test"]
    feature_cfg = {"pairs": ["BTC/USDT"], "exchange": "binanceus", "features": [{"name": "feat_test"}]}
    combos: list[dict] = []

    def _fake_request(prompt, config):  # noqa: ANN001
        return json.dumps({"expressions": []}), {}

    monkeypatch.setattr(expr.llm_utils, "request_completion", _fake_request)

    llm_args = SimpleNamespace(
        llm_count=1,
        llm_base_url="https://mock",
        llm_api_key="dummy",
        llm_model="gpt-3.5-turbo",
        llm_temperature=0.2,
        llm_max_tokens=128,
        llm_retries=1,
        llm_timeout=10,
    )

    with pytest.raises(ValueError):
        expr.generate_llm_expressions(
            df,
            feature_cols,
            feature_cfg,
            combos,
            timeframe="1h",
            label_period=3,
            score_method="correlation",
            complexity_penalty=0.01,
            backtest_weight=0.0,
            periods_per_year=8760,
            stability_windows=2,
            stability_min_samples=5,
            llm_options=llm_args,
        )


def test_training_pipeline_lightgbm(tmp_path):
    data_root = tmp_path / "data"
    exchange_dir = data_root / "binanceus"
    exchange_dir.mkdir(parents=True)

    df = _sample_dataframe(60)
    df[["date", "open", "high", "low", "close", "volume"]].to_feather(
        exchange_dir / "BTC_USDT-1h.feather"
    )

    feature_cfg = {
        "label_period_candles": 3,
        "timeframe": "1h",
        "exchange": "binanceus",
        "pairs": ["BTC/USDT"],
        "features": [
            {"name": "feat_rsi_14", "type": "rsi", "period": 14},
            {"name": "feat_return_z_5", "type": "return_zscore", "period": 5},
        ],
    }
    feature_path = tmp_path / "freqai_features.json"
    feature_path.write_text(json.dumps(feature_cfg), encoding="utf-8")

    config = {
        "data": {
            "feature_file": str(feature_path),
            "data_dir": str(data_root),
            "exchange": "binanceus",
            "pairs": ["BTC/USDT"],
            "timeframe": "1h",
            "label_period": 3,
        },
        "model": {
            "name": "lightgbm",
            "params": {
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": 0.1,
                "num_leaves": 15,
                "num_boost_round": 20,
            },
        },
        "training": {"validation_ratio": 0.2},
        "output": {"model_dir": str(tmp_path / "models")},
    }

    result = TrainingPipeline(config).run()
    assert result.model_path.exists()
    assert "rmse_train" in result.metrics

def test_training_pipeline_pytorch(tmp_path):
    data_root = tmp_path / "data"
    exchange_dir = data_root / "binanceus"
    exchange_dir.mkdir(parents=True)

    df = _sample_dataframe(80)
    df[["date", "open", "high", "low", "close", "volume"]].to_feather(
        exchange_dir / "BTC_USDT-1h.feather"
    )

    feature_cfg = {
        "label_period_candles": 3,
        "timeframe": "1h",
        "exchange": "binanceus",
        "pairs": ["BTC/USDT"],
        "features": [
            {"name": "feat_rsi_14", "type": "rsi", "period": 14},
            {"name": "feat_return_z_5", "type": "return_zscore", "period": 5},
        ],
    }
    feature_path = tmp_path / "freqai_features.json"
    feature_path.write_text(json.dumps(feature_cfg), encoding="utf-8")

    config = {
        "data": {
            "feature_file": str(feature_path),
            "data_dir": str(data_root),
            "exchange": "binanceus",
            "pairs": ["BTC/USDT"],
            "timeframe": "1h",
            "label_period": 3,
        },
        "model": {
            "name": "pytorch_mlp",
            "params": {
                "hidden_dims": [32, 16],
                "epochs": 5,
                "batch_size": 32,
                "learning_rate": 1e-3,
                "model_dir": str(tmp_path / "models_pt"),
            },
        },
        "training": {"validation_ratio": 0.25},
        "output": {"model_dir": str(tmp_path / "models_pt")},
    }

    result = TrainingPipeline(config).run()
    assert result.model_path.exists()
    assert "rmse_train" in result.metrics

def test_trading_env_step(tmp_path):
    gym = pytest.importorskip("gymnasium")
    data_root = tmp_path / "data"
    exchange_dir = data_root / "binanceus"
    exchange_dir.mkdir(parents=True)

    df = _sample_dataframe(40)
    df[["date", "open", "high", "low", "close", "volume"]].to_feather(
        exchange_dir / "BTC_USDT-1h.feather"
    )

    feature_cfg = {
        "label_period_candles": 3,
        "timeframe": "1h",
        "exchange": "binanceus",
        "pairs": ["BTC/USDT"],
        "features": [
            {"name": "feat_rsi_14", "type": "rsi", "period": 14},
            {"name": "feat_return_z_5", "type": "return_zscore", "period": 5},
        ],
    }
    feature_path = tmp_path / "freqai_features.json"
    feature_path.write_text(json.dumps(feature_cfg), encoding="utf-8")

    builder = FeatureDatasetBuilder(
        {
            "feature_file": str(feature_path),
            "data_dir": str(data_root),
            "exchange": "binanceus",
            "pairs": ["BTC/USDT"],
            "timeframe": "1h",
            "label_period": 3,
        }
    )
    dataset = builder.build()
    env = TradingEnv(dataset)

    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_rl_trainer_train(tmp_path):
    pytest.importorskip("stable_baselines3")
    data_root = tmp_path / "data"
    exchange_dir = data_root / "binanceus"
    exchange_dir.mkdir(parents=True)

    df = _sample_dataframe(80)
    df[["date", "open", "high", "low", "close", "volume"]].to_feather(
        exchange_dir / "BTC_USDT-1h.feather"
    )

    feature_cfg = {
        "label_period_candles": 3,
        "timeframe": "1h",
        "exchange": "binanceus",
        "pairs": ["BTC/USDT"],
        "features": [
            {"name": "feat_rsi_14", "type": "rsi", "period": 14},
            {"name": "feat_return_z_5", "type": "return_zscore", "period": 5},
        ],
    }
    feature_path = tmp_path / "freqai_features.json"
    feature_path.write_text(json.dumps(feature_cfg), encoding="utf-8")

    config = {
        "data": {
            "feature_file": str(feature_path),
            "data_dir": str(data_root),
            "exchange": "binanceus",
            "pairs": ["BTC/USDT"],
            "timeframe": "1h",
            "label_period": 3,
        },
        "training": {"total_timesteps": 200},
        "output": {"model_dir": str(tmp_path / "models_rl")},
    }

    result = RLTrainer(config).train()
    assert result.model_path.exists()
    assert result.timesteps == 200


def test_aux_model_predictor_handles_missing(tmp_path):
    df = _sample_dataframe()
    predictor = AuxModelPredictor(tmp_path)
    signals = predictor.augment(df.copy())
    assert signals == {}

def test_rl_trainer_writes_summary(tmp_path, monkeypatch):
    pytest.importorskip("gymnasium")

    data_root = tmp_path / "data"
    exchange_dir = data_root / "binanceus"
    exchange_dir.mkdir(parents=True)

    df = _sample_dataframe(80)
    df[["date", "open", "high", "low", "close", "volume"]].to_feather(
        exchange_dir / "BTC_USDT-1h.feather"
    )

    feature_cfg = {
        "timeframe": "1h",
        "exchange": "binanceus",
        "pairs": ["BTC/USDT"],
        "features": [
            {"name": "feat_rsi_14", "type": "rsi", "period": 14},
            {"name": "feat_return_z_5", "type": "return_zscore", "period": 5}
        ]
    }
    feature_path = tmp_path / "freqai_features.json"
    feature_path.write_text(json.dumps(feature_cfg), encoding="utf-8")

    calls: list[int] = []

    class DummyPPO:
        def __init__(self, policy, env, **kwargs):
            self.policy = SimpleNamespace(device="cpu")

        def learn(self, total_timesteps: int) -> None:
            calls.append(total_timesteps)

        def save(self, path: str | Path) -> None:
            Path(path).write_text("model", encoding="utf-8")

    monkeypatch.setattr('agent_market.freqai.rl.trainer.PPO', DummyPPO)

    config = {
        "data": {
            "feature_file": str(feature_path),
            "data_dir": str(data_root),
            "exchange": "binanceus",
            "pairs": ["BTC/USDT"],
            "timeframe": "1h",
            "label_period": 12
        },
        "policy": "MlpPolicy",
        "algo": {"learning_rate": 0.001},
        "training": {"total_timesteps": 500},
        "output": {"model_dir": str(tmp_path / "models_rl")}
    }

    trainer = RLTrainer(config)
    result = trainer.train()

    summary_path = Path(config["output"]["model_dir"]) / "training_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["timesteps"] == 500
    assert summary["model_path"] == str(result.model_path)
    assert summary["features"], "features should not be empty"

