import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import freqtrade.scripts.freqai_expression_agent as expr


def _sample_dataframe(size: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=size, freq="h")
    close = np.linspace(100, 110, size)
    df = pd.DataFrame(
        {
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

    # Missing文件触发异常
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

    args = SimpleNamespace(
        feature_file=expr.FEATURE_FILE,
        timeframe=None,
        config=config_path,
    )

    feature_file, timeframe, settings = expr._resolve_cli_context(args)

    assert feature_file == feature_path
    assert timeframe == "1h"
    assert settings is not None
    assert settings.data_dir == exchange_dir

