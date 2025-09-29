import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_market.agent_flow import AgentFlow, AgentFlowConfig, load_agent_flow_config


def _sample_dataframe(size: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=size, freq="h")
    close = np.linspace(100, 110, size)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(size, 1000.0),
        }
    )


def test_agent_flow_runs_sections(monkeypatch):
    calls = []

    monkeypatch.setattr(AgentFlow, 'run_feature_generation', lambda self, cfg: calls.append(('feature', cfg)))
    monkeypatch.setattr(AgentFlow, 'run_expression_generation', lambda self, cfg: calls.append(('expression', cfg)))
    monkeypatch.setattr(AgentFlow, 'run_ml_training', lambda self, cfg: calls.append(('ml', cfg)))
    monkeypatch.setattr(AgentFlow, 'run_rl_training', lambda self, cfg: calls.append(('rl', cfg)))
    monkeypatch.setattr(AgentFlow, 'run_backtest', lambda self, cfg: calls.append(('backtest', cfg)))

    cfg = AgentFlowConfig(
        feature={'args': ['--foo']},
        expression={'args': ['--bar']},
        ml_training={'config': {'model': {'name': 'lightgbm'}}},
        rl_training={'config': {'training': {'total_timesteps': 10}}},
        backtest={'command': ['echo', 'test']},
    )

    AgentFlow(cfg).run()
    assert [name for name, _ in calls] == ['feature', 'expression', 'ml', 'rl', 'backtest']


def test_agent_flow_backtest_feedback(tmp_path, monkeypatch):
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    feedback_path = tmp_path / 'feedback.json'
    commands: list[list[str]] = []

    def fake_run(cmd, cwd=None):  # noqa: ANN001
        commands.append(list(cmd))
        if 'backtesting' in cmd:
            data = {
                'strategy': {
                    'ExpressionLongStrategy': {
                        'profit_total_pct': 1.23,
                        'profit_total_abs': 12.3,
                        'trades': 10,
                        'profit_mean_pct': 0.5,
                        'winrate': 0.6,
                        'max_drawdown_abs': 3.0,
                        'best_pair': {'key': 'BTC/USDT'},
                        'worst_pair': {'key': 'ETH/USDT'},
                        'backtest_start': '2021-01-01',
                        'backtest_end': '2021-01-31',
                    }
                },
                'strategy_comparison': [
                    {'key': 'ExpressionLongStrategy', 'profit_total_pct': 1.23}
                ],
            }
            zip_path = results_dir / 'backtest-result-test.zip'
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr('backtest-result-test.json', json.dumps(data))
            (results_dir / '.last_result.json').write_text(json.dumps({'latest_backtest': zip_path.name}), encoding='utf-8')

    monkeypatch.setattr(AgentFlow, '_run_command', staticmethod(fake_run))

    cfg = AgentFlowConfig(
        expression={'args': [], 'feedback_path': str(feedback_path)},
        backtest={
            'command': ['freqtrade', 'backtesting'],
            'results_dir': str(results_dir),
            'feedback_path': str(feedback_path),
        },
    )

    AgentFlow(cfg).run()
    assert feedback_path.exists()
    summary = json.loads(feedback_path.read_text(encoding='utf-8'))
    assert summary['strategy'] == 'ExpressionLongStrategy'
    assert summary['trades'] == 10

    commands.clear()
    AgentFlow(AgentFlowConfig(expression={'args': [], 'feedback_path': str(feedback_path)})).run()
    expr_cmd = commands[0]
    assert '--feedback' in expr_cmd
    assert str(feedback_path) in expr_cmd


def test_agent_flow_ml_only(tmp_path):
    data_root = tmp_path / 'data'
    exchange_dir = data_root / 'binanceus'
    exchange_dir.mkdir(parents=True)
    df = _sample_dataframe()
    df.to_feather(exchange_dir / 'BTC_USDT-1h.feather')

    feature_cfg = {
        'label_period_candles': 3,
        'timeframe': '1h',
        'exchange': 'binanceus',
        'pairs': ['BTC/USDT'],
        'features': [
            {'name': 'feat_rsi_14', 'type': 'rsi', 'period': 14},
            {'name': 'feat_return_z_5', 'type': 'return_zscore', 'period': 5},
        ],
    }
    feature_path = tmp_path / 'freqai_features.json'
    feature_path.write_text(json.dumps(feature_cfg), encoding='utf-8')

    config = AgentFlowConfig(
        ml_training={
            'config': {
                'data': {
                    'feature_file': str(feature_path),
                    'data_dir': str(data_root),
                    'exchange': 'binanceus',
                    'pairs': ['BTC/USDT'],
                    'timeframe': '1h',
                    'label_period': 3,
                },
                'model': {
                    'name': 'lightgbm',
                    'params': {
                        'objective': 'regression',
                        'metric': 'rmse',
                        'learning_rate': 0.1,
                        'num_leaves': 15,
                        'num_boost_round': 20,
                        'model_dir': str(tmp_path / 'models_ml'),
                    },
                },
                'training': {'validation_ratio': 0.2},
                'output': {'model_dir': str(tmp_path / 'models_ml')},
            }
        }
    )

    AgentFlow(config).run(['ml'])
    assert any((tmp_path / 'models_ml').glob('*.txt'))


def test_load_agent_flow_config(tmp_path):
    payload = {
        'feature': {'args': []},
        'expression': None,
        'ml_training': {'config': {}},
    }
    config_path = tmp_path / 'agent_config.json'
    config_path.write_text(json.dumps(payload), encoding='utf-8')

    cfg = load_agent_flow_config(config_path)
    assert isinstance(cfg, AgentFlowConfig)
    assert cfg.feature == {'args': []}
    assert cfg.expression is None
    assert cfg.ml_training == {'config': {}}

    missing_path = tmp_path / 'missing.json'
    with pytest.raises(FileNotFoundError):
        load_agent_flow_config(missing_path)

def test_run_ml_training_with_configs(monkeypatch):
    calls: list[str] = []

    class DummyPipeline:
        def __init__(self, config):
            self.config = config

        def run(self) -> None:
            calls.append(self.config.get('model', {}).get('name'))

    monkeypatch.setattr('agent_market.agent_flow.TrainingPipeline', DummyPipeline)

    cfg = AgentFlowConfig(
        ml_training={
            'configs': [
                {'model': {'name': 'model_a'}, 'data': {}, 'training': {}, 'output': {}},
                {'model': {'name': 'model_b'}, 'data': {}, 'training': {}, 'output': {}},
            ]
        }
    )

    AgentFlow(cfg).run(['ml'])
    assert calls == ['model_a', 'model_b']
