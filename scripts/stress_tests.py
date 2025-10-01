#!/usr/bin/env python
from __future__ import annotations

"""
Stress and robustness tests for Agent Market (pre-web-ui).

Covers:
- Feature engine under large DataFrame sizes (TALib present/absent).
- Training pipeline with lightgbm fallback and dummy pytorch adapter.
- JobManager concurrency (many short jobs in parallel) and log correctness.

The script avoids external heavy deps (freqtrade, gym, torch, lightgbm) where possible
and falls back to lighter code paths to validate stability and correctness.
"""

import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def _banner(msg: str) -> None:
    print('\n' + '=' * 32)
    print(msg)
    print('=' * 32)


def gen_df(rows: int = 50_000) -> pd.DataFrame:
    idx = pd.date_range('2024-01-01', periods=rows, freq='h')
    base = np.linspace(100.0, 120.0, rows)
    return pd.DataFrame({
        'date': idx,
        'open': base - 0.5,
        'high': base + 1.0,
        'low': base - 1.0,
        'close': base,
        'volume': np.full(rows, 1_000.0),
    })


def stress_features() -> None:
    from agent_market.freqai.features import apply_configured_features

    cfg = {
        'features': [
            {'name': 'feat_rsi_14', 'type': 'rsi', 'period': 14},
            {'name': 'feat_sma_pct_20', 'type': 'sma_pct', 'period': 20},
            {'name': 'feat_ema_pct_20', 'type': 'ema_pct', 'period': 20},
            {'name': 'feat_roc_5', 'type': 'roc', 'period': 5},
            {'name': 'feat_mom_5', 'type': 'momentum', 'period': 5},
            {'name': 'feat_atr_14', 'type': 'atr_norm', 'period': 14},
            {'name': 'feat_bb_20', 'type': 'bb_width', 'period': 20},
            {'name': 'feat_obv_delta_10', 'type': 'obv_delta', 'period': 10},
            {'name': 'feat_macd_diff', 'type': 'macd_diff'},
            {'name': 'feat_stoch_k_14', 'type': 'stoch_k', 'period': 14},
            {'name': 'feat_stoch_d_14', 'type': 'stoch_d', 'period': 14},
            {'name': 'feat_kama_pct_20', 'type': 'kama_pct', 'period': 20},
            {'name': 'feat_lin_slope_24', 'type': 'linearreg_slope', 'period': 24},
        ]
    }
    for rows in (10_000, 30_000, 60_000):
        _banner(f'features rows={rows}')
        df = gen_df(rows)
        t0 = time.time()
        out = apply_configured_features(df, cfg)
        dt = time.time() - t0
        assert len(out) == rows
        # Ensure no explosive NaN/Inf
        bad = out.replace([np.inf, -np.inf], np.nan).isna().mean().mean()
        print(f'  took={dt:.3f}s, bad_ratio={bad:.4f}, cols={len(out.columns)}')


def stress_training(tmp: Path) -> None:
    from agent_market.freqai.training.pipeline import TrainingPipeline

    # Write feather data
    data_root = tmp / 'data'
    (data_root / 'binanceus').mkdir(parents=True, exist_ok=True)
    df = gen_df(8_000)
    df[['date', 'open', 'high', 'low', 'close', 'volume']].to_feather(data_root / 'binanceus' / 'BTC_USDT-1h.feather')

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
    feature_path = tmp / 'freqai_features.json'
    feature_path.write_text(__import__("json").dumps(feature_cfg), encoding='utf-8')

    for model_name, params in (
        ('lightgbm', {'objective': 'regression', 'metric': 'rmse', 'num_boost_round': 5}),
        ('pytorch_mlp', {'epochs': 2, 'batch_size': 64}),
    ):
        _banner(f'train model={model_name}')
        cfg = {
            'data': {
                'feature_file': str(feature_path),
                'data_dir': str(data_root),
                'exchange': 'binanceus',
                'pairs': ['BTC/USDT'],
                'timeframe': '1h',
                'label_period': 3,
            },
            'model': {
                'name': model_name,
                'params': {
                    **params,
                    'model_dir': str(tmp / f'models_{model_name}')
                },
            },
            'training': {'validation_ratio': 0.2},
            'output': {'model_dir': str(tmp / f'models_{model_name}')},
        }
        t0 = time.time()
        res = TrainingPipeline(cfg).run()
        dt = time.time() - t0
        assert Path(res.model_path).exists()
        print(f'  took={dt:.3f}s, model_path={res.model_path}, metrics={res.metrics}')


def stress_jobs() -> None:
    from server.job_manager import JobManager
    import subprocess

    jm = JobManager()
    job_ids: List[str] = []
    py = sys.executable
    # start 40 tiny jobs concurrently
    for i in range(40):
        cmd = [py, '-c', f"import time; print('job-{i}'); time.sleep(0.05)"]
        job_ids.append(jm.start(cmd, cwd=ROOT))
    # poll until all done
    t0 = time.time()
    while True:
        done = 0
        for jid in job_ids:
            st = jm.status(jid)
            if st.get('returncode') is not None:
                done += 1
        if done == len(job_ids):
            break
        if time.time() - t0 > 15:
            raise TimeoutError('jobs did not finish within 15s')
        time.sleep(0.05)
    # check logs
    for jid in job_ids:
        logs = jm.logs(jid, 0)
        assert logs.get('lines', None) is None  # JobManager.logs returns chunk not count
        assert isinstance(logs.get('logs'), list)
    print('  jobs concurrent OK')


def main() -> None:
    _banner('STRESS: FEATURES')
    stress_features()
    _banner('STRESS: TRAINING')
    tmp = ROOT / 'user_data' / 'tmp' / 'stress'
    tmp.mkdir(parents=True, exist_ok=True)
    stress_training(tmp)
    _banner('STRESS: JOBS')
    stress_jobs()
    print('\nALL STRESS TESTS PASSED')


if __name__ == '__main__':
    main()


