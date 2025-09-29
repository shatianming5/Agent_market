import json
from pathlib import Path

import numpy as np
import pandas as pd
import talib.abstract as ta

from freqtrade.user_data.strategies.FreqAIExampleStrategy import (
    FreqAIExampleStrategy,
    load_expression_config,
    load_feature_config,
)

strategy = FreqAIExampleStrategy()
feature_cfg = load_feature_config()
expr_cfg = load_expression_config()

# Load OHLCV data for BTC/USDT 1h
pair_path = Path('freqtrade/user_data/data/binanceus/BTC_USDT-1h.feather')
df = pd.read_feather(pair_path)

# Apply features and expressions using strategy helpers
df = strategy._apply_generated_features(df.copy(), feature_cfg)
df, expr_cols, expr_meta = strategy._apply_generated_expressions(df, expr_cfg)

entry_votes = []
exit_votes = []
signal_components = []
entry_weights = []
exit_weights = []
expectations = []

for name in expr_cols:
    meta = expr_meta.get(name, {})
    z_col = f'{name}_z'
    if z_col not in df.columns:
        series = df[name]
        mean_ref = meta.get('signal_mean')
        if not isinstance(mean_ref, (int, float)) or not np.isfinite(mean_ref):
            mean_ref = float(series.mean(skipna=True))
        std_ref = meta.get('signal_std')
        if not isinstance(std_ref, (int, float)) or not np.isfinite(std_ref) or abs(std_ref) < 1e-9:
            std_ref = float(series.std(skipna=True, ddof=0))
            if not np.isfinite(std_ref) or abs(std_ref) < 1e-9:
                std_ref = 1.0
        z_series = (series - mean_ref) / (std_ref + 1e-9)
        df[z_col] = z_series.replace([np.inf, -np.inf], np.nan)
    z_values = df[z_col].fillna(0.0)
    entry_thr = meta.get('entry_threshold') or 0.0
    exit_thr = meta.get('exit_threshold') or 0.0
    entry_weight = meta.get('entry_expectation')
    if not isinstance(entry_weight, (int, float)) or not np.isfinite(entry_weight) or entry_weight <= 0:
        entry_weight = 0.01
    entry_weights.append(float(entry_weight))
    entry_votes.append((z_values >= entry_thr).astype(float) * float(entry_weight))
    signal_components.append(z_values * float(entry_weight))
    exit_weight = meta.get('exit_expectation')
    if not isinstance(exit_weight, (int, float)) or not np.isfinite(exit_weight):
        exit_weight = 0.01
    else:
        exit_weight = abs(exit_weight) or 0.01
    exit_weights.append(float(exit_weight))
    exit_votes.append((z_values <= exit_thr).astype(float) * float(exit_weight))
    expectation = meta.get('entry_expectation')
    if isinstance(expectation, (int, float)) and np.isfinite(expectation):
        expectations.append(float(expectation))

weight_sum = float(np.sum(entry_weights)) if entry_weights else 0.0
exit_weight_sum = float(np.sum(exit_weights)) if exit_weights else 0.0
if weight_sum > 0 and entry_votes:
    entry_df = pd.concat(entry_votes, axis=1)
    df['llm_entry_ratio'] = (entry_df.sum(axis=1) / weight_sum).clip(0, 1)
    signal_df = pd.concat(signal_components, axis=1)
    df['llm_signal'] = signal_df.sum(axis=1) / weight_sum
else:
    df['llm_entry_ratio'] = 0.0
    df['llm_signal'] = 0.0
if exit_weight_sum > 0 and exit_votes:
    exit_df = pd.concat(exit_votes, axis=1)
    df['llm_exit_ratio'] = (exit_df.sum(axis=1) / exit_weight_sum).clip(0, 1)
else:
    df['llm_exit_ratio'] = 0.0

df['llm_expectation'] = float(np.mean(expectations)) if expectations else 0.0

df['ema_fast'] = ta.EMA(df, timeperiod=55)
df['ema_slow'] = ta.EMA(df, timeperiod=200)
df['trend_up'] = df['ema_fast'] > df['ema_slow']

entry_threshold = float(strategy.long_signal_threshold.value)
print('Entry threshold', entry_threshold)
print('llm_signal describe', df['llm_signal'].describe())
print('llm_entry_ratio describe', df['llm_entry_ratio'].describe())
print('llm_expectation unique', df['llm_expectation'].unique()[:5])
print('trend_up ratio', df['trend_up'].mean())

default_prediction = df.get('prediction')
if default_prediction is not None:
    print('prediction describe', default_prediction.describe())

mask = (
    df['trend_up']
    & (df['llm_signal'] >= entry_threshold)
    & (df['llm_expectation'] > 0)
)
print('Signals count', int(mask.sum()), 'of', len(df))
print('First indices where signal True', df.index[mask][:10].tolist())
