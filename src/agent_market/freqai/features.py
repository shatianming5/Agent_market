from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from pandas import DataFrame

# Try TA-Lib first; if not available, fallback to pandas_ta implementations where possible.
try:  # pragma: no cover - runtime dependency
    import talib.abstract as ta  # type: ignore
    _HAS_TALIB = True
except Exception:  # pragma: no cover
    ta = None  # type: ignore
    _HAS_TALIB = False
    try:
        import pandas_ta as pta  # type: ignore
    except Exception:
        pta = None  # type: ignore


def apply_configured_features(dataframe: DataFrame, feature_cfg: Dict) -> DataFrame:
    features = feature_cfg.get('features', [])
    if not features:
        return dataframe

    for feat in features:
        name = feat.get('name') or f"feat_{feat.get('type')}_{feat.get('period')}"
        if name in dataframe.columns:
            continue
        kind = (feat.get('type') or '').lower()
        raw_period = feat.get('period')
        period = int(raw_period) if raw_period not in (None, '', 'None') else None
        requires_period = kind not in {'macd_diff', 'psar_ratio'}
        if requires_period and (period is None or period <= 0):
            continue
        try:
            if kind == 'rsi':
                if _HAS_TALIB:
                    dataframe[name] = ta.RSI(dataframe, timeperiod=period)  # type: ignore
                elif pta is not None:
                    dataframe[name] = pta.rsi(dataframe['close'], length=period)
                else:
                    raise ImportError('RSI requires TA-Lib or pandas_ta')
            elif kind == 'mfi':
                if _HAS_TALIB:
                    dataframe[name] = ta.MFI(dataframe, timeperiod=period)  # type: ignore
                elif pta is not None:
                    dataframe[name] = pta.mfi(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'], length=period)
                else:
                    raise ImportError('MFI requires TA-Lib or pandas_ta')
            elif kind == 'adx':
                if _HAS_TALIB:
                    dataframe[name] = ta.ADX(dataframe, timeperiod=period)  # type: ignore
                elif pta is not None:
                    adx = pta.adx(dataframe['high'], dataframe['low'], dataframe['close'], length=period)
                    col = next((c for c in adx.columns if c.startswith('ADX_')), adx.columns[-1])
                    dataframe[name] = adx[col]
                else:
                    raise ImportError('ADX requires TA-Lib or pandas_ta')
            elif kind == 'cci':
                if _HAS_TALIB:
                    dataframe[name] = ta.CCI(dataframe, timeperiod=period)  # type: ignore
                elif pta is not None:
                    dataframe[name] = pta.cci(dataframe['high'], dataframe['low'], dataframe['close'], length=period)
                else:
                    raise ImportError('CCI requires TA-Lib or pandas_ta')
            elif kind == 'cmo':
                if _HAS_TALIB:
                    dataframe[name] = ta.CMO(dataframe, timeperiod=period)  # type: ignore
                elif pta is not None:
                    dataframe[name] = pta.cmo(dataframe['close'], length=period)
                else:
                    raise ImportError('CMO requires TA-Lib or pandas_ta')
            elif kind == 'ema_pct':
                ema = ta.EMA(dataframe, timeperiod=period) if _HAS_TALIB else (pta.ema(dataframe['close'], length=period) if pta is not None else None)  # type: ignore
                dataframe[name] = ema / (dataframe['close'] + 1e-9) - 1
            elif kind == 'sma_pct':
                sma = ta.SMA(dataframe, timeperiod=period) if _HAS_TALIB else (pta.sma(dataframe['close'], length=period) if pta is not None else None)  # type: ignore
                dataframe[name] = sma / (dataframe['close'] + 1e-9) - 1
            elif kind == 'wma_pct':
                wma = ta.WMA(dataframe, timeperiod=period) if _HAS_TALIB else (pta.wma(dataframe['close'], length=period) if pta is not None else None)  # type: ignore
                dataframe[name] = wma / (dataframe['close'] + 1e-9) - 1
            elif kind == 'tema_pct':
                tema = ta.TEMA(dataframe, timeperiod=period) if _HAS_TALIB else (pta.tema(dataframe['close'], length=period) if pta is not None else None)  # type: ignore
                dataframe[name] = tema / (dataframe['close'] + 1e-9) - 1
            elif kind == 'vwap_pct':
                typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3.0
                price_volume = (typical_price * dataframe['volume']).rolling(period, min_periods=max(1, period // 2)).sum()
                volume_sum = dataframe['volume'].rolling(period, min_periods=max(1, period // 2)).sum()
                dataframe[name] = (price_volume / (volume_sum + 1e-9)) / (dataframe['close'] + 1e-9) - 1
            elif kind == 'roc':
                dataframe[name] = (ta.ROC(dataframe, timeperiod=period) if _HAS_TALIB else (pta.roc(dataframe['close'], length=period) if pta is not None else None))  # type: ignore
            elif kind == 'momentum':
                dataframe[name] = (ta.MOM(dataframe, timeperiod=period) if _HAS_TALIB else (pta.mom(dataframe['close'], length=period) if pta is not None else None))  # type: ignore
            elif kind == 'atr_norm':
                atr = (ta.ATR(dataframe, timeperiod=period) if _HAS_TALIB else (pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=period) if pta is not None else None))  # type: ignore
                dataframe[name] = atr / (dataframe['close'] + 1e-9)
            elif kind == 'bb_width':
                if _HAS_TALIB:
                    upper, middle, lower = ta.BBANDS(dataframe, timeperiod=period, nbdevup=2, nbdevdn=2)  # type: ignore
                else:
                    bb = pta.bbands(dataframe['close'], length=period, std=2.0) if pta is not None else None
                    if bb is None:
                        raise ImportError('BBANDS requires TA-Lib or pandas_ta')
                    lower, middle, upper = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
                dataframe[name] = (upper - lower) / (middle + 1e-9)
            elif kind == 'range_pct':
                dataframe[name] = (dataframe['high'] - dataframe['low']) / (dataframe['close'] + 1e-9)
            elif kind == 'volume_ratio':
                rolling_vol = dataframe['volume'].rolling(period, min_periods=max(1, period // 2)).mean()
                dataframe[name] = dataframe['volume'] / (rolling_vol + 1e-9)
            elif kind == 'return_zscore':
                returns = dataframe['close'].pct_change()
                rolling_mean = returns.rolling(period, min_periods=max(2, period // 2)).mean()
                rolling_std = returns.rolling(period, min_periods=max(2, period // 2)).std(ddof=0)
                dataframe[name] = (returns - rolling_mean) / (rolling_std + 1e-9)
            elif kind == 'cmf':
                price_range = (dataframe['high'] - dataframe['low']).replace(0, np.nan)
                multiplier = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / price_range
                money_flow_volume = multiplier.fillna(0.0) * dataframe['volume']
                mfv_sum = money_flow_volume.rolling(period, min_periods=max(1, period // 2)).sum()
                volume_sum = dataframe['volume'].rolling(period, min_periods=max(1, period // 2)).sum()
                dataframe[name] = mfv_sum / (volume_sum + 1e-9)
            elif kind == 'obv_delta':
                obv = (ta.OBV(dataframe) if _HAS_TALIB else (pta.obv(dataframe['close'], dataframe['volume']) if pta is not None else None))  # type: ignore
                base = obv.shift(period).abs() if period else obv.shift(1).abs()
                dataframe[name] = obv.diff(period or 1) / (base + 1e-9)
            elif kind == 'ema':
                dataframe[name] = (ta.EMA(dataframe, timeperiod=period) if _HAS_TALIB else (pta.ema(dataframe['close'], length=period) if pta is not None else None))  # type: ignore
            elif kind == 'macd_diff':
                if _HAS_TALIB:
                    macd_df = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)  # type: ignore
                    dataframe[name] = macd_df['macd'] - macd_df['macdsignal']
                else:
                    m = pta.macd(dataframe['close'], fast=12, slow=26, signal=9) if pta is not None else None
                    if m is None:
                        raise ImportError('MACD requires TA-Lib or pandas_ta')
                    dataframe[name] = m.iloc[:, 0] - m.iloc[:, 2]
            elif kind == 'stoch_k':
                if _HAS_TALIB:
                    stoch = ta.STOCH(dataframe, fastk_period=period, slowk_period=3, slowd_period=3)  # type: ignore
                    dataframe[name] = stoch['slowk']
                else:
                    st = pta.stoch(dataframe['high'], dataframe['low'], dataframe['close'], k=period, d=3, smooth_k=3) if pta is not None else None
                    if st is None:
                        raise ImportError('STOCH requires TA-Lib or pandas_ta')
                    dataframe[name] = st.iloc[:, 0]
            elif kind == 'stoch_d':
                if _HAS_TALIB:
                    stoch = ta.STOCH(dataframe, fastk_period=period, slowk_period=3, slowd_period=3)  # type: ignore
                    dataframe[name] = stoch['slowd']
                else:
                    st = pta.stoch(dataframe['high'], dataframe['low'], dataframe['close'], k=period, d=3, smooth_k=3) if pta is not None else None
                    if st is None:
                        raise ImportError('STOCH requires TA-Lib or pandas_ta')
                    dataframe[name] = st.iloc[:, 1]
            elif kind == 'psar_ratio':
                if _HAS_TALIB:
                    sar = ta.SAR(dataframe)  # type: ignore
                else:
                    ps = pta.psar(dataframe['high'], dataframe['low'], dataframe['close']) if pta is not None else None
                    if ps is None:
                        raise ImportError('PSAR requires TA-Lib or pandas_ta')
                    sar = ps.iloc[:, 0].combine_first(ps.iloc[:, 1])
                dataframe[name] = sar / (dataframe['close'] + 1e-9) - 1
            elif kind == 'kama_pct':
                kama = (ta.KAMA(dataframe, timeperiod=period) if _HAS_TALIB else (pta.kama(dataframe['close'], length=period) if pta is not None else None))  # type: ignore
                dataframe[name] = kama / (dataframe['close'] + 1e-9) - 1
            elif kind == 'linearreg_slope':
                if _HAS_TALIB:
                    dataframe[name] = ta.LINEARREG_SLOPE(dataframe, timeperiod=period)  # type: ignore
                else:
                    dataframe[name] = dataframe['close'].diff().rolling(period, min_periods=max(2, period // 2)).mean()
            elif kind == 'realized_vol':
                returns = dataframe['close'].pct_change()
                dataframe[name] = returns.rolling(period, min_periods=max(2, period // 2)).std(ddof=0) * np.sqrt(24 * 365)
            elif kind == 'return_skew':
                returns = dataframe['close'].pct_change()
                dataframe[name] = returns.rolling(period, min_periods=max(3, period // 2)).skew()
            elif kind == 'volume_zscore':
                volume = dataframe['volume']
                rolling_mean = volume.rolling(period, min_periods=max(2, period // 2)).mean()
                rolling_std = volume.rolling(period, min_periods=max(2, period // 2)).std(ddof=0)
                dataframe[name] = (volume - rolling_mean) / (rolling_std + 1e-9)
            elif kind == 'price_zscore':
                close = dataframe['close']
                rolling_mean = close.rolling(period, min_periods=max(2, period // 2)).mean()
                rolling_std = close.rolling(period, min_periods=max(2, period // 2)).std(ddof=0)
                dataframe[name] = (close - rolling_mean) / (rolling_std + 1e-9)
            elif kind == 'donchian_width':
                rolling_high = dataframe['high'].rolling(period, min_periods=max(1, period // 2)).max()
                rolling_low = dataframe['low'].rolling(period, min_periods=max(1, period // 2)).min()
                dataframe[name] = (rolling_high - rolling_low) / (dataframe['close'] + 1e-9)
            elif kind == 'vwap_slope':
                typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3.0
                price_volume = (typical_price * dataframe['volume']).rolling(period, min_periods=max(1, period // 2)).sum()
                volume_sum = dataframe['volume'].rolling(period, min_periods=max(1, period // 2)).sum()
                vwap = price_volume / (volume_sum + 1e-9)
                dataframe[name] = vwap.diff().rolling(period, min_periods=max(1, period // 2)).mean()
            else:
                continue
        except Exception:
            continue
        dataframe[name] = dataframe[name].replace([np.inf, -np.inf], np.nan)

    combos = feature_cfg.get('feature_combos', [])
    local_dict = {col: dataframe[col] for col in dataframe.columns}
    for combo in combos:
        name = combo.get('name')
        formula = combo.get('formula')
        if not name or name in dataframe.columns or not formula:
            continue
        try:
            result = eval(formula, {'__builtins__': {}}, local_dict)
            series = _ensure_series(result, dataframe.index).astype(float)
            series = series.replace([np.inf, -np.inf], np.nan)
            dataframe[name] = series
            local_dict[name] = series
        except Exception:
            continue
    return dataframe


def _ensure_series(data, index) -> pd.Series:
    if isinstance(data, pd.Series):
        return data
    return pd.Series(data, index=index)

