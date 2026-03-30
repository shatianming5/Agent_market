# Freqtrade Strategy Templates — OpenCode Reference

OpenCode: 当你需要写策略时，从这里复制模板并修改。
每个模板都是真正的 freqtrade IStrategy，已通过 L2 回测验证。

## A. 技术指标策略 (type_B_meanrev)

```python
from freqtrade.strategy import IStrategy
class MyMeanRev(IStrategy):
    timeframe = "1h"; can_short = False
    minimal_roi = {"0": 0.08, "120": 0.03}; stoploss = -0.04
    trailing_stop = True; startup_candle_count = 30

    def populate_indicators(self, df, metadata):
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        sma = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        df["bb_lower"] = sma - 2 * std
        df["bb_middle"] = sma
        return df

    def populate_entry_trend(self, df, metadata):
        df.loc[(df["close"] < df["bb_lower"]) & (df["rsi"] < 35), ["enter_long", "enter_tag"]] = (1, "oversold")
        return df

    def populate_exit_trend(self, df, metadata):
        df.loc[(df["close"] > df["bb_middle"]) | (df["rsi"] > 65), "exit_long"] = 1
        return df
```

## B. 配对交易策略 (type_C_pairs) — 需要 informative_pairs

关键：用 `informative_pairs()` 获取参考品种数据，在 `populate_indicators` 中计算 spread z-score。

```python
from freqtrade.strategy import IStrategy
import numpy as np
from pandas import DataFrame

class MyPairsStrategy(IStrategy):
    timeframe = "1h"; can_short = False
    minimal_roi = {"0": 0.03, "120": 0.015}; stoploss = -0.035
    trailing_stop = True; startup_candle_count = 100

    pair_b = "SOL/USDT"  # 参考品种
    lookback = 80; entry_z = 3.0; exit_z = 0.5

    def informative_pairs(self):
        return [(self.pair_b, self.timeframe)]  # 必须声明!

    def populate_indicators(self, df, metadata):
        inf = self.dp.get_pair_dataframe(pair=self.pair_b, timeframe=self.timeframe)
        if inf.empty: df["zscore"] = 0; return df
        inf = inf[["date", "close"]].rename(columns={"close": "close_b"})
        df = df.merge(inf, on="date", how="left")
        df["close_b"] = df["close_b"].ffill().fillna(0)
        log_a = np.log(df["close"].values + 1e-10)
        log_b = np.log(df["close_b"].values + 1e-10)
        n = len(df); hr = np.full(n, np.nan)
        for i in range(self.lookback, n):
            la, lb = log_a[i-self.lookback:i+1], log_b[i-self.lookback:i+1]
            try: hr[i] = np.linalg.lstsq(np.column_stack([lb, np.ones(len(lb))]), la, rcond=None)[0][0]
            except: pass
        spread = log_a - hr * log_b
        sm = DataFrame({"s": spread})["s"].rolling(self.lookback).mean().values
        ss = DataFrame({"s": spread})["s"].rolling(self.lookback).std().values + 1e-10
        df["zscore"] = (spread - sm) / ss
        return df

    def populate_entry_trend(self, df, metadata):
        df.loc[(df["zscore"] < -self.entry_z), ["enter_long", "enter_tag"]] = (1, "pairs_cheap")
        return df

    def populate_exit_trend(self, df, metadata):
        df.loc[(df["zscore"].abs() < self.exit_z) | (df["zscore"] > self.entry_z), "exit_long"] = 1
        return df
```

Config 必须: `pair_whitelist = ["DOGE/USDT"]` (只交易主品种)

## C. 篮子动量策略 (type_D_momentum) — 多品种 informative_pairs

关键：`informative_pairs()` 返回所有品种，`populate_indicators` 中计算 cross-sectional rank。

```python
class MyBasketStrategy(IStrategy):
    timeframe = "1h"; can_short = False; startup_candle_count = 200
    universe = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT"]
    top_n = 3; momentum_lookback = 168

    def informative_pairs(self):
        return [(p, self.timeframe) for p in self.universe]

    def populate_indicators(self, df, metadata):
        current = metadata["pair"]
        returns = {}
        for pair in self.universe:
            if pair == current:
                returns[pair] = df["close"].pct_change(self.momentum_lookback)
            else:
                inf = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
                if not inf.empty:
                    merged = df[["date"]].merge(inf[["date","close"]].rename(columns={"close":"c"}), on="date", how="left")
                    returns[pair] = merged["c"].pct_change(self.momentum_lookback)
        # Rank current pair
        # ... (see full template in type_D_momentum/)
```

Config 必须: `pair_whitelist = [全部品种]`, `max_open_trades = 3`

## D. ML 策略 (type_F_ml) — 加载训练好的模型

关键：在 `populate_indicators` 中加载模型并调用 `predict()`。

```python
class MyMLStrategy(IStrategy):
    # 直接用 FreqtradeMLStrategy — 它是通用的
    # 自动发现最新模型，加载特征+表达式+预测
    # 支持: LightGBM (.txt), XGBoost, PyTorch (.pt), pickle (.pkl)
    pass  # 继承 FreqtradeMLStrategy 即可
```

或者自定义:
```python
def populate_indicators(self, df, metadata):
    import lightgbm as lgb
    model = lgb.Booster(model_file="artifacts/models/lightgbm_real/lightgbm_model.txt")
    # 计算特征...
    df["ml_pred"] = model.predict(feature_matrix)
    return df
```

## 回测命令

```bash
# 生成配置
python3 -c "from workspace.freqtrade_config_gen import generate_config; generate_config('pairs_doge_sol', pairs=['DOGE/USDT'])"

# 回测
python3 scripts/freqtrade_cli.py backtesting \
  --userdir user_data \
  --config workspace/configs/ft_pairs_doge_sol_gate.json \
  --strategy MyPairsStrategy \
  --strategy-path workspace/strategies/type_C_pairs \
  --timerange 20250601-20260301 \
  --cache none
```
