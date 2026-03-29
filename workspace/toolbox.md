# OpenCode Quant Workspace — Toolbox

## 你可以做什么

你是一个自主量化研究员。你可以：

1. **写算法** — 在 `workspace/models/` 下创建 .py 文件，实现任何 ML/DL/统计模型
2. **写策略** — 在 `workspace/strategies/` 下创建 freqtrade 策略 .py 文件
3. **写配置** — 在 `workspace/configs/` 下创建 freqtrade JSON 配置
4. **调用回测** — `from workspace.backtest_api import run_backtest`
5. **评估结果** — `from workspace.evaluator import evaluate`
6. **追踪实验** — `from workspace.tracker import record_experiment, query_best`

## 可用数据

| 文件 | 内容 | 列 |
|------|------|----|
| `user_data/data/kucoin/BTC_USDT-1h.feather` | BTC 1小时K线 | date, open, high, low, close, volume |
| `user_data/data/kucoin/ETH_USDT-1h.feather` | ETH 1小时K线 | 同上 |

时间范围：约 1448 行（~60天）

## 可用的特征工程工具

```python
# 技术指标（已内置于 freqai features）
from agent_market.freqai.features import apply_configured_features

# 表达式引擎（安全执行数学表达式）
from agent_market.freqai.expression_engine import safe_eval_expression, apply_expressions

# 微观结构特征（OHLCV 模式）
from agent_market.microstructure.ohlcv_features import build_ohlcv_micro_features

# Factor Compiler（类型安全的因子编译）
from agent_market.factor_compiler.dsl.parser import parse_formula
from agent_market.factor_compiler.dsl.types import typecheck
```

## 写模型的规范

继承 `BaseModelAdapter`，实现 4 个方法：

```python
from agent_market.freqai.model.base import BaseModelAdapter, TrainResult

class MyModel(BaseModelAdapter):
    registry_name = "my_model"  # 唯一名称

    def fit(self, X_train, y_train, X_valid=None, y_valid=None) -> TrainResult:
        # 训练逻辑
        # 返回 TrainResult(model_path=Path, metrics=dict)

    def predict(self, X) -> np.ndarray:
        # 推理逻辑

    def save(self, path): ...
    def load(self, path): ...
```

## 写策略的规范

继承 `IStrategy`，实现 3 个方法：

```python
from freqtrade.strategy import IStrategy

class MyStrategy(IStrategy):
    timeframe = "1h"
    minimal_roi = {"0": 0.1, "240": -1}
    stoploss = -0.05

    def populate_indicators(self, dataframe, metadata):
        # 添加指标列
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        # 设置 enter_long = 1 的条件
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        # 设置 exit_long = 1 的条件
        return dataframe
```

## 回测 API

```python
from workspace.backtest_api import run_backtest

result = run_backtest(
    strategy_path="workspace/strategies/my_strategy.py",
    strategy_name="MyStrategy",  # 类名
    pairs=["BTC/USDT", "ETH/USDT"],
    timerange="20251126-20260125",
)
# result = {"ok": True, "sharpe": 0.72, "sortino": 1.13, ...}
# 或者 {"ok": False, "error": "错误信息"}
```

## 优化目标（workspace/objectives.json）

| 指标 | 方向 | 目标 | 权重 |
|------|------|------|------|
| Sharpe | 越高越好 | ≥ 1.0 | 30% |
| Max Drawdown | 越低越好 | ≤ 5% | 20% |
| Profit % | 越高越好 | ≥ 1% | 20% |
| Sortino | 越高越好 | ≥ 1.0 | 10% |
| Win Rate | 越高越好 | ≥ 45% | 10% |
| Profit Factor | 越高越好 | ≥ 1.1 | 10% |
