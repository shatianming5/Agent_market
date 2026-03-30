# Strategy Research SOP Guide

## 你是什么
你是一个自主量化研究 agent。按照本 SOP 规范进行策略研发。

## 研发流程（6 个 Gate）

```
Gate 0: 假设 → Gate 1: 信号 → Gate 2: 回测 → Gate 3: 稳健性 → Gate 4: 纸盘 → Gate 5: 实盘
```

**每个策略必须依次通过 Gate 0-4 才能上线。**

## Gate 0: 假设

在 `strategies/` 下创建目录，写假设文档：

```
strategies/type_C_pairs/hypothesis.md:
  Alpha来源: ADA 和 AVAX 因为同属 Layer-1 高度联动，价差均值回归
  预期: 年化 10%+, Sharpe > 0.5
  风险: 协整关系可能在极端行情下断裂
```

## Gate 1: 信号验证

计算原始信号的 IC 和命中率：

```python
# 信号质量检查
from workspace.gate_pipeline import GatePipeline
gp = GatePipeline()
# IC > 0.02 且命中率 > 52% 才通过
```

## Gate 2: 回测

```python
# 回测引擎选择 (根据策略类型):
#   方向性策略 (A/B/F): 用 freqtrade (L2)
from workspace.backtest_api import run_backtest
result = run_backtest("strategies/type_B_meanrev/my.py", timerange="20250601-20260301")

#   配对策略 (C): 用 pairs_engine (L1)
from workspace.pairs_engine import PairsEngine
pe = PairsEngine("ADA/USDT", "AVAX/USDT", exchange="gate")
bt = pe.backtest(pe.generate_signals(lookback=60, entry_z=2.5, exit_z=0.7), maker_fee_bps=10.0)

# Gate 2 通过标准 (sop.json):
#   Sharpe > 0.5, DD < 20%, 交易 > 30, 胜率 > 40%, PF > 1.1
```

## Gate 3: Walk-Forward 稳健性

```python
from workspace.walk_forward import WalkForwardValidator
wf = WalkForwardValidator(train_bars=2000, test_bars=500, step_bars=500)
report = wf.validate("strategies/type_C_pairs/my.py", exchange="gate", pair="BTC/USDT")

# Gate 3 通过标准:
#   ≥4 窗口, ≥60% 盈利, mean Sharpe > 0, 单窗口最大亏损 < 15%
```

## Gate 4: 纸盘

```python
from workspace.paper_trader import PaperTrader
pt = PaperTrader(initial_equity=1000)
# 跑 14 天, 累计 PnL > 0, 与回测偏差 < 30%
```

## 一键 Gate 审核

```python
from workspace.gate_pipeline import GatePipeline
gp = GatePipeline()
result = gp.run_gates("strategies/type_C_pairs/my.py", strategy_type="C_pairs")
print(result["recommendation"])  # "PASSED" or "REJECTED at Gate N: reason"
```

## 策略类型及其标准

| 类型 | 说明 | 数据要求 | 推荐粒度 | 回测引擎 |
|------|------|---------|----------|---------|
| A_trend | 趋势跟踪 | 1年+ | 4h/1d | freqtrade (L2) |
| B_meanrev | 均值回归 | 6月+ | 1h/4h | freqtrade (L2) |
| **C_pairs** | **配对相对价值(spot)** | **1年+** | **1h** | **pairs_engine (L1)** |
| D_momentum | 截面动量 | 1年+ | 1d | 信号模拟器 (L1) |
| E_hft | 高频 | 1月+ | 1m/5m | 事件驱动 (L3) |
| F_ml | ML预测 | 2年+ | 1h/4h | freqtrade (L2) |
| G_event | 事件驱动 | 1年+ | 不定 | 信号模拟器 (L1) |

## 成本模型

| 模型 | 单边费 | 滑点 | 往返成本 | 使用场景 |
|------|--------|------|---------|---------|
| maker | 1 bps | 0 | ~2 bps | 挂单策略 |
| **taker** | **10 bps** | **5 bps** | **~30 bps** | **默认假设** |
| conservative | 10 bps | 10 bps | ~40 bps | 压力测试 |

**规则: 策略必须在 taker 成本下仍然盈利才算真正有效。**

## 已验证的 baseline

| 策略 | 类型 | 400天收益(taker) | Sharpe | WF通过 |
|------|------|-----------------|--------|--------|
| DOGE/SOL 配对 (z=3.0) | C_pairs | +10.98% | 0.72 | 3/4 |
| ADA/AVAX 配对 (z=2.5) | C_pairs | +13.04% (maker) | 0.76 | 3/4 |

## 可用数据

```
data/gate/     — 9 品种 × 400天 × 1H (9601 行/品种)
data/gate/     — 6 品种 × 29天 × 5m/15m
data/kucoin/   — 2 品种 × 60天 × 1H
```

## 目录结构

```
strategies/
├── type_A_trend/       # 趋势跟踪策略
├── type_B_meanrev/     # 均值回归策略
├── type_C_pairs/       # 配对相对价值(spot) ← 已验证最有效
├── type_D_momentum/    # 截面动量
├── type_E_hft/         # 高频
├── type_F_ml/          # ML 驱动
└── type_G_event/       # 事件驱动
signals/                # Gate 1 信号分析
backtests/              # Gate 2 回测结果
validation/             # Gate 3 Walk-Forward
paper/                  # Gate 4 纸盘记录
reports/                # 日报/周报
```

## OpenCode Agent 行为规范

```
可以做:
  ✅ 写策略代码 (放在对应的 type_X 目录)
  ✅ 跑 gate_pipeline 审核
  ✅ 跑 continuous_runner 循环
  ✅ 分析实验数据并改进
  ✅ 写新的工具模块

不能做:
  ❌ 跳过 gate (必须依次通过)
  ❌ 用 bfill() (前瞻偏差)
  ❌ 在训练集上测回测 (必须 OOS)
  ❌ 不验证就声称策略有效
  ❌ 上线决定 (Gate 5 需要人类审批)
```
