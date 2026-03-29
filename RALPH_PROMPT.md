# Ralph Loop: 自主量化策略研发迭代

## 目标
利用已搭建的 workspace 系统，自主研发出至少一个在回测中盈利（Sharpe > 0.5）的交易策略。

## 已有工具（直接调用，不需要重新实现）
```python
import sys; sys.path.insert(0, ".")
from workspace.backtest_api import run_backtest        # 回测 API
from workspace.evaluator import evaluate               # 多目标评估
from workspace.tracker import record_experiment, query_best, compare  # 实验追踪
from workspace.orchestrator import run_experiment      # 一键实验
from workspace.model_loader import scan_and_register   # 动态模型加载
```

## 已有数据
- user_data/data/kucoin/BTC_USDT-1h.feather (1448 行，~60天)
- user_data/data/kucoin/ETH_USDT-1h.feather (1448 行，~60天)
- 时间范围：20251126-20260125
- 市场环境：BTC 期间约 -2.47%（震荡偏空）

## 每轮迭代必须执行的完整 checklist

### CHECK-1: 分析历史实验
```python
from workspace.tracker import list_experiments, query_best
exps = list_experiments()
best = query_best("sharpe", 3)
# 打印历史最佳和最差，分析失败原因
```

### CHECK-2: 设计新策略假设
基于历史实验的反馈，提出一个新的策略假设：
- 为什么之前的策略亏损？
- 新策略解决了什么问题？
- 预期在什么市场环境下有效？

### CHECK-3: 编写策略代码
在 workspace/strategies/ 下创建新的 .py 文件：
- 必须继承 IStrategy
- 必须实现 populate_indicators / populate_entry_trend / populate_exit_trend
- 策略名称必须唯一
- 代码必须语法正确（先 python3 -c "import ast; ast.parse(open('file').read())" 验证）

### CHECK-4: 语法验证
```bash
python3 -c "import ast; ast.parse(open('workspace/strategies/NEW.py').read()); print('SYNTAX OK')"
```

### CHECK-5: 回测执行
```python
result = run_backtest("workspace/strategies/NEW.py", "ClassName", timerange="20251126-20260125")
assert result["ok"], f"Backtest failed: {result.get('error')}"
```

### CHECK-6: 多目标评估
```python
score = evaluate(result)
print(f"Score: {score['total_score']}/100 ({score['grade']})")
# 逐项检查每个目标
for name, d in score["details"].items():
    print(f"  {name}: {d['value']} | target={d['target']} | met={d['met']}")
```

### CHECK-7: 记录实验
```python
record = record_experiment(
    backtest_result=result,
    evaluation=score,
    strategy_name="...",
    notes="策略假设和设计理由",
    tags=["iteration_N"],
)
```

### CHECK-8: 对比历史最佳
```python
if len(list_experiments()) >= 2:
    c = compare(record["id"], best_id)
    print(f"vs best: winner=#{c['winner']}")
```

### CHECK-9: 分析改进方向
根据 evaluator 的 suggestions 和当前弱项，确定下一轮改进方向：
- Sharpe 低 → 改信号质量
- DD 高 → 加止损/仓位管理
- 胜率低 → 更严格的过滤条件
- PF 低 → 让赢家跑更久/亏家砍更快

### CHECK-10: 提交代码
```bash
git add workspace/strategies/NEW.py workspace/results/
git commit -m "feat(workspace): iteration N — 策略名 (score=XX, sharpe=XX)"
```

## 策略设计思路池（按优先级尝试）
1. **趋势跟踪 + 严格过滤**：只在强趋势中交易，用多重确认（EMA排列+ADX+成交量）
2. **统计套利/配对**：BTC vs ETH 的价差回归
3. **波动率收缩后突破**：Squeeze（BB收缩于KC内）后方向性突破
4. **时间模式**：利用小时级别的周期性（亚洲/欧洲/美国时段）
5. **反转信号组合**：RSI背离 + 吞没形态 + 支撑位
6. **ML 预测驱动**：用 workspace 的 Ridge/LightGBM 模型预测信号
7. **自适应参数**：根据最近 N 根 K 线的波动率动态调整所有参数

## 约束
- 每轮只写 1 个新策略，做深做透
- 必须执行全部 10 个 CHECK（CHECK-1 到 CHECK-10）
- 策略代码不超过 200 行
- 使用真实数据，禁止模拟
- 每轮结尾必须有文字总结（含 checklist 完成状态表格）

## 成功条件
当满足以下全部条件时输出 completion promise：
1. 至少 10 个不同策略已回测并记录
2. 至少 1 个策略 Sharpe > 0（在震荡市中盈利）
3. 最佳策略的 evaluator 分数 > 40
4. experiments.jsonl 有完整的实验记录链
