# Ralph Loop: 搭建 OpenCode 自主量化研究工作台

## 总目标
搭建一个 workspace，让 OpenCode agent 拥有量化研究员的完全自主权：自己写算法代码、写策略逻辑、生成配置、调用回测 API、根据多目标反馈自我迭代。

## 架构设计

```
workspace/
├── models/              ← agent 写的算法代码（ML/DL/统计模型）
├── strategies/          ← agent 写的 freqtrade 策略
├── configs/             ← agent 生成的回测配置
├── results/             ← 每次实验的结果记录
├── objectives.json      ← 多目标优化定义
└── toolbox.md           ← 可用的工具/数据/API 说明
```

## 需要实现的 4 个核心模块

### 模块 1: 回测 API (workspace/backtest_api.py)
封装 freqtrade 回测为一个 Python 函数调用：
```python
result = run_backtest(
    strategy_path="workspace/strategies/my_strategy.py",
    config_path="workspace/configs/backtest.json",
    timerange="20251126-20260125",
    pairs=["BTC/USDT", "ETH/USDT"],
)
# result = {sharpe, sortino, max_dd, profit_pct, win_rate, trades, calmar, profit_factor, ...}
```
- 输入：策略 .py 文件路径 + 配置
- 输出：标准化的指标字典
- 必须处理异常（策略代码有 bug 时返回 error 而不是崩溃）
- 结果自动写入 workspace/results/

### 模块 2: 动态模型加载器 (workspace/model_loader.py)
让 agent 写的模型代码自动被 pipeline 识别：
- 扫描 workspace/models/*.py
- 找到继承 BaseModelAdapter 的类
- 自动注册到 ModelRegistry
- 策略中可以通过 model_name 引用

### 模块 3: 多目标评估器 (workspace/evaluator.py)
读取 objectives.json，判断回测结果是否达标：
```json
{
  "targets": {
    "sharpe": {"min": 1.0, "weight": 0.3},
    "max_drawdown_pct": {"max": 5.0, "weight": 0.2},
    "profit_pct": {"min": 1.0, "weight": 0.2},
    "win_rate": {"min": 0.45, "weight": 0.15},
    "profit_factor": {"min": 1.1, "weight": 0.15}
  },
  "constraints": {
    "min_trades": 50,
    "max_avg_duration_hours": 48
  }
}
```
- 计算综合得分 (0-100)
- 标记哪些目标达标/未达标
- 给出改进建议（哪个指标最差，应该往哪个方向调）

### 模块 4: 实验追踪器 (workspace/tracker.py)
每次实验自动记录：
- 实验 ID + 时间戳
- 使用的策略代码 SHA256
- 使用的模型代码 SHA256
- 配置快照
- 回测结果指标
- 多目标评分
- 写入 workspace/results/experiments.jsonl（追加模式）
- 支持查询历史最佳、对比两次实验

## 实现顺序（每轮迭代的 checklist）

### 迭代 1: workspace 骨架 + objectives
- [ ] 创建 workspace/ 完整目录结构
- [ ] 写 workspace/objectives.json（多目标定义）
- [ ] 写 workspace/toolbox.md（可用工具/数据说明）
- [ ] 写 workspace/strategy_template.py（策略基类模板）
- [ ] pytest 通过

### 迭代 2: 回测 API
- [ ] 实现 workspace/backtest_api.py
- [ ] run_backtest() 函数：接收策略路径+配置 → 返回指标字典
- [ ] 错误处理：策略代码有 bug 时返回 {error: "..."} 不崩溃
- [ ] 写测试 tests/test_backtest_api.py
- [ ] 用现有策略验证 API 能正常返回结果

### 迭代 3: 动态模型加载器
- [ ] 实现 workspace/model_loader.py
- [ ] scan_workspace_models() 扫描并注册
- [ ] 写一个示例模型 workspace/models/example_ridge.py（Ridge 回归）
- [ ] 验证示例模型能被 pipeline 加载和训练

### 迭代 4: 多目标评估器
- [ ] 实现 workspace/evaluator.py
- [ ] evaluate(result, objectives) → {score, details, suggestions}
- [ ] 写测试验证评分逻辑
- [ ] 用真实回测结果验证

### 迭代 5: 实验追踪器
- [ ] 实现 workspace/tracker.py
- [ ] record_experiment() 写入 experiments.jsonl
- [ ] query_best() 查询历史最佳
- [ ] compare(id1, id2) 对比两次实验

### 迭代 6: agent 写第一个自定义策略
- [ ] 用 LLM 在 workspace/strategies/ 写一个新策略
- [ ] 用 backtest_api 回测
- [ ] 用 evaluator 评分
- [ ] 用 tracker 记录

### 迭代 7: agent 写第一个自定义模型
- [ ] 用 LLM 在 workspace/models/ 写一个新模型（如 LSTM/GRU）
- [ ] 通过 model_loader 动态注册
- [ ] 配合策略跑回测
- [ ] 评分 + 记录

### 迭代 8: 端到端自动化
- [ ] 写 workspace/orchestrator.py 串联所有模块
- [ ] orchestrator 读 objectives → 写策略/模型 → 回测 → 评估 → 记录
- [ ] 验证完整循环跑通

### 迭代 9: 优化迭代
- [ ] orchestrator 读取历史实验结果
- [ ] 根据 evaluator 的建议自动改进策略/模型
- [ ] 跑多轮对比实验

### 迭代 10: 最终验证
- [ ] 所有模块测试通过
- [ ] 至少 3 个不同策略/模型的回测结果
- [ ] 实验追踪有完整记录
- [ ] 多目标评分有改善趋势
- [ ] workspace/toolbox.md 更新为最终版

## 约束
- 所有代码在 workspace/ 目录下（不改动 src/ 核心代码，只调用）
- 回测 API 必须 catch 异常，不能因为 agent 写的代码有 bug 就崩溃
- 每次迭代结束必须有文字总结
- 每次有代码改动 git commit
- 使用真实数据（user_data/data/kucoin/*.feather）
