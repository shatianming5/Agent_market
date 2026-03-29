# Ralph Loop: 搭建 LLM 自主策略迭代优化器

## 总目标
在 workspace 中实现 `auto_improver.py`——一个 LLM 驱动的自主策略优化器。它能：
1. 读取历史实验结果和评估反馈
2. 分析失败原因（LLM 推理）
3. 自动生成改进版策略代码（LLM 写代码）
4. 回测 → 评估 → 记录 → 对比
5. 循环迭代直到达标

## LLM 配置
- endpoint: http://localhost:4141/v1
- model: gpt-5.2
- api_key: _

## 实现计划（10 轮迭代，每轮有细颗粒度 CHECK）

### 迭代 1-2: auto_improver 核心引擎
实现 workspace/auto_improver.py，包含：

```python
class AutoImprover:
    def analyze_history() -> str
        # 读取 experiments.jsonl，用 LLM 分析失败原因

    def generate_strategy(analysis: str, iteration: int) -> Path
        # LLM 根据分析生成新策略代码，写入 workspace/strategies/

    def validate_strategy(path: Path) -> bool
        # 语法检查 + import 检查

    def run_and_evaluate(path: Path) -> dict
        # 回测 + 评估 + 记录（调用 orchestrator）

    def run_cycle(max_iterations: int = 5) -> dict
        # 完整自动循环：analyze → generate → validate → run → evaluate → repeat
```

CHECK 清单（迭代 1）：
- [ ] auto_improver.py 文件创建
- [ ] LLM 连通性验证（localhost:4141 可调用）
- [ ] analyze_history() 能读取 experiments.jsonl 并返回分析文本
- [ ] generate_strategy() 能调用 LLM 生成合法的策略 .py 代码
- [ ] validate_strategy() 能检查语法和 IStrategy 继承
- [ ] python3 -c "from workspace.auto_improver import AutoImprover" 不报错

CHECK 清单（迭代 2）：
- [ ] run_and_evaluate() 端到端跑通（策略→回测→评估→记录）
- [ ] run_cycle(max_iterations=1) 完成一次完整循环
- [ ] 生成的策略确实不同于已有策略（代码 SHA256 不同）
- [ ] experiments.jsonl 有新记录
- [ ] 结果保存到 workspace/results/

### 迭代 3-4: 反馈闭环 + 错误恢复
- [ ] LLM 分析能引用具体的历史指标数据（不是泛泛而谈）
- [ ] LLM 生成的策略能引用已有最佳策略的代码作为参考
- [ ] 策略代码有 bug 时：auto_improver 捕获错误 → 让 LLM 修复 → 重试（最多 3 次）
- [ ] run_cycle(max_iterations=3) 能连续跑 3 轮不崩溃
- [ ] 每轮生成的策略名称递增（auto_v1, auto_v2, ...）

### 迭代 5-6: 运行 auto_improver，积累实验
- [ ] 运行 run_cycle(max_iterations=5)，产出 5 个新策略
- [ ] 所有策略都成功回测（没有因 bug 跳过的）
- [ ] experiments.jsonl 有完整 5 条新记录
- [ ] query_best() 能看到新策略的排名
- [ ] 至少有 1 个新策略 Sharpe > 历史最佳

### 迭代 7-8: 策略组合 + 参数优化
- [ ] auto_improver 能读取最佳策略代码，让 LLM 做微调（改参数/阈值）
- [ ] 实现 parameter_sweep()：对最佳策略的关键参数做网格搜索
- [ ] 网格搜索结果记录到 experiments.jsonl
- [ ] 找到最优参数组合

### 迭代 9: 最终验证 + 报告
- [ ] 累计至少 15 个实验记录
- [ ] query_best("sharpe", 1) 返回的策略 Sharpe > 0
- [ ] evaluator 分数 > 40
- [ ] 生成最终研究报告 workspace/results/final_report.json
- [ ] 报告包含：策略进化历程、最佳策略代码、所有实验对比

### 迭代 10: 清理 + 文档 + 全量测试
- [ ] pytest 全部通过（含新增测试）
- [ ] workspace/toolbox.md 更新 auto_improver 使用说明
- [ ] git commit 所有变更
- [ ] 验证完整流程：auto_improver.run_cycle() 可一键启动

## 约束
- auto_improver 生成的策略代码必须在 workspace/strategies/ 下
- LLM 生成的代码必须经过语法验证才能回测
- 每次 LLM 调用必须有超时和重试
- 每轮迭代结尾必须有文字总结（含 CHECK 状态表）
- 每轮有代码改动必须 git commit
