# Ralph Loop: 让 OpenCode 自主写 ML/DL/RL 算法

## 总目标
扩展 auto_improver，让 opencode 能自主编写 ML/DL/RL 模型代码，训练模型，再写策略调用模型预测做交易。

## 完整链路
```
opencode → 写模型代码(workspace/models/xxx.py)
         → 训练模型(TrainingPipeline)
         → 写策略代码(workspace/strategies/ml_xxx.py, 加载模型做预测)
         → 回测 → 评估 → 记录
```

## 已有基础
- ModelRegistry: lightgbm, xgboost, catboost, pytorch_mlp, ridge_regression
- model_loader.py: 动态扫描 workspace/models/ 注册新模型
- TrainingPipeline: 支持任何 BaseModelAdapter
- ExpressionLongStrategy: 示范了如何加载模型做预测
- backtest_api.py / evaluator.py / tracker.py 全部就绪
- opencode run -m custom/gpt-5.2 已验证可用

## 10 轮迭代计划

### 迭代 1: auto_improver 增加 generate_model() + train_model()
CHECK:
- [ ] generate_model(model_type, iteration) 调用 opencode 写模型代码
- [ ] model_type 支持: "ml"(sklearn/lightgbm), "dl"(pytorch), "rl"(简单Q-learning)
- [ ] 写出的代码继承 BaseModelAdapter，有 fit/predict/save/load
- [ ] train_model(model_path) 调用 TrainingPipeline 训练
- [ ] 训练产出 model 文件 + training_summary.json
- [ ] python3 -c "from workspace.auto_improver import AutoImprover; ai=AutoImprover(); print('OK')"

### 迭代 2: generate_ml_strategy() — 写加载模型的策略
CHECK:
- [ ] generate_ml_strategy(model_dir, iteration) 写策略代码
- [ ] 策略在 populate_indicators 中加载模型、做预测
- [ ] 策略用模型预测值做 entry/exit 决策
- [ ] 语法验证通过

### 迭代 3: 端到端验证 — 写模型 → 训练 → 写策略 → 回测
CHECK:
- [ ] opencode 写一个 ML 模型（如 GradientBoosting 变体）
- [ ] 训练成功，产出 training_summary.json
- [ ] opencode 写配套策略
- [ ] 回测成功，返回 Sharpe/DD 等指标
- [ ] experiments.jsonl 记录完整

### 迭代 4: opencode 写 DL 模型（PyTorch）
CHECK:
- [ ] opencode 写一个 LSTM 或 GRU 模型
- [ ] 训练成功
- [ ] 配套策略回测成功
- [ ] 对比 ML vs DL 结果

### 迭代 5: opencode 写 RL 模型
CHECK:
- [ ] opencode 写一个简单 RL（Q-learning / DQN-lite）模型
- [ ] 训练成功
- [ ] 配套策略回测成功

### 迭代 6: run_full_cycle() — 自动 ML/DL/策略迭代
CHECK:
- [ ] 新方法 run_full_cycle(model_types=["ml","dl"], iterations=3)
- [ ] 每轮: 分析 → 选模型类型 → 写模型 → 训练 → 写策略 → 回测 → 评估
- [ ] 3 轮不崩溃

### 迭代 7-8: 批量实验积累
CHECK:
- [ ] 跑 run_full_cycle 积累至少 5 个 ML/DL 实验
- [ ] experiments.jsonl 中有 ML/DL 标签的记录
- [ ] query_best 能区分模型类型

### 迭代 9: 最终报告
CHECK:
- [ ] 生成 final_report.json: 策略进化、ML vs DL vs RL 对比
- [ ] 至少 1 个 ML/DL 策略 Sharpe > 0

### 迭代 10: 测试 + 清理
CHECK:
- [ ] pytest 全部通过
- [ ] toolbox.md 更新 ML/DL/RL 使用说明
- [ ] git commit

## 关键约束
- 模型代码必须在 workspace/models/ 下
- 策略代码必须在 workspace/strategies/ 下
- 训练数据用 user_data/data/kucoin/*.feather
- opencode 写代码有 bug → 自动修复重试（最多 3 次）
- 每轮结尾必须有文字总结 + CHECK 表
