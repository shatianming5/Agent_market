# AI 学习框架总体设计

本文档给出 Agent Market 项目下一阶段的量化学习统一框架设计，力求在现有 FreqAI/LLM 自动因子挖掘能力上，扩展出可插拔的机器学习、深度学习与强化学习管线，并搭建可由智能 Agent 主导的端到端流程。

## 目标概览

1. **统一数据/特征接口**：复用现有 `freqai_feature_agent.py` 产出的配置，提供面向模型的标准化特征矩阵与标签构建器。
2. **模型注册与训练调度**：抽象出统一的 `ModelAdapter` 接口，支持 LightGBM、XGBoost、CatBoost、PyTorch、强化学习策略等多种实现，并允许后续扩展。
3. **多阶段训练流水线**：
   - 特征装载 → 训练集/验证集切分 → 模型训练 → 回测评估 → 结果落盘。
   - 支持网格/贝叶斯调参与多模型集成。
4. **强化学习集成**：在策略层面注入 RL Agent（如 Stable-Baselines3），复用同一数据装载接口，将交易环境包装为 Gym-like 环境。
5. **Agent 编排**：提供统一的 `agent_flow.py`（待实现），串联数据下载、特征更新、表达式生成、模型训练、回测与报告，支撑自动调度与闭环反馈。

## 总体架构

```
                 +----------------+
                 |  Data Manager  |
                 | (行情/标签构建) |
                 +--------+-------+
                          |
             +------------v-------------+
             | Feature/Expression Store |
             | (freqai_features /       |
             |  freqai_expressions)     |
             +------------+-------------+
                          |
        +-----------------v-------------------+
        |           Training Orchestrator     |
        |-------------------------------------|
        | · ModelRegistry (Adapter 工厂)       |
        | · PipelineRunner                    |
        | · Callback / Logger                 |
        +---------+---------------+----------+
                  |               |
        +---------v-----+   +-----v----------+
        | ML Trainers   |   | DL / RL Trainers|
        | (LightGBM,    |   | (PyTorch, SB3)  |
        |  XGBoost, …)  |   |                 |
        +---------------+   +-----------------+
                  |               |
                  +-------+-------+
                          |
                 +--------v--------+
                 | Backtest / Eval |
                 | (Freqtrade, RL) |
                 +--------+--------+
                          |
                 +--------v---------+
                 | Reports & Agent  |
                 | (LLM Feedback,   |
                 |  Dashboard, …)   |
                 +------------------+
```

## 核心组件说明

### 1. 数据与特征层
- `FeatureDatasetBuilder`：负责将 `freqai_features.json` 与历史数据转化为模型可用矩阵（pandas / numpy / torch Tensor）。
- `LabelBuilder`：支持回归（收益）、分类（上涨概率）及 RL Reward。应支持多时间窗口与多资产。

### 2. 模型适配层
定义抽象基类 `BaseModelAdapter`，约束以下接口：

```python
class BaseModelAdapter(ABC):
    name: str

    @abstractmethod
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, **kwargs): ...

    @abstractmethod
    def predict(self, X): ...

    @abstractmethod
    def save(self, path: Path): ...

    @abstractmethod
    def load(self, path: Path): ...
```

预期实现：
- `LightGBMAdapter`（现有 FreqAI 默认模型迁移至新接口）。
- `XGBoostAdapter`、`CatBoostAdapter`。
- `PyTorchAdapter`：支持自定义神经网络（LSTM、Transformer、MLP）。
- `RLAdapter`：封装 Stable-Baselines3 策略，持有交易环境。

### 3. 训练调度
`TrainingPipeline` 负责：
1. 读取配置（模型类别、超参、时间范围、特征组合）。
2. 调用 `FeatureDatasetBuilder` 获取训练/验证集。
3. 选择 `ModelAdapter` 训练模型。
4. 将预测结果注入回测（调用 Freqtrade CLI 或内嵌回测 API）。
5. 产出评估指标（收益、Sharpe、R²、胜率、最大回撤）。
6. 更新反馈文件（供 LLM 使用）。

### 4. 深度学习与强化学习
- **DL**：依赖 `torch`，支持 GPU/CPU；训练过程中允许自定义 `collate_fn`、数据增强、早停。
- **RL**：封装 `TradingEnv`（Gym 环境），以 `FeatureDatasetBuilder` 的输出构建状态，动作对应仓位调整或信号触发；训练完成后可导出策略参数，转化为 Freqtrade 策略或直接运行在仿真环境。

### 5. Agent 全流程控制
新增脚本 `scripts/agent_flow.py`（后续阶段实现）：
1. 下载/更新行情数据。
2. 调用 `freqai_feature_agent.py` 构建特征。
3. 调用 `freqai_expression_agent.py` + LLM 生成表达式。
4. 执行 `TrainingPipeline`（可指定算法列表）。
5. 跑回测并生成总结。
6. 记录日志、更新反馈、可选地触发下一轮优化。

Agent Flow 需要具备任务配置（YAML/JSON）、步骤幂等性、异常恢复与历史记录归档。

## 阶段划分

| 阶段 | 目标 | 关键交付 |
| --- | --- | --- |
| Stage 1 | 架构设计（本文档） | 架构文档、更新 README | 
| Stage 2 | 实现 ML 模型接口与管线 | `BaseModelAdapter` 及 LightGBM/XGB 接入，流水线脚本 | 
| Stage 3 | 深度学习集成 | PyTorch 训练适配器、配置示例 | 
| Stage 4 | 强化学习骨架 | TradingEnv、SB3 接口、简单策略 demo | 
| Stage 5 | Agent Flow 实现 | 自动化编排脚本、日志与测试 | 

## 依赖建议

- 机器学习：`xgboost>=1.7`, `catboost>=1.2`。
- 深度学习：`torch>=2.2`, `pytorch-lightning`（可选）。
- 强化学习：`stable-baselines3>=2.3`, `gymnasium`。
- 实验追踪（可选）：`mlflow` 或 `wandb`。

## 风险与注意事项

- **性能**：特征矩阵尺寸较大时需考虑内存与训练时间；建议引入增量训练与采样机制。
- **一致性**：Freqtrade/FreqAI 与自定义模块应共用数据切分逻辑，避免“训练/回测窗口不一致”。
- **可维护性**：保持模型接口最小化，所有外部库调用隐藏在 Adapter 内部，防止脚本间互相依赖。
- **安全性**：LLM 生成的表达式仍需离线验证；强化学习策略上线前必须经过多区间回测。

---
后续阶段将在此设计基础上逐步落地，实现可扩展的多模态量化研究平台。
## 已完成功能进展

- **Stage 2**：完成模型抽象与 LightGBM/XGBoost/CatBoost 适配器实现。
- **Stage 3**：集成 PyTorch MLP 训练适配器，覆盖单元测试。
- **Stage 4**：实现 `TradingEnv` 与 `RLTrainer`（PPO），形成强化学习训练骨架。

- **Stage 5**：实现 Agent Flow 脚本，串联特征、表达式、ML、RL 与回测步骤。

- **Stage 5**：新增 Agent Flow (脚本 + 编排)，支持步骤筛选、日志落盘、回测摘要自动生成（位于 `user_data/llm_feedback/latest_backtest_summary.json`）并回灌至表达式生成。
