# 强化学习交易环境与训练骨架

本节提供基于 Gymnasium 的 TradingEnv 与 Stable-Baselines3 的 PPO 训练骨架。适用于验证 RL 在本项目特征体系下的可行性与接口对接。

## 组件
- 环境：`TradingEnv`（gymnasium.Env）
  - 状态：由 `FeatureDatasetBuilder` 构造的特征向量（与传统 ML 共享特征体系）
  - 动作：离散 3 值（0: 持有，1: 做多，2: 做空）
  - 奖励：与下一步收益成比例（做多奖励 label，做空奖励 -label），可通过 `reward_positive`/`reward_negative` 调整
- 训练器：`RLTrainer`（PPO）
  - 读入数据配置构建环境 → 训练 → 保存模型与摘要
  - 输出 `artifacts/models/rl_ppo_demo/ppo_trading_env.zip` 和 `training_summary.json`

代码位置：
- 环境：`src/agent_market/freqai/rl/env.py`
- 训练器：`src/agent_market/freqai/rl/trainer.py`
- 脚本：`scripts/train_rl.py`
- 示例配置：`configs/train_ppo.json`

## 依赖
- `gymnasium`、`stable-baselines3`（以及其 extras），注意本地 numpy 版本兼容性：
```
pip install gymnasium "stable-baselines3[extra]"
```
如遇 numpy 版本冲突，请将 numpy 降至 1.x（或安装与当前环境兼容的 SB3 版本）。

## 运行示例
- 使用配置文件：
```
python scripts/train_rl.py --config configs/train_ppo.json
```
- 或用参数快速启动：
```
python scripts/train_rl.py \
  --feature-file user_data/freqai_features.json \
  --data-dir freqtrade/user_data/data \
  --exchange binanceus \
  --timeframe 1h \
  --pairs "BTC/USDT" \
  --total-timesteps 10000 \
  --policy MlpPolicy \
  --algo-params '{"learning_rate":0.0003,"n_steps":2048,"batch_size":64}' \
  --model-dir artifacts/models/rl_ppo_demo
```

## 后端 API（可选）
- `POST /run/rl_train`，body：`{"config": "configs/train_ppo.json"}`
- 返回 job_id，可通过 `/jobs/{job_id}/logs` 查看训练日志

## 提示
- 当前环境为最简骨架，用于验证流程：
  - 奖励函数可扩展（如引入滑点、手续费、风险惩罚、持仓状态）；
  - 状态可改为窗口式（stack 最近 N 帧特征），增强时序记忆；
  - 训练后可将策略导出为规则或在仿真中评估，并与传统 ML 进行 A/B 对比。
