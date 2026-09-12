"""freqai — FreqAI 训练 / RL / 外部特征流水线。

子包：

  * ``training``       — 数据集构造 + LightGBM/XGBoost/CatBoost gradient_boosting 流水线
  * ``model``          — 模型适配（gradient_boosting，PyTorch optional）
  * ``rl``             — Stable-Baselines3 RL 训练（TradingEnv 包装）
  * ``external_data``  — 外部特征接入（new / on-chain / cross-asset）
"""
