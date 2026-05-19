"""agent_market — LLM + 量化研究工作台业务核心。

子包导览（按"我现在该读哪里"组织；详见根 ``AGENTS.md`` 与 ``docs/repo_inventory.md``）：

  * ``wq_brain``       — WorldQuant BRAIN agentic alpha mining 主驱动子系统
  * ``factor_lab``     — 因子挖掘 / 回测 / 部署统一框架
  * ``factor_hub``     — 因子注册表 / 评估存储 / 部署 API
  * ``strategy_miner`` — LLM 驱动的策略级挖掘
  * ``factor_compiler``— 因子编译 (DSL → 可执行) + 检查 + 评分
  * ``freqai``         — FreqAI 训练流水线 (gradient_boosting / RL / 外部特征)
  * ``microstructure`` — 市场微观结构（capture / LOB / features）
  * ``tca``            — Transaction Cost Analysis
  * ``flow_ext``       — agent_flow 步骤分发与 artifacts
  * ``agents``         — provider-agnostic agent executors

顶层 flat 模块（功能性，不构成 domain 子系统）：

  * ``agent_flow``       — 离线主流水线编排（dispatch flow_ext.steps）
  * ``flow_steps``       — flow 步骤的具体实现
  * ``runtime_bootstrap``/``runtime_preflight`` — 运行时引导 + 预检
  * ``paths``/``config``/``utils`` — 路径、配置、通用工具
  * ``strategy_factory`` — 策略工厂 artifact finalization
  * ``strategy_registry``/``factor_memory``/``run_artifacts`` — 注册表 / 记忆 / artifacts 写入
  * ``portfolio_opt``    — HRP / 组合优化
  * ``backtest_results`` — 回测结果数据结构
  * ``factor_multiagent``— 因子多 agent 协作
"""
