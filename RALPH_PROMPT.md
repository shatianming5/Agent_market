# Ralph Loop: 驱动 Agent Market 生产可盈利策略

## 目标
让 Agent Market 的完整管道真正跑通，利用 LLM (localhost:4141/v1, gpt-5.2) 驱动因子生成、编译、评估、训练、回测，并整合微观结构特征和 TCA 成本分析，最终产出可验证收益的策略文件。

## 当前环境
- LLM API: http://localhost:4141/v1 (支持 gpt-4o-mini, gpt-4.1, gpt-5.2)
- 数据: user_data/data/kucoin/BTC_USDT-1h.feather, ETH_USDT-1h.feather
- 回测框架: freqtrade (本地子模块)
- ML: LightGBM/XGBoost (需确认已安装)
- Factor Compiler: src/agent_market/factor_compiler/ (完整)
- 微观结构: src/agent_market/microstructure/ (OHLCV 模式可用)
- TCA: src/agent_market/tca/ (完整)

## 每次迭代的工作流

### Phase 1: 诊断与修复 (前 1-3 轮)
1. 验证 LLM 连通性: 用 Python 测试 localhost:4141/v1 是否正常响应
2. 验证依赖: 检查 lightgbm, xgboost, freqtrade 等是否可导入，缺什么装什么
3. 验证数据: 确认 feather 文件可读，列名正确
4. 修复阻塞: 如果 agent_flow.py 跑不通，逐步定位并修复问题
5. 先跑最小闭环: python scripts/agent_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json --steps feature expression ml backtest

### Phase 2: 启用 LLM 驱动 (第 4-6 轮)
1. 创建/修改配置文件启用 LLM 表达式生成:
   - 确保配置中 llm.enabled=true, llm.base_url=http://localhost:4141/v1, llm.model=gpt-5.2
2. 运行 LLM 驱动的表达式生成: --steps feature expression
3. 验证 LLM 生成的表达式质量: 检查 user_data/freqai_expressions_selected.json
4. 用 Factor Compiler 编译并验证表达式: --steps factor_compile factor_eval
5. 如果表达式质量差，调整 prompt 模板或参数后重试

### Phase 3: 全管道 + 扩展 (第 7-9 轮)
1. 跑完整管道: --steps feature micro_feature expression factor_compile factor_eval ml backtest tca
2. 分析回测结果: 检查 backtest-result-*.zip 和 latest_backtest_summary.json
3. 分析 TCA 报告: 检查 IS 分解、spread/impact 是否合理
4. 分析微观结构特征: 检查 features.parquet 是否正常产出
5. 根据回测和 TCA 结果调整策略参数（如 label_period, 表达式筛选阈值等）
6. 如果回测亏损，分析原因并调整:
   - 表达式质量差 → 改进 LLM prompt
   - 过拟合 → 增大 purge/embargo, 减小 train_period
   - 成本太高 → 从 TCA 报告中找到成本最高的组件并优化

### Phase 4: 迭代优化 (第 10 轮)
1. 汇总所有轮次的回测结果
2. 对比不同配置下的 Sharpe/drawdown/费用
3. 输出最终的最优策略配置和回测结果
4. 确保所有产物落盘: run_meta.json, backtest_summary, tca_report, factor_scores

## 关键文件位置
- 主入口: scripts/agent_flow.py
- 配置模板: configs/agent_flow_kucoin_cpu_nollm.json
- LLM 配置: .env (已更新为 localhost:4141)
- LLM 调用: src/agent_market/freqai/llm.py
- 表达式引擎: src/agent_market/freqai/expression_engine.py
- 训练管道: src/agent_market/freqai/training/pipeline.py
- Factor Compiler: src/agent_market/factor_compiler/
- 微观结构: src/agent_market/microstructure/micro_feature.py
- TCA: src/agent_market/tca/report.py
- 策略模板: freqtrade/ 子模块

## 约束
- 使用真实数据 (user_data/data/kucoin/*.feather)，禁止模拟数据
- 每次迭代结束必须有文字总结（否则 Ralph Loop hook 会崩）
- 代码改动要保持向后兼容，不破坏现有测试
- 每次有实质性改动后 git commit
- 如果某个步骤失败，先诊断原因再修复，不要跳过

## 成功标准
当满足以下全部条件时，输出 completion promise:
1. LLM 驱动的表达式生成可正常工作
2. Factor Compiler 编译+评估链路跑通
3. 完整管道 (feature→expression→ml→backtest→tca) 至少成功运行一次
4. 回测结果有明确的 Sharpe ratio 和 drawdown 数据
5. TCA 报告包含完整的 IS 分解
6. 微观结构特征正常产出
