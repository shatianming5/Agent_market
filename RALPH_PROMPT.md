# Ralph Loop: 打造真实可交易的 Agent 策略系统

## 总目标
将当前研究原型升级为可在真实市场运行的 agent-driven 策略挖掘系统。

## 10 轮迭代计划

### 迭代 1: 下载 2 年历史数据
解决：60 天数据导致 ML 必然过拟合
CHECK:
- [ ] 用 freqtrade download-data 下载 BTC/USDT ETH/USDT 2 年 1H 数据
- [ ] 额外下载 4-6 个品种：SOL/USDT, DOGE/USDT, XRP/USDT, AVAX/USDT
- [ ] 验证数据行数 >= 15000 (2年×365×24/1)
- [ ] 数据保存为 feather 格式到 user_data/data/kucoin/
- [ ] 更新 objectives.json 的数据配置

### 迭代 2: Walk-Forward 滚动验证器
解决：单次切分不可靠
CHECK:
- [ ] 实现 workspace/walk_forward.py
- [ ] WalkForwardValidator 类：将 2 年数据切成 N 个窗口
- [ ] 每窗口：用前 6 月训练，后 2 月测试，滚动前进
- [ ] 返回每个窗口的 Sharpe/DD/profit，加上聚合统计（均值±标准差）
- [ ] 只有全部窗口平均 Sharpe > 0 才算通过
- [ ] 集成到 auto_improver.run_full_cycle()

### 迭代 3: 真实交易成本建模
解决：0.1% 固定费不现实
CHECK:
- [ ] 创建 workspace/cost_model.py
- [ ] 实现 SlippageModel：基于成交量估算滑点
- [ ] 实现 spread_cost = f(volatility, volume)
- [ ] 修改 freqtrade 回测配置：加入 realistic fee + slippage
- [ ] 对比：固定费 vs 真实成本 模型下的策略表现差异

### 迭代 4: 多品种数据 + 品种选择器
解决：2 个品种不够分散
CHECK:
- [ ] 验证 6+ 品种数据已下载
- [ ] 实现 workspace/universe_selector.py
- [ ] 基于流动性+波动率自动筛选品种
- [ ] 更新 backtest_api 支持多品种回测
- [ ] 运行 auto_improver 在新品种上生成策略

### 迭代 5: 特征选择 + 正则化
解决：50-80 维特征导致过拟合
CHECK:
- [ ] 实现 workspace/feature_selector.py
- [ ] 方法：mutual_info + 递归特征消除(RFE) + L1 正则化
- [ ] 目标：从 50-80 维降到 10-20 维核心特征
- [ ] 验证：降维后 train/valid IC 差距缩小
- [ ] 集成到 auto_improver 的训练流程

### 迭代 6: 用新数据+验证器跑 auto_improver
解决：在真实条件下验证 agent 策略
CHECK:
- [ ] auto_improver.run_full_cycle() 使用 2 年数据
- [ ] Walk-Forward 验证（非单次切分）
- [ ] 真实成本模型
- [ ] 特征选择后的模型
- [ ] 至少 3 个策略跑完整流程
- [ ] 检查 OOS 平均 Sharpe

### 迭代 7: 策略集成 + 市场状态
解决：单策略不稳定
CHECK:
- [ ] 实现 workspace/ensemble.py
- [ ] 多策略信号加权组合（等权 / IC 加权 / 风险平价）
- [ ] 市场状态检测（高波/低波/趋势/震荡）
- [ ] 不同状态激活不同策略子集
- [ ] 回测集成策略 vs 单策略对比

### 迭代 8: 仓位管理 + 风控
解决：无风控=赌博
CHECK:
- [ ] 实现 workspace/risk_manager.py
- [ ] Kelly 公式仓位管理
- [ ] 最大回撤熔断（DD>X% 自动停止）
- [ ] 单笔最大亏损限制
- [ ] 品种间相关性检查（防集中暴露）

### 迭代 9: 纸盘验证框架
解决：回测不等于实盘
CHECK:
- [ ] 实现 workspace/paper_trader.py
- [ ] 接入交易所实时数据（WebSocket）
- [ ] 模拟下单（不花真钱）
- [ ] 记录每笔模拟交易到日志
- [ ] 对比纸盘 vs 回测差异

### 迭代 10: 最终集成 + 验证
CHECK:
- [ ] 完整链路：数据 → 特征选择 → 模型训练 → Walk-Forward → 集成 → 风控 → 回测
- [ ] auto_improver 使用全部升级后的基础设施
- [ ] 生成最终报告：最佳策略 + 全窗口 Sharpe 分布
- [ ] pytest 通过
- [ ] 所有模块文档更新

## 约束
- 数据下载需要网络，如果交易所不可达则跳过该品种
- Walk-Forward 计算量大，每个窗口限制训练时间
- 每轮结尾必须有 CHECK 表 + 文字总结
- 每轮有代码改动必须 git commit
