# scripts/ — CLI 入口分类索引

> 70 个 Python 脚本按用途分类；每条最多 1 行说明 + "运行示例 / 状态"。
> 旧版 `FACTOR_LAB_README.md` 已移除；脚本导航以本文件为准。

## 主入口（"如果你只能跑一个，就跑这些"）

| 脚本 | 用途 | 一句话 |
|---|---|---|
| `agent_flow.py` | **离线主流水线** | `--steps feature expression ml backtest`；步骤 dispatcher 在 `src/agent_market/flow_ext/step_dispatch.py` |
| `factor_lab.py` | **研究 CLI 总入口** | 子命令 `data / features / mine / validate / backtest / rank-export / rank-backtest / rl / combo / deploy / hub` |
| `wq_brain.py` | **WQ BRAIN agentic alpha mining CLI** | `simulate / pool submit-worker / sync-status / pre-check / pre-check-local / kaggle-import / ...`，~3100 行 |
| `strategy_miner.py` | **LLM 策略级挖掘** | `--config configs/strategy_miner_default.json` |
| `smoke_test.py` | **黄金路径冒烟** | 跑通即代表后端 + flow + 主依赖完好 |
| `freqtrade_cli.py` | freqtrade 命令包装 | 桥接 vendored freqtrade |

## 数据获取 / 清洗

| 脚本 | 数据源 / 用途 |
|---|---|
| `fetch_ccxt_ohlcv.py` / `download_15m_ccxt.py` | CCXT OHLCV |
| `fetch_binance_bulk.py` | Binance 批量历史 |
| `okx_universe_download.py` / `okx_universe_download_queue.py` | OKX universe 拉取（含队列） |
| `build_candles.py` / `clean_ohlcv.py` / `normalize.py` | 蜡烛构造 / 清洗 / 归一化 |
| `lob_snapshot.py` / `lob_rebuild.py` / `micro_capture.py` / `micro_features.py` / `micro_feature_merge_hourly.py` | 微观结构 / LOB 快照与特征 |
| `news_harvester.py` / `x_recent_search.py` / `x_stream.py` | 新闻 / X(Twitter) 数据 |
| `dex_indexer.py` / `live_subscribe.py` | 链上 / 实时订阅 |

## 因子挖掘 / 评估 / 训练

| 脚本 | 用途 |
|---|---|
| `gp_factor_mine_v2.py` | GP 因子挖掘当前实现；`gp_factor_mine.py` 仅保留为兼容 wrapper |
| `factor_compile.py` / `factor_eval.py` | 因子编译 + 评估 |
| `verify_factors.py` / `tag_factor_snoop_level.py` | 因子验证 / 数据窥探等级标记 |
| `walk_forward_pca.py` / `select_by_perm_importance.py` | walk-forward / 重要性筛选 |
| `train_pipeline.py` / `train_rl.py` | ML 训练 / RL 训练 |
| `rl_generate_signals.py` | RL 信号生成 |
| `verify_model_training.py` | 训练验证 |
| `gen_4h_seeds.py` / `gen_researcher_seeds.py` | 种子生成（4h / researcher） |

## 回测 / 报告 / 部署

| 脚本 | 用途 |
|---|---|
| `backtest_wrapper.py` | freqtrade 回测包装 |
| `report_backtest.py` / `tca_report.py` / `dq_report.py` | 回测 / TCA / 数据质量报告 |
| `eval_protocol.py` | 评估协议执行 |
| `vbt_ma_rsi.py` | vectorbt MA+RSI 基线 |
| `e2e_smoke_flow.py` / `closed_loop_demo.py` | 端到端 / 闭环 demo |

## strategy_miner 配套

| 脚本 | 用途 |
|---|---|
| `prepare_strategy_miner_data.py` | 数据准备 |
| `make_strategy_miner_opencode_config.py` | opencode config 生成 |
| `strategy_miner_preflight.py` / `strategy_miner_backtest.py` | preflight / backtest 子任务 |
| `verify_strategy_miner_run.py` / `run_strategy_miner_recovery.py` | 验证 / 恢复 |
| `strategy_factory_acceptance.py` / `bootstrap_strategy_factory_loops.py` | 验收 / 启动 |
| `strategy_loop_monitor.py` | loop 监视器 |
| `audit_mining_real_test.py` | 真实测试审计 |
| `backfill_global_strategy_memory.py` | 全局 strategy memory 回填 |

## 维护 / 元数据 / 工具

| 脚本 | 用途 |
|---|---|
| `clean_workspace.py` / `clean_workspace.ps1` | 工作区清理（含 PowerShell 版） |
| `gc_jobs.py` / `gc_runs.py` | 旧 jobs/runs 垃圾回收 |
| `expr_agent_wrapper.py` / `freqai_expression_agent.py` / `freqai_feature_agent.py` | 表达式 / FreqAI 特征 agent 包装 |
| `llm_config_optimizer.py` | LLM 配置自动优化 |
| `remine_factors_clean.py` | 清洗后重新挖矿 |
| `xray_sub_to_config.py` | xray 订阅 → config |
| `_lib.py` | 通用辅助库（不直接执行） |

## ws_production / 单独工作区

| 脚本 | 用途 |
|---|---|
| `ws_production_factor_cycle.py` / `ws_production_monitor.py` / `ws_production_preflight.py` | ws_production 独立 cycle / monitor / preflight |

> ⚠️ `ws_production` 是**独立**实验区；不要假设它和主 flow 共享配置或 artifacts 路径。

## 已知架构缺口（不要在生产路径使用）

- `python scripts/wq_brain.py pool resubmit-all` — 没有走完整的 quota 预留 + outcome 持久化（详见 `AUTO_REVIEW.md` R1-#4 / R2-#3 / R3-#3）。生产请用 `pool submit-worker` 替代。
- `python scripts/wq_brain.py scan --auto-submit` (实现在 `src/agent_market/wq_brain/scan_runner.py:148`) — 同样绕过 truth/quota 栈。

## 写新脚本前先确认

1. 是否已有同类脚本（grep 名字 + 用途）。
2. 是否应该作为 `factor_lab.py` 或 `wq_brain.py` 的子命令而不是新文件。
3. 单脚本 ≥ 500 行考虑拆分到 `src/agent_market/<package>/` 让 CLI 只做 argparse + dispatch。
