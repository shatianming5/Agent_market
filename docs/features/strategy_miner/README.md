# Strategy Miner

[返回主 README](../../../README.md) · [返回功能索引](../README.md)

Strategy Miner 是 LLM 多角色策略级挖掘系统，用 planner、coder、reviewer、backtester、evaluator 等阶段自动生成、修复、回测、评分和迭代 Freqtrade 策略。

## 代码入口

| 位置 | 用途 |
|---|---|
| `scripts/strategy_miner.py` | CLI 入口 |
| `src/agent_market/strategy_miner/` | 核心实现 |
| `configs/strategy_miner_*.json` | 挖掘配置 |
| `server/api/routes/strategy_miner.py` | HTTP API |
| `artifacts/runs/<run_id>/strategy_miner/` | 运行产物 |

## 运行

```bash
python scripts/strategy_miner.py --config configs/strategy_miner_default.json
python -m agent_market.strategy_miner --config configs/strategy_miner_default.json
```

覆盖迭代数或模型：

```bash
python scripts/strategy_miner.py \
  --config configs/strategy_miner_default.json \
  --max-iterations 1 \
  --model <model-name> \
  --verbose
```

恢复中断：

```bash
python scripts/strategy_miner.py \
  --resume artifacts/runs/<run_id>/strategy_miner/checkpoint.json
```

## 流程

| 阶段 | 说明 |
|---|---|
| `STRATEGY_GEN` | planner 设计方案，coder 生成策略，reviewer 审查 |
| `TRAIN_MODEL` | 可选，ML/DL/RL 家族候选写入训练证据 |
| `BACKTEST` | freqtrade backtesting，失败可进入 repair |
| `EVALUATION` | 打分并应用 gates |
| `ANALYSIS` | LLM 总结经验并指导下轮 |
| 收尾 | sealed holdout、benchmark、portfolio、promotion chain |

## 产物目录

```text
artifacts/runs/<run_id>/strategy_miner/
  checkpoint.json
  proposal.json
  leaderboard.json
  run_meta.json
  events.jsonl
  economics.json
  promotion_log.jsonl
  portfolio_plan.json
  candidates/
  agent_traces/
  backtests/
  training/
```

## API

```text
POST /strategy-miner/start
GET  /strategy-miner/status/{job_id}
GET  /strategy-miner/runs
GET  /strategy-miner/runs/{run_id}
GET  /strategy-miner/runs/{run_id}/proposal
GET  /strategy-miner/runs/{run_id}/leaderboard
GET  /strategy-miner/runs/{run_id}/candidates
POST /strategy-miner/runs/{run_id}/approve
POST /strategy-miner/runs/{run_id}/backtest
GET  /strategy-miner/results
GET  /strategy-miner/results/{run_id}
```

## 安全约束

候选策略在 sandbox 中处理。系统会限制高风险 import 和函数调用，并检查 Freqtrade 必需方法，如 `populate_indicators`、`populate_entry_trend`、`populate_exit_trend`。

## 验证

```bash
python scripts/strategy_miner.py --help
python scripts/strategy_miner_preflight.py --help
```

