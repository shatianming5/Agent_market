# Agent Flow 离线主流水线

[返回主 README](../../../README.md) · [返回功能索引](../README.md)

Agent Flow 是仓库的本地离线编排层，把 feature、expression、ML/RL、backtest、TCA、report 等步骤串成一个可复现 run。

## 代码入口

| 位置 | 用途 |
|---|---|
| `scripts/agent_flow.py` | CLI 入口 |
| `src/agent_market/agent_flow.py` | run_id、preflight、step 顺序、run_meta 写入 |
| `src/agent_market/flow_ext/step_dispatch.py` | step 到 handler 的真相表 |
| `src/agent_market/flow_steps.py` | 具体 step 执行逻辑 |
| `src/agent_market/run_artifacts.py` | run artifact 容器 |
| `configs/agent_flow_*.json` | Flow 配置 |

## 黄金路径

```bash
python scripts/agent_flow.py \
  --config configs/agent_flow_kucoin_cpu_nollm.json \
  --steps feature expression ml backtest
```

可用 Makefile：

```bash
make flow
make flow-smoke
```

## 支持的主要 step

| Step | 说明 |
|---|---|
| `feature` | 生成 / 合并 FreqAI 特征 |
| `expression` | LLM 或非 LLM 表达式生成、评分与 factor memory 写入 |
| `ml` | LightGBM / XGBoost / CatBoost / stacked / ridge classifier 等训练路径 |
| `rl` | PPO / recurrent PPO / BC 相关训练或评估路径 |
| `backtest` | Freqtrade 回测包装与摘要 |
| `portfolio` | HRP 风格组合权重与报告 |
| `capture` | 市场数据采集 |
| `lob_rebuild` | LOB 重建 |
| `micro_feature` | 微观结构特征生成 |
| `factor_compile` | FactorSpec DSL 编译与静态检查 |
| `factor_eval` | 因子评分与评估产物 |
| `tca` | 交易成本分析 |
| `strategy_miner` | 在同一 run 下启动策略挖掘 |
| `report` | 打包本次 run 的关键产物 |

## 闭环 demo

隔离 workspace，避免覆盖默认 `artifacts/` 与 `user_data/`：

```bash
python scripts/closed_loop_demo.py
```

直接跑完整 fixture：

```bash
python scripts/agent_flow.py \
  --config configs/agent_flow_closed_loop_demo_fixture.json \
  --steps capture lob_rebuild micro_feature factor_compile factor_eval ml backtest tca report
```

## 产物

每次 run 会写：

```text
artifacts/run_meta.json
artifacts/runs/<run_id>/run_meta.json
artifacts/runs/<run_id>/preflight.json
```

`run_meta.json` 记录 run_id、状态、config snapshot、Python/freqtrade 信息、preflight、每个 step 的状态和 artifacts 路径。

## API 对应

```text
POST /flow/run
GET  /flow/progress/{job_id}
GET  /flow/stream/{job_id}
WS   /flow/ws/{job_id}
GET  /flow/run-meta/latest
GET  /flow/run-meta/{run_id}
GET  /flow/runs/list
```

## 验证

```bash
python scripts/e2e_smoke_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json
python scripts/smoke_test.py
pytest -q
```

