# 落地计划（研究工作台 / 本地单机 / CPU / 默认不启用 LLM）

## 目标（MVP）

把 Agent Market 作为“可复现”的策略研究工作台落地：在本地单机环境下，从数据 → 特征 → 表达式 → 训练 → 回测 → 结果/反馈，形成一条稳定的黄金路径，且每一步都有可验收的产物与命令。

## 非目标（MVP 不做）

- 生产级服务（鉴权、限流、审计、SLA、灰度等）
- 商业化 SaaS（多租户、计费、配额、合规）
- 分布式训练与大规模并行实验（可后续扩展）

## 默认假设

- 运行模式：本地单机
- 算力：CPU
- 数据源：KuCoin（使用 `user_data/data/kucoin/*-1h.feather` 作为默认复现种子）
- LLM：默认不启用（可选增强）
- 回测：默认离线（不依赖交易所 API；仅使用本地历史数据）

## 黄金路径（唯一推荐入口）

配置文件：`configs/agent_flow_kucoin_cpu_nollm.json`

依赖安装（推荐）：

```bash
pip install -r requirements-full.txt
```

运行：

```bash
python scripts/agent_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json --steps feature expression ml backtest
```

可选：加入 Portfolio（HRP 风险平价）步骤：

```bash
python scripts/agent_flow.py --config configs/agent_flow_kucoin_cpu_nollm_portfolio.json --steps feature portfolio expression ml backtest
```

## 产物规范（验收用）

必须生成：

- 特征：`user_data/freqai_features_real.json`
- 表达式 Top-N：`user_data/freqai_expressions_selected.json`
- 训练摘要：`artifacts/models/**/training_summary.json`
- 回测结果：`user_data/backtest_results/backtest-result-*.zip`
- 回测摘要（供下一轮反馈/对比）：`user_data/llm_feedback/latest_backtest_summary.json`

## 验收（必须全部通过）

1) API 冒烟：

```bash
python scripts/smoke_test.py
```

2) 端到端冒烟（会跑一次完整 Flow 并检查产物落盘）：

```bash
python scripts/e2e_smoke_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json
```

## Pro 扩展（Phase 1：micro_feature + TCA）

> 这部分在不引入真实 L2/订单簿数据的前提下，先把“可度量执行成本/结果归因”的交付链路跑通。

推荐配置（含 micro_feature + tca 步骤）：

```bash
python scripts/agent_flow.py --config configs/agent_flow_pro_kucoin_smoke.json --steps feature expression ml backtest micro_feature tca
```

独立脚本（也可脱离 Flow 单跑）：

```bash
# 生成 OHLCV micro features（产出 parquet + manifest）
python scripts/micro_features.py --config user_data/config_freqai_kucoin.json --run-id <run_id> --timerange 20260101-20260110

# 从最新 backtest zip 生成 TCA 报告
python scripts/tca_report.py --run-id <run_id>
```

Pro 产物（run_id 归档）：

- micro features：`artifacts/runs/<run_id>/micro_feature/features.parquet`
- micro manifest：`artifacts/runs/<run_id>/micro_feature/manifest.json`
- TCA report：`artifacts/runs/<run_id>/tca/tca_report.json`

## 后续增强（不影响 MVP 验收）

- 测试与 CI：引入 `pytest` + GitHub Actions，把 `scripts/e2e_smoke_flow.py` 纳入自动验收。
- 证据链：每次运行生成 `run_id`，并落盘 `artifacts/run_meta.json`（包含 config 快照、依赖版本、关键指标）。
- 前端体验：修复编码/字体/CDN 依赖的降级路径；一键运行黄金路径并展示产物检查结果。
- LLM 接入：通过 `/settings` 或 `.env` 管理 base_url/api_key/model；支持“先不启用/按需启用”的切换。
