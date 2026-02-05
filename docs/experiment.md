# Experiment Matrix

> 目的：把 `docs/plan.md` / `plan.md` 里“可以跑”的路径固化成 **可复现、可对比、可落盘** 的实验条目（Exp-XXX），并记录 Smoke/Full 的结果与证据链（产物路径、关键指标）。
>
> 注意：本仓库当前默认是 **本地单机 / CPU / 离线数据（user_data/data）**；因此 Full 以“完整 timerange + 完整训练轮数（例如 LightGBM num_boost_round=220）+ 完整回测 + TCA 报告”为准，而不是追求极端耗时的超参搜索。

## Metrics Persistence Contract（当前仓库）

Full 运行必须落盘（至少）：

- Flow 元信息：`artifacts/run_meta.json` + `artifacts/runs/<run_id>/run_meta.json`
- 训练摘要：`artifacts/models/**/training_summary.json`
- 回测 zip：`user_data/backtest_results/backtest-result-*.zip`
- 回测摘要：`user_data/llm_feedback/latest_backtest_summary.json`
- micro features：`artifacts/runs/<run_id>/micro_feature/features.parquet` + `manifest.json`
- TCA v1：`artifacts/runs/<run_id>/tca/tca_report.json` +（可选）`tca_report.html`

## Experiments

### Exp-001 — Pro: KuCoin CPU no-LLM Flow + micro_feature + TCA(v1)

- Goal: 产出一条“可卖 Pro（Phase1）”的证据链：训练 + 回测 + micro features + TCA v1 schema（允许占位 proxy）
- Config (Smoke): `configs/agent_flow_pro_kucoin_smoke.json`
- Config (Full): `configs/agent_flow_pro_kucoin_full.json`
- Command (Smoke):

```bash
python scripts/agent_flow.py --config configs/agent_flow_pro_kucoin_smoke.json --steps feature micro_feature expression ml backtest tca
```

- Command (Full):

```bash
python scripts/agent_flow.py --config configs/agent_flow_pro_kucoin_full.json --steps feature micro_feature expression ml backtest tca
```

- Smoke prerequisite:
  - `pytest -q`
  - `python scripts/smoke_test.py`

- Status:
  - Smoke: PASS (2026-02-05)
  - Full: PASS (2026-02-05)
  - Prereq: `pytest -q` PASS (2026-02-05, `60 passed`)
  - Prereq: `python scripts/smoke_test.py` PASS (2026-02-05, `passed 17/17 checks`)

- Results (fill after run):
  - Latest (2026-02-05):
    - run_id: `8959d6af73d9`
    - runtime: `~0:00:06` (from `artifacts/runs/8959d6af73d9/run_meta.json`)
    - artifacts:
      - run_meta: `artifacts/run_meta.json` and `artifacts/runs/8959d6af73d9/run_meta.json`
      - micro_feature: `artifacts/runs/8959d6af73d9/micro_feature/features.parquet` and `artifacts/runs/8959d6af73d9/micro_feature/manifest.json`
      - training_summary: `artifacts/models/lightgbm_real/training_summary.json`
      - backtest_zip: `user_data/backtest_results/backtest-result-2026-02-05_14-09-15.zip`
      - backtest_summary: `user_data/llm_feedback/latest_backtest_summary.json`
      - tca_report: `artifacts/runs/8959d6af73d9/tca/tca_report.json`
      - tca_html: `artifacts/runs/8959d6af73d9/tca/tca_report.html`
    - metrics:
      - train.rmse_train: `0.0019330`
      - train.rmse_valid: `0.0220801`
      - backtest.profit_total_pct: `0.10%`
      - backtest.trades: `187`
      - backtest.sharpe: `0.6159`
      - tca.schema_version: `1.0`
      - tca.summary.trades: `187`
      - tca.summary.fees_total: `0.3740`
      - tca.orders_count: `374`
      - tca.fills_count: `374`
      - tca.is_total_quote_ccy: `0.3740`
      - tca.participation.overall: `1.37e-05`
  - Smoke (2026-02-05):
    - run_id: `7f553d779854`
    - backtest_zip: `user_data/backtest_results/backtest-result-2026-02-05_14-08-59.zip`
    - tca.orders_count: `44`
  - Previous (2026-02-05):
    - run_id: `8a5719a44ee6`
    - backtest_zip: `user_data/backtest_results/backtest-result-2026-02-05_04-13-58.zip`
  - Older (2026-02-05):
    - run_id: `aa906f78af28`
    - backtest_zip: `user_data/backtest_results/backtest-result-2026-02-05_03-29-23.zip`
  - Oldest (2026-02-05):
    - run_id: `adca4702a65e`
    - backtest_zip: `user_data/backtest_results/backtest-result-2026-02-05_01-20-08.zip`

### Exp-002 — MVP: KuCoin CPU no-LLM 黄金路径（feature → expression → ml → backtest）

- Goal: 本地单机/CPU/离线数据下可复现跑通，并生成 MVP 验收要求的产物（features/expressions/training_summary/backtest_zip/backtest_summary/run_meta）
- Config (Smoke): `configs/agent_flow_kucoin_cpu_nollm_smoke.json`
- Config (Full): `configs/agent_flow_kucoin_cpu_nollm.json`
- Command (Smoke):

```bash
python scripts/e2e_smoke_flow.py --config configs/agent_flow_kucoin_cpu_nollm_smoke.json --steps feature expression ml backtest
```

- Command (Full):

```bash
python scripts/e2e_smoke_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json --steps feature expression ml backtest
```

- Notes:
  - `scripts/e2e_smoke_flow.py` 会在缺失 `user_data/data/<exchange>/*.feather` 时自动生成 demo OHLCV（用于 smoke/开发）

- Status:
  - Smoke: PASS (2026-02-05)
  - Full: PASS (2026-02-05)
  - Prereq: `pytest -q` PASS (2026-02-05, `60 passed`)

- Results (fill after run):
  - Smoke (2026-02-05):
    - run_id: `5a39348960b3`
    - runtime: `~0:00:05` (from `artifacts/runs/5a39348960b3/run_meta.json`)
    - artifacts:
      - run_meta: `artifacts/run_meta.json` and `artifacts/runs/5a39348960b3/run_meta.json`
      - training_summary: `artifacts/models/lightgbm_real/training_summary.json`
      - backtest_zip: `user_data/backtest_results/backtest-result-2026-02-05_14-09-55.zip`
      - backtest_summary: `user_data/llm_feedback/latest_backtest_summary.json`
    - metrics:
      - train.rmse_train: `0.0019330`
      - train.rmse_valid: `0.0220801`
      - backtest.profit_total_pct: `0.05%`
      - backtest.trades: `22`
      - backtest.sharpe: `2.9387`
  - Full (2026-02-05):
    - run_id: `65d3a3031d01`
    - runtime: `~0:00:06` (from `artifacts/runs/65d3a3031d01/run_meta.json`)
    - artifacts:
      - run_meta: `artifacts/run_meta.json` and `artifacts/runs/65d3a3031d01/run_meta.json`
      - training_summary: `artifacts/models/lightgbm_real/training_summary.json`
      - backtest_zip: `user_data/backtest_results/backtest-result-2026-02-05_14-10-11.zip`
      - backtest_summary: `user_data/llm_feedback/latest_backtest_summary.json`
    - metrics:
      - train.rmse_train: `0.0019330`
      - train.rmse_valid: `0.0220801`
      - backtest.profit_total_pct: `0.10%`
      - backtest.trades: `187`
      - backtest.sharpe: `0.6159`
