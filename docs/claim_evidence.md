# Claim Evidence

## Claim-001: MVP 黄金路径端到端可复现并落盘产物
- Required:
  - 本地单机 / CPU / 离线数据可跑通：`feature → expression → ml → backtest`
  - 必须生成（按 `docs/plan.md`）：
    - `artifacts/run_meta.json`（包含 `run_id`/config hash/版本信息/产物路径）
    - `user_data/freqai_features_real.json`
    - `user_data/freqai_expressions_selected.json`
    - `artifacts/models/**/training_summary.json`
    - `user_data/backtest_results/backtest-result-*.zip`
    - `user_data/llm_feedback/latest_backtest_summary.json`
- Evidence:
  - Exp-002 Full (2026-02-05): `run_id=65d3a3031d01`
    - run_meta: `artifacts/runs/65d3a3031d01/run_meta.json`
    - backtest_zip: `user_data/backtest_results/backtest-result-2026-02-05_14-10-11.zip`
    - training_summary: `artifacts/models/lightgbm_real/training_summary.json`
- Judgment: Yes
- Gap: None

## Claim-002: API 冒烟验收可通过（不依赖启动服务）
- Required:
  - `python scripts/smoke_test.py` 返回 0，覆盖 `/health`、`/settings`、jobs 错误封装、以及 `/run/*` 的 validation-only checks
- Evidence:
  - Exp-001 prereq (2026-02-05): `python scripts/smoke_test.py` PASS（`passed 17/17 checks`）
- Judgment: Yes
- Gap: None

## Claim-003: E2E 冒烟（含产物检查）可通过
- Required:
  - `python scripts/e2e_smoke_flow.py --config configs/agent_flow_kucoin_cpu_nollm_smoke.json` PASS
  - `python scripts/e2e_smoke_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json` PASS
- Evidence:
  - Exp-002 Smoke (2026-02-05): `run_id=5a39348960b3`（`e2e_smoke_flow.py` 输出 `OK`）
  - Exp-002 Full (2026-02-05): `run_id=65d3a3031d01`（`e2e_smoke_flow.py` 输出 `OK`）
- Judgment: Yes
- Gap: None

## Claim-004: Pro 扩展（micro_feature + TCA v1）链路可跑通并落盘
- Required:
  - Full Flow 产物包含：
    - micro features：`artifacts/runs/<run_id>/micro_feature/features.parquet` + `manifest.json`
    - TCA report：`artifacts/runs/<run_id>/tca/tca_report.json`（可选 HTML）
- Evidence:
  - Exp-001 Full (2026-02-05): `run_id=8959d6af73d9`
    - micro_feature: `artifacts/runs/8959d6af73d9/micro_feature/features.parquet`
    - tca_report: `artifacts/runs/8959d6af73d9/tca/tca_report.json`
- Judgment: Yes
- Gap: None

## Claim-005: 实验产物可按 run_id 追溯（Metrics Persistence Contract 满足）
- Required:
  - 每次 Flow 运行至少落盘：
    - `artifacts/runs/<run_id>/run_meta.json`（含步骤状态 + 产物索引）
    - 回测 zip（可追溯到具体文件名）
    -（Pro）micro_feature 与 tca_report 路径稳定存在
- Evidence:
  - Exp-001 Full: `artifacts/runs/8959d6af73d9/run_meta.json`（含 micro_feature/tca_report/backtest_zips）
  - Exp-002 Full: `artifacts/runs/65d3a3031d01/run_meta.json`（含 backtest_zips/training_summaries）
- Judgment: Yes
- Gap: None

