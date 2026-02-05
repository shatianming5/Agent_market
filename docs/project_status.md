# Project Status（当前处境与计划差距）

Updated: 2026-02-05

> 本文是“现在仓库处境”与“计划差距”的入口页；所有结论均以仓库内已落盘的文档与产物为准。

## 快速入口（建议按此顺序阅读）

1) MVP 落地计划（验收口径）：`docs/plan.md`
2) 实验矩阵与结果（run_id / 指标 / 产物路径）：`docs/experiment.md`
3) Claim → Evidence 映射（是否可证明）：`docs/claim_evidence.md`
4) Missing/Ambiguous backlog（已清零但保留历史）：`docs/mohu.md`
5) 90 天闭环产品（一键闭环脚本 + 验收/导出/前端查看）：`docs/product_90d.md`
6) Proposal（更大范围的计划）：`plan.md`
7) Proposal 的逐章差距审计（不省略）：`docs/plan_gap_planmd.md`
8) Proposal 的差距抽取视图（仅 PARTIAL/MISSING）：`docs/plan_gap_planmd_partial_missing.md`

## 仓库快照（source of truth）

- Branch: `main`
- Remote: `origin git@github.com:shatianming5/Agent_market.git`
- 关键提交（证据链/文档入口）：
  - `2974e5b`：实现层面关闭 `plan.md` 主要工程缺口（Factor Compiler + microstructure + TCA + TCA 成本分解等）
  - `020a6a6`：强制纳入 reference model artifacts（原本被 `.gitignore` 忽略）
  - `10c5b12`：强制纳入实验“证据链”关键产物（原本被 `.gitignore` 忽略）
  - `11601a1`：补齐 docs 状态入口页与 “仅 PARTIAL/MISSING” 的 gap 抽取视图

## 计划口径澄清（非常重要）

本仓库存在两份“计划”，口径不同：

- `docs/plan.md`：**MVP 落地计划**（本地单机/CPU/离线数据，验收=smoke + e2e + 产物落盘）。该口径下已闭环（见下文）。
- `plan.md`：**Proposal**（Factor Compiler DSL + 微观结构特征表 + TCA schema + Flow 扩展 + 90 天路线）。该口径下仍存在大量 `PARTIAL`（例如：类型系统深化、TCA 深度指标、LLM FactorSpec 输出与反馈闭环等），详见 `docs/plan_gap_planmd.md`。

## ABC 闭环现状（基于仓库文档与落盘证据）

### A）Plan ↔ Implementation（以 `docs/plan.md` 为验收口径）

- `docs/mohu.md`：`## Missing` 与 `## Ambiguous` **均无未勾选项**
- `docs/verify_log.md`：包含逐项验收日志与命令
- 运行根目录可配置：通过环境变量将 `artifacts/` 与 `user_data/` 指向隔离目录，避免测试/开发运行覆盖已纳入 git 的证据链文件（见 `src/agent_market/paths.py` 与 `tests/conftest.py`）。

### B）Experiments（以 `docs/experiment.md` 为矩阵口径）

- `Exp-001` / `Exp-002`：Smoke/Full 均记录为 PASS（2026-02-05）
- 关键 run_id（用于追溯证据链）：
  - Exp-001 Full：`8959d6af73d9`
  - Exp-002 Full：`65d3a3031d01`

### C）Claims ↔ Evidence（以 `docs/claim_evidence.md` 为口径）

- Claim-001..005：全部 `Judgment: Yes`，且 `Gap: None`

## “plan.md（Proposal）”的 gap（严格以审计文档为准）

- 全量逐章审计（不省略）：`docs/plan_gap_planmd.md`
- 仅 gap 抽取视图：`docs/plan_gap_planmd_partial_missing.md`
- 当前无明确 `MISSING`（仅剩 `PARTIAL`）：以 `docs/plan_gap_planmd_partial_missing.md` 为准

## 证据链产物（已纳入 git 的部分）

> 默认情况下 `.gitignore` 会忽略 `artifacts/` 与大部分 `user_data/`；为保证“plan-grade 可复现”，我们强制纳入了关键产物（见上述提交）。

- Model（reference）：
  - `artifacts/models/lightgbm_real/*`
  - `artifacts/models/rl_real/*`
- Evidence（runs + backtest + summaries）：
  - `artifacts/run_meta.json`
  - `artifacts/runs/<run_id>/**`（包含 micro_feature 与 tca 报告）
  - `user_data/backtest_results/backtest-result-2026-02-05_*.zip`（选取与 Exp-001/Exp-002 对应的结果）
  - `user_data/llm_feedback/latest_backtest_summary.json`

## 最小复核命令（本地）

```bash
pytest -q
python scripts/smoke_test.py
python scripts/e2e_smoke_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json
```
