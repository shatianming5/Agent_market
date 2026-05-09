# docs/ INDEX — 文档状态与读取顺序

> `docs/` 下当前 24 篇 .md（含 `architecture.md`，不含本 INDEX）+ 子目录 `docs/plans/` 2 篇 + `docs/proposals/` 1 篇 + `docs/legacy/` 4 篇；状态标签让 agent / 新读者第一眼分辨哪些是**当前事实**，哪些是**历史/证据/差距**记录。状态会过期 — 如有冲突以仓库当前代码为准。
>
> 推荐读取顺序：先 **CURRENT** 类，后 **EVIDENCE/STATUS**，最后 **PROPOSAL** 或具体子系统 RUNBOOK。

## 状态分类

- **CURRENT** — 描述当前能跑、当前推荐路径或当前架构，仍是事实。
- **EVIDENCE** — 计划 ↔ 实现的证据 / 验收日志 / Claim 映射；不能凭它推断未来计划，但可以凭它确认已发生的状态。
- **STATUS** — 仓库处境快照（含计划口径解释）；定期过期，看顶部时间。
- **RUNBOOK** — 操作手册 / 一次性运行指南；先确认对应代码仍存在再用。
- **PROPOSAL** — 拟定方案；可能尚未实现或部分实现。
- **HISTORICAL** — 一次性恢复 / 升级报告 / 旧版差距审计；归档参考。

## 索引

| 文档 | 状态 | 一句话用途 | 最近更新 |
|---|---|---|---|
| [`architecture.md`](architecture.md) | CURRENT | 系统分层 / 模块归属心智模型（L1 编排 / L2 域 / L3 核心 / L4 产物）；**改动前必读** | 2026-05-09 |
| [`repo_inventory.md`](repo_inventory.md) | CURRENT | 仓库目录 / 入口 / 核心模块 tour（**第一站**） | 持续维护 |
| [`project_status.md`](project_status.md) | STATUS | "当前处境 + 计划差距"；解释 plan 双轨 + 2026-05-09 结构 cleanup commits | 2026-05-09 |
| [`deep_dive.md`](deep_dive.md) | CURRENT | 30 分钟通览：UI → API → flow → artifacts | — |
| [`plan.md`](plan.md) | CURRENT (MVP) | MVP 落地计划；`feature → expression → ml → backtest` | — |
| [`product_90d.md`](product_90d.md) | CURRENT (一键闭环) | 90 天闭环：micro → compile → eval → train → backtest → tca → report | — |
| [`llm_pipeline.md`](llm_pipeline.md) | CURRENT (子系统) | LLM 表达式生成 + Top-N 进化（`scripts/freqai_expression_agent.py`） | — |
| [`real_backtest_plan.md`](real_backtest_plan.md) | RUNBOOK | "可逐一调试"的真实流程回测模板 | — |
| [`agentic_first_run_bigmodel.md`](agentic_first_run_bigmodel.md) | RUNBOOK | BigModel GLM 跑首条 agentic 策略 | — |
| [`strategy_factory_operator_runbook.md`](strategy_factory_operator_runbook.md) | RUNBOOK | factor flow → strategy miner → promotion | — |
| [`strategy_miner_alpha_runbook.md`](strategy_miner_alpha_runbook.md) | RUNBOOK | 分组 alpha 搜索方案 | — |
| [`strategy_miner_maxpower.md`](strategy_miner_maxpower.md) | RUNBOOK | "最大力度"实跑配置 (no-template + multiagent) | — |
| [`okx_intraday_download_plan.md`](okx_intraday_download_plan.md) | PROPOSAL | OKX USDT-SWAP rank-factor 数据集生产方案 | — |
| [`quantconnect_hybrid_plan.md`](quantconnect_hybrid_plan.md) | PROPOSAL | LEAN 桥接验证层（独立 event-driven 执行检查） | — |
| [`opencode_strategy_factory_architecture.md`](opencode_strategy_factory_architecture.md) | PROPOSAL (架构方向) | 把仓库整理成分层策略工厂的方向 | — |
| [`strategy_miner_agentic_upgrade.md`](strategy_miner_agentic_upgrade.md) | PROPOSAL (已落地节选) | OpenCode-driven agentic upgrade 方案（部分已实现） | — |
| [`experiment.md`](experiment.md) | EVIDENCE | 实验矩阵：Exp-XXX run_id / 指标 / 产物路径 | — |
| [`claim_evidence.md`](claim_evidence.md) | EVIDENCE | Claim → Evidence 映射（是否可证明） | — |
| [`mohu.md`](mohu.md) | EVIDENCE (closed) | Missing/Ambiguous backlog（已清零，保留历史） | — |
| [`verify_log.md`](verify_log.md) | EVIDENCE | 逐项验收命令 + 结果日志 | — |
| [`plan_changelog.md`](plan_changelog.md) | EVIDENCE (stale 2026-02-05) | plan ↔ evidence ↔ implementation 循环的计划改写记录；从 2026-02-05 后未更新 | 2026-02-05 |
| [`plan_gap_planmd.md`](plan_gap_planmd.md) | EVIDENCE | 原根 `plan.md` / Proposal（现 `docs/proposals/agent_market_proposal.md`）逐章差距审计 | 2026-02-05 |
| [`plan_gap_planmd_partial_missing.md`](plan_gap_planmd_partial_missing.md) | EVIDENCE | 上面那份的 PARTIAL/MISSING 抽取视图 | 2026-02-05 |
| [`strategy_miner_recovery_report.md`](strategy_miner_recovery_report.md) | HISTORICAL | 2026-03-05 strategy_miner `best_reward=-inf` 修复报告 | 2026-03-05 |

### `docs/plans/` 子目录

| 文档 | 状态 | 一句话用途 | 最近更新 |
|---|---|---|---|
| [`plans/2026-04-03-review-deferred-items.md`](plans/2026-04-03-review-deferred-items.md) | HISTORICAL | 2026-04-03 review 后的延期事项清单 | 2026-04-03 |
| [`plans/2026-04-05-harness-v2-upgrade.md`](plans/2026-04-05-harness-v2-upgrade.md) | HISTORICAL | 2026-04-05 harness v2 升级方案 | 2026-04-05 |

### `docs/proposals/` 子目录（2026-05-09 plan-name 拆分）

| 文档 | 状态 | 一句话用途 | 移入时间 |
|---|---|---|---|
| [`proposals/agent_market_proposal.md`](proposals/agent_market_proposal.md) | PROPOSAL | 早期完整 Proposal（原根 `plan.md`，656 行）；包含 Factor Compiler DSL + 微观结构 + TCA schema + 90 天路线 | 2026-05-09 |

### `docs/legacy/` 子目录（2026-05-09 root cleanup 归档）

| 文档 | 状态 | 一句话用途 | 移入时间 |
|---|---|---|---|
| [`legacy/README.md`](legacy/README.md) | INDEX | 归档目录的索引 + 移入原因 | 2026-05-09 |
| [`legacy/HARNESS_SPEC.md`](legacy/HARNESS_SPEC.md) | HISTORICAL | 早期 harness 验收规格 | 2026-05-09 |
| [`legacy/HARNESS_ACCEPTANCE.md`](legacy/HARNESS_ACCEPTANCE.md) | HISTORICAL | 早期 harness 验收记录 | 2026-05-09 |
| [`legacy/RALPH_PROMPT.md`](legacy/RALPH_PROMPT.md) | HISTORICAL | 旧版 agent system prompt | 2026-05-09 |

## 计划口径双轨（重要）

仓库存在两份"计划"，已 2026-05-09 通过命名澄清：

- **MVP** — [`docs/plan.md`](plan.md)：本地单机 / CPU / 离线数据；验收 = smoke + e2e + 产物落盘。**当前已闭环**。
- **Proposal** — [`docs/proposals/agent_market_proposal.md`](proposals/agent_market_proposal.md)：Factor Compiler DSL + 微观结构 + TCA schema + 90 天路线。**部分 PARTIAL**，详见 `plan_gap_planmd.md`。
- 根 [`/plan.md`](../plan.md)：兼容性指针，引导到上面两份。

新读者：先把 `docs/plan.md` 当成"事实"，把 `docs/proposals/agent_market_proposal.md` 当成"还没全做完的计划"。两者的差距由 `docs/project_status.md` 解释。

> **历史文档兼容性**：`docs/verify_log.md` / `docs/mohu.md` / `docs/plan_changelog.md` / `docs/plan_gap_planmd*.md` 中早期写 "`plan.md`" 的地方，都是指现在的 `docs/proposals/agent_market_proposal.md`。这些历史文档不再大规模 rewrite，靠本节统一映射理解。

## Reviewer 反馈

文档完整度问题在 [`/AUTO_REVIEW.md`](../AUTO_REVIEW.md) 第 9 轮 review 中讨论；本 INDEX 是对该轮 fix#5 的直接产物。
