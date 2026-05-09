# docs/legacy/ — 归档的历史文档

> 本目录存放**已不再 current** 但出于历史索引价值需要保留的文档。如果你正在做 review / planning / git archeology，这里可能有线索；如果你在执行当前的运营操作，请优先看 `docs/INDEX.md` 中标注 CURRENT 的文件。

## 索引

| 文件 | 原位置 | 移入时间 | 一句话用途 |
|---|---|---|---|
| [`HARNESS_SPEC.md`](HARNESS_SPEC.md) | `/HARNESS_SPEC.md` (root) | 2026-05-09 | 早期 harness 验收规格；可作为旧版引用，不代表当前评测口径 |
| [`HARNESS_ACCEPTANCE.md`](HARNESS_ACCEPTANCE.md) | `/HARNESS_ACCEPTANCE.md` (root) | 2026-05-09 | 早期 harness 验收记录；同上 |
| [`RALPH_PROMPT.md`](RALPH_PROMPT.md) | `/RALPH_PROMPT.md` (root) | 2026-05-09 | 旧版 agent system prompt；当前的 agent prompt 在 `src/agent_market/wq_brain/prompts/agent_brief.md` |

## 移入原因

根目录原本同时存在 **当前用户文档** (`README.md`, `AGENTS.md`) 与 **历史规格 / 旧 prompt**，导致新读者难以分辨哪些是"现在的事实"。第二轮 project-structure review loop（`AUTO_REVIEW.md` 同日）之后，根目录只保留：

```
AGENTS.md       # AI agent 入口
AUTO_REVIEW.md  # review loop 评分日志（持续追加）
plan.md         # 兼容性指针（指向 docs/proposals/agent_market_proposal.md 与 docs/plan.md）
README.md       # 用户视角快速开始
```

其余历史文档统一归档到本目录。后续 review（同日）将 Proposal 从根 `plan.md` 移到 `docs/proposals/agent_market_proposal.md`；本目录与 Proposal 都作为 docs/ 子目录归档管理。
