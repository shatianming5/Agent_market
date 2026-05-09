# AGENTS.md — single landing page for AI agents

> 这份文件是 AI 协作 agent（Claude / Codex / opencode / hermes 等）进入本仓库时**第一个应该读的文件**。30 秒内告诉你：项目是什么、文件去哪找、能跑什么、什么不要碰。
>
> 人类读者推荐先读 [`README.md`](README.md) 与 [`docs/repo_inventory.md`](docs/repo_inventory.md)。

## 1. 项目一句话

`Agent_market` 是一个把 **LLM 表达式生成 + 量化研究 + 回测 + 部署** 串起来的工作台。当前主要驱动子系统是 `wq_brain`（WorldQuant BRAIN agentic alpha mining）。

## 2. 系统地图（按"我现在该读哪里"组织）

| 你想做的事 | 第一站 |
|---|---|
| 研究当前主驱动 alpha 挖掘逻辑 | [`src/agent_market/wq_brain/`](src/agent_market/wq_brain/) — 模块全有 docstring |
| 改 / 调度 LLM agent 的提示 | [`src/agent_market/wq_brain/prompts/agent_brief.md`](src/agent_market/wq_brain/prompts/agent_brief.md) |
| 找仓库目录索引（最权威） | [`docs/repo_inventory.md`](docs/repo_inventory.md) |
| 找系统分层 / 模块归属（核心心智模型） | [`docs/architecture.md`](docs/architecture.md) |
| 看每篇文档的状态（current / historical / deprecated） | [`docs/INDEX.md`](docs/INDEX.md) |
| 找 CLI 入口 | [`scripts/README.md`](scripts/README.md) |
| 看用户视角快速开始 | [`README.md`](README.md) |
| 看进行中的 review loop 评分历史 | [`AUTO_REVIEW.md`](AUTO_REVIEW.md) (loop 持久化日志，仅供参考) |

## 3. 主要入口（黄金命令）

```bash
# 后端服务
uvicorn server.main:app --host 127.0.0.1 --port 8000

# 离线主流水线（feature → expression → ml → backtest）
python scripts/agent_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json \
    --steps feature expression ml backtest

# Factor Lab 研究 CLI（最常用）
python scripts/factor_lab.py <subcommand>      # data | features | mine | validate | backtest | rank-export | rl | combo | deploy | hub

# WQ BRAIN agentic mining loop
python scripts/wq_brain.py simulate --tag <tag> --expr "<FASTEXPR>"
python scripts/wq_brain.py pool submit-worker --tag <tag> --max 20 --one-per-cluster

# Strategy-level mining
python scripts/strategy_miner.py --config configs/strategy_miner_default.json

# 测试
pytest -q
```

## 4. 必读环境变量

| 变量 | 用途 | 说明 |
|---|---|---|
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` | LLM provider | 推荐 OpenAI 兼容接口 |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | 兼容别名 | 同上，hermes 老命名 |
| `WQ_EMAIL` / `WQ_PASSWORD` | WorldQuant BRAIN 凭据 | `wq_brain` **必需**；见 `src/agent_market/wq_brain/client.py:364-370` |
| `WQ_API_BASE` | WQ API 根 (可选) | 默认 `https://api.worldquantbrain.com` |
| `WQ_MAX_CONCURRENT` | WQ 并发上限 (可选) | 默认 `3` |
| `WQB_DATA_BACKEND` | OHLCV backend | `stooq` (默认) / `yfinance` / `auto`；用于 `wq_brain fetch-data` |
| `WQ_QUOTA_SIM_HARD` / `WQ_QUOTA_SUBMIT_HARD` | 配额硬上限 | 当 WQ 提高额度时手动放宽 |
| `AGENT_MARKET_API_KEY` | server 鉴权 | 生产环境启用 |
| `AGENT_MARKET_ARTIFACTS_ROOT` | 可重定向 artifacts/ | 测试隔离时常用 |

`.env` 是当前生效的；`.env.minimax.bak*`、`.env.cliproxy` 是历史备份，**不要从中读凭据**。

## 5. 不要碰的目录（运行时产出，不是仓库内容）

- `artifacts/` — 各 run 的产物、模型、factor_lab 与 rank_portfolio 的输出。会很大。
- `runtime_configs/`、`runtime_logs/`、`runtime_manifests/` — 服务运行时快照。
- `logs/` — 日志归档。
- `.tmp/`、`.pytest_cache/`、`.venv*/`、`.opencode/`、`.playwright-mcp/` — 工具缓存。
- `freqtrade/` — vendored snapshot，**只读**；不要在这里改 freqtrade。
- `user_data/` — Freqtrade 工作区（含真实 OHLCV / 回测产物）；改之前确认目的。
- `ws_production/` — 独立 production-workspace 实验区（adaptive_params / auto_improver / ...），与主流水线 *基本独立*；入门读 `ws_production/GUIDE.md`。
- `workspace/` — 模板源（被 `create_workspace.py` 拷贝到新生成的 `ws_<id>/`），不是当前推荐的研究入口；只读为主。

`.gitignore` 是权威清单。如果你看到一个 tracked 文件落在 `.gitignore` 范围内，那是历史遗留 — 不要因此推断它是"被刻意保留的当前事实"。

## 6. 测试 / 验证

- `pytest -q` — 全套，~5 分钟（取决于慢测试）
- `pytest tests/test_wq_brain_*.py -q` — wq_brain 子集（487 项，<2 秒）
- `python scripts/smoke_test.py` — 连通性 + 黄金路径冒烟

## 7. 协作规约（项目层）

- **真实数据 / 真实配置**：禁止用模拟 / 占位 / "默认配置" 替代真实路径与凭据。冒烟脚本须以真实 config + 真实 data root 跑一次最小闭环。
- **远端执行**：远程 SSH 命令统一用 `ssh ${SSH_USER}@${SSH_HOST} "set -Eeuo pipefail; cd ${PROJECT_DIR} && <cmd>"` 模板；占位变量缺失时停止并告知。
- **简体中文**：用户对话默认用简体中文回答。
- **不生成总结文档**：不主动新建 README / 总结说明 / 使用指南；除非用户明确要求。本 `AGENTS.md` 是对该规则的例外，因为它是 AI agent 的入口契约。

## 8. WQ BRAIN agent 子系统的关键事实

- 8 轮 review loop 已把 `pool submit-worker` 路径打到 8.4/10 "controlled autonomous OK"。详见 `AUTO_REVIEW.md`（49 KB review 日志）。
- **安全使用方式**：单进程 `pool submit-worker --tag <tag> --max 20`，**不要**与 `sync-status` / `dedup` / `resubmit-all` / `scan --auto-submit` 并发。
- 状态机：`UNSUBMITTED → ACTIVE` / `LOCAL_BLOCKED` / `SELF_CORR_BLOCKED` / `REJECTED` / `VERIFICATION_FAILED`。`pool sync-status` 默认保留 LOCAL_BLOCKED / SELF_CORR_BLOCKED；用 `--reset-local-blocks` 强制覆盖。
- 487 个测试覆盖了 multi-blocker override、authoritative_ids、status precedence、concurrent-writer merge 等。

## 9. 变更前先读

如果你即将动以下任何一处，请先 grep + 读对应文档：

- `src/agent_market/wq_brain/pool.py::AlphaPool._save` — 跨进程 fcntl + 状态优先级 merge；改之前读 `AUTO_REVIEW.md` 中第二轮 R2-CRIT / R3-CRIT。
- `scripts/wq_brain.py` (~2200 行) — CLI 单文件壳，分支多；先确认子命令以避免重复实现。
- `src/agent_market/agent_flow.py` — 主流水线编排，所有 step 经此进出。
- `docs/plan.md` (MVP, 已闭环) vs `docs/proposals/agent_market_proposal.md` (Proposal, 部分 PARTIAL) — 不同口径；先看 `docs/project_status.md`。**注**：根 `/plan.md` 现在只是兼容性指针。

## 10. 仍未收敛的项（不要在不知情时挑战）

- `python scripts/wq_brain.py pool resubmit-all` 与 `python scripts/wq_brain.py scan --auto-submit` 已被 R2 review gate (`--legacy-unsafe`) 锁住：默认拒绝执行，必须显式加 `--legacy-unsafe` 才会走旧代码（无 quota / 无 local-jaccard / 无 self-corr / 无持久化）。生产请用 `python scripts/wq_brain.py pool submit-worker`。
- `src/agent_market/wq_brain/vendor_quantgpt/mcp_server.py` 包含直接 `submit_alpha` / `auto_submit` 调用，**但当前仓库内无任何模块 import 它**（已 grep 验证），属于 dead vendored snapshot；如未来要启用 MCP，必须先把同样的 `--legacy-unsafe` 门类应用到 mcp_server。
- `src/agent_market/` 顶层 14 个平铺模块 + 11 个子包未分层（core / flow / domain），任何重排都需先扫所有 import 路径。
- 计划口径双轨（`docs/proposals/agent_market_proposal.md` 是 Proposal vs `docs/plan.md` MVP），通过根 `plan.md` 兼容性指针引导；未来可能合并。

---

文档版本：与 `AUTO_REVIEW.md` 第 8 轮（2026-05-09）同步；如有冲突以仓库当前代码为准。
