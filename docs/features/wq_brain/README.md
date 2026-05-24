# WQ BRAIN Agentic Alpha Mining

[返回主 README](../../../README.md) · [返回功能索引](../README.md)

`wq_brain` 是当前仓库的主驱动子系统，用于 WorldQuant BRAIN alpha mining、FASTEXPR 生成/验证、模拟、提交、pool 管理和 agentic 研究循环。

## 代码入口

| 位置 | 用途 |
|---|---|
| `scripts/wq_brain.py` | 统一 CLI 入口，包含 auth / simulate / pool / agent / colony 等子命令 |
| `src/agent_market/wq_brain/` | 子系统核心实现 |
| `src/agent_market/wq_brain/prompts/agent_brief.md` | LLM agent 主提示 |
| `AUTO_REVIEW.md` | review loop 历史与生产路径风险说明 |
| `tests/test_wq_brain_*.py` | WQ BRAIN 快速回归测试 |

## 主要能力

- WQ 凭据验证：`auth`
- FASTEXPR 本地校验、失败诊断、变异建议：`validate`、`mutate`
- 单 alpha WQ 模拟与提交：`simulate`、`submit`
- WQ self-correlation gate 与本地近重复预检：`pre-check`、`pre-check-local`
- alpha pool 管理：`pool list`、`pool status`、`pool sync-status`、`pool submit-worker`
- 安全提交 worker：quota、local Jaccard、self-correlation、状态持久化、并发 writer merge
- 本地数据链路：`fetch-data`、`update-data`、`kaggle-fetch`、`kaggle-import`
- 本地模拟与防过拟合：`local-simulate`、`anti-overfit`、`calibrate-local`、`seed-calibration`
- 研究辅助：论文搜索、FASTEXPR docs、SymPy、web search、URL fetch、worldquant-skill 搜索
- LLM/agent loop：`ping-llm`、`endpoint failover`、`agent`、`colony`
- 报告与 review：`report`、`review`

## 推荐命令

```bash
python scripts/wq_brain.py auth
python scripts/wq_brain.py docs
python scripts/wq_brain.py validate --expr "<FASTEXPR>"
python scripts/wq_brain.py simulate --tag <tag> --expr "<FASTEXPR>"
python scripts/wq_brain.py pool status --tag <tag>
python scripts/wq_brain.py pool submit-worker --tag <tag> --max 20 --one-per-cluster
```

## 必需环境变量

| 变量 | 说明 |
|---|---|
| `WQ_EMAIL` | WorldQuant BRAIN 邮箱 |
| `WQ_PASSWORD` | WorldQuant BRAIN 密码 |
| `WQ_API_BASE` | 可选，默认 `https://api.worldquantbrain.com` |
| `WQ_MAX_CONCURRENT` | 可选，默认 `3` |
| `WQB_DATA_BACKEND` | `stooq` / `yfinance` / `auto` |
| `WQ_QUOTA_SIM_HARD` / `WQ_QUOTA_SUBMIT_HARD` | 配额硬上限 |

如果启用 agent loop，还需要 LLM 环境变量：

```bash
OPENAI_BASE_URL=...
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

兼容别名：`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`。

## 产物和状态

WQ BRAIN 相关产物通常落在 `artifacts/` 下对应 tag/run 的目录，pool 状态由子系统持久化管理。提交 worker 会保存 outcome，避免只靠日志判断真实提交状态。

## 安全边界

- 生产路径使用单进程 `pool submit-worker --tag <tag> --max 20`。
- 不要和 `sync-status`、`dedup`、`resubmit-all`、`scan --auto-submit` 并发操作同一个 pool。
- `pool resubmit-all` 和 `scan --auto-submit` 是 legacy unsafe 路径，默认被 review gate 约束；生产不要绕过。
- 改 `AlphaPool._save` 前先读 `AUTO_REVIEW.md` 中状态优先级、fcntl merge、concurrent writer 相关 review。

## 验证

```bash
pytest tests/test_wq_brain_*.py -q
python scripts/wq_brain.py ping-llm
python scripts/wq_brain.py auth
```

