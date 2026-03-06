# Strategy Miner MaxPower（no-template + multiagent + 实跑）

本页提供一套“最大力度”实跑配置与复现命令，用于确保 strategy miner：
- **强制 no-template**：禁用 `strategy_miner` 的 template provider 与模板兜底；leaderboard 过滤 template 来源
- **multiagent/subagent**：planner/coder/reviewer/backtester 多角色协作；每候选写入 agent trace
- **repair 回合**：`repair_attempts` 可配（启用时 clamp 到 **3–8**）+ 失败分类 + 定向修复/重试
- **完整验证闭环**：静态校验（语法/anti-lookahead）→ freqtrade backtesting → 风险门禁（min_trades/DD/winrate）→ leaderboard

## 配置文件
- `configs/strategy_miner_maxpower.json`
  - `budget.provider`: `openai_compatible`（当前已切换为 LLM API 路线）
  - `budget.model`: `gpt-5.2`
  - `budget.multiagent_enabled=true`
  - `budget.candidates_per_iteration=2`
  - `budget.max_iterations=4`
  - `budget.repair_attempts=5`（启用后实际 clamp 到 3–8）
  - `budget.max_parallel_candidates=2`
  - `budget.max_parallel_roles=1`（更稳；如需加速可调到 2）
  - `backtest.freqtrade_config=user_data/config_freqai.json`
  - `backtest.timerange=20260101-20260201`
  - `evaluation.min_trades=5` / `evaluation.max_abs_drawdown=80.0`


当前默认就是外部 LLM 路线（`openai_compatible + gpt-5.2`）。
请在运行前设置：

```bash
export OPENAI_API_KEY="<your-key>"
export OPENAI_API_BASE="http://10.150.240.117:38889"
# 兼容变量（代码会优先读取 OPENAI_BASE_URL / LLM_BASE_URL）
export OPENAI_BASE_URL="$OPENAI_API_BASE"
```

> 代码会自动把 `OPENAI_API_BASE` 兼容映射到 `OPENAI_BASE_URL`，并处理 `/v1` 路径。

## 依赖
需要 `freqtrade` 才能做真实回测。

- `provider=heuristic`：不需要 LLM/`opencode`，完全离线可跑（默认）。
- `provider=opencode_cli`：需要可用的 `opencode` 模型/额度。

推荐：
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-full.txt
```

## 实跑（推荐）
```bash
source .venv/bin/activate
python scripts/run_strategy_miner_recovery.py --config configs/strategy_miner_maxpower.json -v
```

如果你使用仓库自带环境（示例）：
```bash
.venv312/bin/python scripts/run_strategy_miner_recovery.py --config configs/strategy_miner_maxpower.json -v
```

输出中会包含 `run_id`、`best_reward`、`leaderboard` 路径与 Top3 摘要。

## 验收/验证
通过条件（与本次验收对齐）：
- `best_reward` 为有限值且 **不等于 `-inf`**
- leaderboard `top3` 中不存在模板候选（`TemplateRsiStrategy` / template 来源被过滤）

验证脚本：
```bash
source .venv/bin/activate
python scripts/verify_strategy_miner_run.py --run-id <run_id> --top 3
```

## 产物位置
默认在：`artifacts/runs/<run_id>/strategy_miner/`（受环境变量 `AGENT_MARKET_RUNS_ROOT` 影响）

常见文件：
- `checkpoint.json`
- `leaderboard.json`
- `candidates/iter_*/<Strategy>.py|.json`
- `backtests/iter_*/<Strategy>/summary.json`
- `agent_traces/iter_*/cand_*/{planner,coder,reviewer,backtester}.json`
