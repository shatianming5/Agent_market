# Strategy Miner MaxPower（no-template + multiagent）

本页提供一套“最大力度”实跑配置与复现命令，用于确保 strategy miner：
- **强制 no-template**：禁用 `strategy_miner` 的 template provider 与模板兜底；leaderboard 过滤 template 来源
- **multiagent/subagent**：planner/coder/reviewer/backtester 多角色协作；每候选写入 agent trace
- **repair 回合**：`repair_attempts` 可配（启用时 clamp 到 **3–8**）+ 失败分类 + 定向修复/重试
- **真实回测闭环**：静态校验 → freqtrade backtesting → 风险门禁（min_trades/DD/winrate）→ leaderboard

## 配置
- `configs/strategy_miner_maxpower.json`
  - `budget.provider`: `opencode`（或 `auto`）
  - `budget.model`: `opencode/gpt-5-nano`（可替换为你的 OpenCode 模型）
  - `budget.multiagent_enabled=true`
  - `budget.candidates_per_iteration=4`
  - `budget.max_parallel_candidates=2`
  - `budget.max_parallel_roles=2`（reviewer/backtester 并行）
  - `budget.repair_attempts=5`（启用后实际在 3–8 之间）
  - `backtest.freqtrade_config=user_data/config_freqai.json`
  - `backtest.timerange=20251229-20260202`（覆盖仓库内置离线 1h OHLCV）
  - `evaluation.min_trades=5`（门禁参数可按需调整）

## 依赖
需要 `freqtrade` 才能做真实回测。

推荐：
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-full.txt
```

## 实跑（推荐）
```bash
source .venv/bin/activate
python scripts/strategy_miner.py --config configs/strategy_miner_maxpower.json --max-iterations 1 -v
```
输出中会包含 `run_id`。

## 验收
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
