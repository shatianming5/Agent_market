"""Prompt templates for the strategy mining agent."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_strategy_gen_prompt(
    *,
    iteration: int,
    sandbox_path: str,
    freqtrade_config: str,
    timerange: str,
    history: List[Dict[str, Any]],
    best_score: float,
    best_strategy_code: Optional[str] = None,
    elite_summaries: Optional[List[Dict[str, Any]]] = None,
    failure_summary: Optional[str] = None,
    candidate_idx: Optional[int] = None,
    candidates_per_iteration: Optional[int] = None,
    provider: str = "auto",
    market_profile: Optional[str] = None,
) -> str:
    history_section = ""
    if history:
        lines = []
        for h in history[-5:]:
            lines.append(
                f"  - Iteration {h.get('iteration', '?')}: "
                f"sharpe={h.get('sharpe', 'N/A')}, "
                f"sortino={h.get('sortino', 'N/A')}, "
                f"profit={h.get('profit_pct', 'N/A')}%, "
                f"trades={h.get('trades', 'N/A')}, "
                f"winrate={h.get('winrate', 'N/A')}, "
                f"profit_factor={h.get('profit_factor', 'N/A')}, "
                f"diagnosis: {h.get('diagnosis', 'N/A')}"
            )
        history_section = "\n## Previous Iterations (most recent 5)\n" + "\n".join(lines) + "\n"

    best_section = ""
    if best_strategy_code and best_score > float("-inf"):
        best_section = (
            f"\n## Current Best Strategy (sharpe={best_score:.4f})\n"
            f"```python\n{best_strategy_code}\n```\n"
        )

    kb_section = ""
    if elite_summaries:
        elite_lines = []
        for e in elite_summaries[:3]:
            elite_lines.append(
                f"  - {e.get('name', '?')}: sharpe={e.get('reward', 0):.4f}, "
                f"profit={e.get('profit_pct', '?')}%, "
                f"trades={e.get('trades', '?')}, winrate={e.get('winrate', '?')}"
            )
        kb_section += "\n## Elite Strategy Archive (top performers)\n" + "\n".join(elite_lines) + "\n"

    if failure_summary and failure_summary != "No recorded failures.":
        kb_section += "\n## Known Failure Patterns (avoid these)\n" + failure_summary + "\n"

    multi_header = ""
    multi_rules = ""
    if candidate_idx is not None and candidates_per_iteration:
        multi_header = f"\n## Candidate {int(candidate_idx) + 1}/{int(candidates_per_iteration)}\n"
        multi_rules = (
            "\n## Multi-candidate rules\n"
            "- This is one of several candidates generated in parallel for the same iteration.\n"
            "- Make this strategy meaningfully different from other candidates (different indicators/logic).\n"
            "- Use a UNIQUE class name to avoid collisions. Recommended: suffix with `_cand{idx}`.\n"
        ).format(idx=int(candidate_idx))

    market_section = ""
    if market_profile:
        market_section = f"\n## Market Profile\n{market_profile}\n"

    # Tool tags section: only for opencode provider
    use_tool_tags = provider in ("opencode", "auto")
    if use_tool_tags:
        tooling_section = """
## Tooling (optional)
You MAY use tool-call tags (OpenCode-style):
- <read filePath="user_data/strategies/Foo.py"/>
- <write filePath="user_data/strategies/Foo.py">...python code...</write>
- <edit filePath="user_data/strategies/Foo.py" oldString="..." newString="..."/>
- <bash command="ls -la"/>
If you use `<write>...</write>`, the **content inside** must be pure Python (no tool tags).
If you do not use tools, reply with a single Python code block.
"""
    else:
        tooling_section = """
## Output format
Reply with a single Python code block containing the complete strategy file.
Do NOT use any XML tool tags (no <read>, <write>, <edit>, <bash> tags).
"""

    return f"""You are a quantitative trading strategy developer. Your goal is to create a FreqTrade strategy that maximizes risk-adjusted returns.

## Task
Create a complete FreqTrade IStrategy class in Python. The strategy file must be saved to:
  {sandbox_path}/user_data/strategies/

## Requirements
1. The class MUST inherit from `freqtrade.strategy.IStrategy`
2. MUST implement: `populate_indicators()`, `populate_entry_trend()`, `populate_exit_trend()`
3. Use Freqtrade v3 signal columns: set `enter_long` / `exit_long` (and optionally `enter_tag`/`exit_tag`). Do NOT use `buy`/`sell` columns
4. Use `pandas_ta` or manual indicator implementations. DO NOT import `talib` or `talib.abstract` — TA-Lib is NOT installed and will cause backtest failure. Use `import pandas_ta as ta` instead
5. No look-ahead / future leakage (MUST NOT use):
   - `shift(-1)`, `diff(-1)`, `pct_change(-1)`
   - `rolling(..., center=True)`
   - `bfill()` / `backfill()` / `fillna(method='bfill')`
   - `np.roll(..., -1)`
6. Set `timeframe = "1h"` (this repo ships offline OHLCV at 1h)
7. Define complete order config using **entry/exit** wording (not buy/sell):
   - `order_types` must include keys: `entry`, `exit`, `stoploss`, `stoploss_on_exchange`
   - `order_time_in_force` must include keys: `entry`, `exit`
   - Use `"market"` for entry and exit order types (avoid limit orders which may result in 0 fills)
8. Ensure the strategy is likely to produce >= 20 trades over the timerange (avoid overly-strict filters).
   - Use relaxed thresholds: e.g. RSI 35-65 instead of 20-80, ADX > 15 instead of > 25
   - Combine at most 3-4 entry conditions (more conditions = fewer trades)
   - Test your logic mentally: would it trigger on a typical trending day?
9. Set reasonable `minimal_roi`, `stoploss` parameters. Prefer wider ROI (e.g. 5-10%) and moderate stoploss (-5% to -10%)
10. Do NOT import os, subprocess, socket, requests, urllib, or use exec/eval/open
11. IMPORTANT: Strategies that produce 0 trades are worthless. Prioritize generating trades over perfect precision.
    If unsure, make entry conditions LESS strict rather than more strict.
{tooling_section}
## Reference
- A reference strategy is at: {sandbox_path}/user_data/strategies/ExpressionLongStrategy_reference.py
- FreqTrade config: {freqtrade_config}
- Timerange for backtesting: {timerange}

## Iteration {iteration}
{multi_header}{history_section}{best_section}{kb_section}{market_section}{multi_rules}
## Instructions
1. Read the reference strategy to understand the expected format
2. Design a novel strategy with clear entry/exit logic
3. Write the strategy file to the strategies directory
4. {'Improve upon the best strategy above — try different indicators, parameters, or logic changes' if best_strategy_code else 'Start with a simple but well-reasoned approach'}

Write the strategy file now. Name it descriptively (e.g., MomentumBreakoutStrategy.py).
"""


def build_analysis_prompt(
    *,
    strategy_code: str,
    backtest_summary: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    m = metrics or {}

    return f"""You are analyzing a FreqTrade trading strategy's backtest results.

## Strategy Code
```python
{strategy_code}
```

## Backtest Results
- Total Profit: {backtest_summary.get('profit_total_pct', 'N/A')}%
- Number of Trades: {backtest_summary.get('trades', 'N/A')}
- Win Rate: {backtest_summary.get('winrate', 'N/A')}
- Max Drawdown: {backtest_summary.get('max_drawdown_abs', 'N/A')}
- Average Profit per Trade: {backtest_summary.get('avg_profit_pct', 'N/A')}%
- Best Pair: {backtest_summary.get('best_pair', 'N/A')}
- Worst Pair: {backtest_summary.get('worst_pair', 'N/A')}

## Professional Metrics
- Sharpe Ratio: {m.get('sharpe', 'N/A')}
- Sortino Ratio: {m.get('sortino', 'N/A')}
- Calmar Ratio: {m.get('calmar', 'N/A')}
- Profit Factor: {m.get('profit_factor', 'N/A')}
- Expectancy: {m.get('expectancy', 'N/A')}
- SQN: {m.get('sqn', 'N/A')}
- CAGR: {m.get('cagr', 'N/A')}
- Max Drawdown %: {m.get('max_drawdown_pct', 'N/A')}

## Task
Analyze the strategy and provide a structured diagnosis.

## Output format (STRICT)
Return a single JSON object (no markdown fences) with these keys:
- strengths: list[string] — what worked well (1-3 items)
- weaknesses: list[string] — main weaknesses focusing on Sharpe, drawdown, profit factor (1-3 items)
- suggestions: list[string] — specific improvements for next iteration with concrete indicators/parameters (1-3 items)
- verdict: string — one of "promising", "mediocre", "poor"
- summary: string — one-sentence overall assessment (max 100 words)
"""


def build_repair_prompt(
    *,
    sandbox_path: str,
    strategy_rel_path: str,
    freqtrade_config: str,
    timerange: str,
    failure: str,
    attempt: int,
    max_attempts: int,
    tool_allowlist: Optional[List[str]] = None,
    bash_allow: bool = True,
    bash_timeout: int = 60,
    bash_allowlist: Optional[List[str]] = None,
    provider: str = "auto",
) -> str:
    tools_s = ", ".join(tool_allowlist or []) or "(default)"
    bash_list_s = "\n".join(f"  - {x}" for x in (bash_allowlist or [])[:20])
    if not bash_list_s:
        bash_list_s = "  - (none)"

    use_tool_tags = provider in ("opencode", "auto")
    if use_tool_tags:
        tool_policy_section = f"""## Tool policy
- Allowed tools: {tools_s}
- Bash enabled: {bash_allow} (timeout={bash_timeout}s)
- Bash allowlist (prefix match):
{bash_list_s}"""
        how_to_work_section = f"""## How to work
1. Start by reading the current file:
   <read filePath="{strategy_rel_path}"/>
2. Apply minimal edits to fix issues:
   <edit filePath="{strategy_rel_path}" oldString="..." newString="..."/>
   or rewrite the full file:
   <write filePath="{strategy_rel_path}">...python code...</write>
3. (Optional) Run quick checks:
   <bash command="python3 -m py_compile {strategy_rel_path}"/>"""
    else:
        tool_policy_section = ""
        how_to_work_section = """## Output format
Reply with a single Python code block containing the complete fixed strategy file.
Do NOT use any XML tool tags."""

    return f"""You are a senior Freqtrade strategy engineer.

## Goal
Repair the existing strategy to pass static validation and run backtesting successfully.

## Context
- Sandbox root: {sandbox_path}
- Strategy file to edit: {sandbox_path}/{strategy_rel_path}
- FreqTrade config: {freqtrade_config}
- Timerange: {timerange}
- Repair attempt: {attempt}/{max_attempts}

## Failure
{failure}

{tool_policy_section}

## Requirements (must keep)
1. The class MUST inherit from `freqtrade.strategy.IStrategy`
2. MUST implement: `populate_indicators()`, `populate_entry_trend()`, `populate_exit_trend()`
3. Use `enter_long` / `exit_long` columns (not `buy`/`sell`)
4. Set `timeframe = "1h"` (this repo ships offline OHLCV at 1h)
5. Ensure `order_types` / `order_time_in_force` use entry/exit keys (not buy/sell) and include required keys
6. Do NOT import os, subprocess, socket, requests, urllib, talib, or use exec/eval/open
7. Use `pandas_ta` (NOT `talib`) for indicators. If the current code uses `talib.abstract as ta`, replace with `import pandas_ta as ta` and update ALL API calls (e.g. `ta.EMA(dataframe, timeperiod=N)` → `ta.ema(dataframe['close'], length=N)`, `ta.BBANDS(...)` → `ta.bbands(...)` etc.)

{how_to_work_section}

Make the changes now.
"""


def build_planner_prompt(
    *,
    iteration: int,
    candidate_idx: int,
    candidates_per_iteration: int,
    freqtrade_config: str,
    timerange: str,
    history: List[Dict[str, Any]],
    elite_summaries: Optional[List[Dict[str, Any]]] = None,
    failure_summary: Optional[str] = None,
) -> str:
    """Planner role: propose a concise strategy design (no code)."""

    history_section = ""
    if history:
        lines = []
        for h in history[-5:]:
            lines.append(
                f"  - Iteration {h.get('iteration', '?')}: "
                f"sharpe={h.get('sharpe', 'N/A')}, "
                f"profit={h.get('profit_pct', 'N/A')}%, "
                f"trades={h.get('trades', 'N/A')}, "
                f"profit_factor={h.get('profit_factor', 'N/A')}"
            )
        history_section = "\n## Recent history (most recent 5)\n" + "\n".join(lines) + "\n"

    kb_section = ""
    if elite_summaries:
        elite_lines = []
        for e in elite_summaries[:3]:
            elite_lines.append(
                f"  - {e.get('name', '?')}: sharpe={e.get('reward', 0):.4f}, "
                f"profit={e.get('profit_pct', '?')}%, trades={e.get('trades', '?')}"
            )
        kb_section += "\n## Elite archive (top performers)\n" + "\n".join(elite_lines) + "\n"

    if failure_summary and failure_summary != "No recorded failures.":
        kb_section += "\n## Known failure patterns (avoid)\n" + failure_summary + "\n"

    return f"""You are the PLANNER agent for an automated Freqtrade strategy miner.

## Task
Propose a concrete trading-strategy design for a new candidate. Do NOT write code.

## Context
- Candidate: {int(candidate_idx) + 1}/{int(candidates_per_iteration)}
- Iteration: {int(iteration)}
- FreqTrade config: {freqtrade_config}
- Backtest timerange: {timerange}
{history_section}{kb_section}
## CRITICAL: Trade generation
- The strategy MUST produce >= 20 trades over the timerange. Strategies with 0 trades are worthless.
- Use at most 3-4 entry conditions. More conditions = fewer trades = likely failure.
- Use relaxed indicator thresholds (e.g. RSI 35-65, ADX > 15, not extreme values).
- Prefer simple, proven patterns: EMA crossovers, RSI mean-reversion, breakout above recent high.
- Timeframe MUST be "1h" (only 1h data is available).

## Output format (STRICT)
Return a single JSON object (no markdown fences) with keys:
- name_hint: string (CamelCase class name suggestion)
- timeframe: string (MUST be "1h")
- indicators: list[string]
- entry_rules: list[string] (keep to 3-4 rules max)
- exit_rules: list[string]
- risk_controls: list[string]
- parameters: object (key->value)

Keep it short and actionable.
"""


def build_reviewer_prompt(
    *,
    strategy_code: str,
    freqtrade_config: str,
    timerange: str,
) -> str:
    """Reviewer role: review code for compatibility + miner validation rules."""

    return f"""You are the REVIEWER agent for a Freqtrade strategy miner.

## Task
Review the strategy for correctness, miner safety rules, and Freqtrade backtest compatibility.

## Miner safety rules (must pass)
- MUST inherit from `freqtrade.strategy.IStrategy`
- MUST implement: populate_indicators(), populate_entry_trend(), populate_exit_trend()
- Do NOT import: os, subprocess, socket, requests, urllib, http, ftplib, smtplib, telnetlib, xmlrpc, shutil, pathlib, talib
- Use `pandas_ta` (NOT `talib`) for all indicators
- Do NOT call: exec, eval, open, __import__, compile, getattr, setattr, delattr, globals, locals
- Avoid look-ahead leakage (negative shift/diff/pct_change, centered rolling, backfill)

## Context
- FreqTrade config: {freqtrade_config}
- Backtest timerange: {timerange}

## Strategy code
```python
{strategy_code}
```

## Output format (STRICT)
Return a single JSON object (no markdown fences) with keys:
- approved: boolean
- issues: list[string]
- fixed_code: string|null (if you can provide a corrected full file)

If approved is false, fixed_code should include a complete corrected strategy when feasible.
"""


def build_backtester_prompt(
    *,
    strategy_code: str,
    freqtrade_config: str,
    timerange: str,
) -> str:
    """Backtester role: propose quick checks and likely failure modes."""

    return f"""You are the BACKTESTER agent for a Freqtrade strategy miner.

## Task
Given the strategy code, propose quick checks and likely failure modes BEFORE running a full backtest.

## Context
- FreqTrade config: {freqtrade_config}
- Backtest timerange: {timerange}

## Strategy code
```python
{strategy_code}
```

## Output format (STRICT)
Return a single JSON object (no markdown fences) with keys:
- quick_checks: list[string]   # commands or checks
- likely_failures: list[string]
- risk_notes: list[string]
"""
