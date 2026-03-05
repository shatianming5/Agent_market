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
    best_reward: float,
    best_strategy_code: Optional[str] = None,
    elite_summaries: Optional[List[Dict[str, Any]]] = None,
    failure_summary: Optional[str] = None,
) -> str:
    history_section = ""
    if history:
        lines = []
        for h in history[-5:]:
            lines.append(
                f"  - Iteration {h.get('iteration', '?')}: "
                f"reward={h.get('reward', 'N/A'):.4f}, "
                f"profit={h.get('profit_pct', 'N/A')}%, "
                f"trades={h.get('trades', 'N/A')}, "
                f"diagnosis: {h.get('diagnosis', 'N/A')}"
            )
        history_section = (
            "\n## Previous Iterations (most recent 5)\n"
            + "\n".join(lines)
            + "\n"
        )

    best_section = ""
    if best_strategy_code and best_reward > float("-inf"):
        best_section = (
            f"\n## Current Best Strategy (reward={best_reward:.4f})\n"
            f"```python\n{best_strategy_code}\n```\n"
        )

    kb_section = ""
    if elite_summaries:
        elite_lines = []
        for e in elite_summaries[:3]:
            elite_lines.append(
                f"  - {e.get('name', '?')}: reward={e.get('reward', 0):.4f}, "
                f"profit={e.get('profit_pct', '?')}%, "
                f"trades={e.get('trades', '?')}, winrate={e.get('winrate', '?')}"
            )
        kb_section += "\n## Elite Strategy Archive (top performers)\n" + "\n".join(elite_lines) + "\n"

    if failure_summary and failure_summary != "No recorded failures.":
        kb_section += (
            "\n## Known Failure Patterns (avoid these)\n"
            + failure_summary
            + "\n"
        )

    return f"""You are a quantitative trading strategy developer. Your goal is to create a FreqTrade strategy that maximizes risk-adjusted returns.

## Task
Create a complete FreqTrade IStrategy class in Python. The strategy file must be saved to:
  {sandbox_path}/user_data/strategies/

## Requirements
1. The class MUST inherit from `freqtrade.strategy.IStrategy`
2. MUST implement: `populate_indicators()`, `populate_entry_trend()`, `populate_exit_trend()`
3. Use standard TA indicators (talib, pandas_ta, or manual calculation)
4. Set reasonable `minimal_roi`, `stoploss`, `timeframe` parameters
5. Do NOT import os, subprocess, socket, requests, or use exec/eval/open

## Reference
- A reference strategy is at: {sandbox_path}/user_data/strategies/ExpressionLongStrategy_reference.py
- FreqTrade config: {freqtrade_config}
- Timerange for backtesting: {timerange}

## Iteration {iteration}
{history_section}{best_section}{kb_section}
## Instructions
1. Read the reference strategy to understand the expected format
2. Design a novel strategy with clear entry/exit logic
3. Write the strategy file to the strategies directory
4. {'Improve upon the best strategy above — try different indicators, parameters, or logic' if best_strategy_code else 'Start with a simple but well-reasoned approach'}

Write the strategy file now. Name it descriptively (e.g., MomentumBreakoutStrategy.py).
"""


def build_analysis_prompt(
    *,
    strategy_code: str,
    backtest_summary: Dict[str, Any],
    reward: float,
    reward_components: Dict[str, float],
) -> str:
    components_str = "\n".join(
        f"  - {k}: {v:.4f}" for k, v in reward_components.items()
    )

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

## Reward Score: {reward:.4f}
## Component Scores
{components_str}

## Task
Provide a concise diagnosis (max 200 words):
1. What worked well in this strategy?
2. What are the main weaknesses?
3. What specific improvements should be tried in the next iteration?
   - Be concrete: suggest specific indicators, parameter ranges, or logic changes
   - Focus on the lowest-scoring components above

Respond with ONLY the diagnosis text, no code.
"""
