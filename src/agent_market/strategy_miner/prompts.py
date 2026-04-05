"""Prompt templates for the strategy mining agent."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _fmt_pct(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text or "0"


def _timeframe_policy_text(
    *,
    base_timeframe: str,
    max_strategy_timeframe: Optional[str] = None,
    allowed_informative_timeframes: Optional[List[str]] = None,
) -> str:
    max_tf = (max_strategy_timeframe or base_timeframe or "").strip() or base_timeframe
    informative = [tf for tf in (allowed_informative_timeframes or []) if tf and tf != base_timeframe]
    line = f'Set `timeframe = "{base_timeframe}"` to match the Freqtrade config.'
    if informative:
        joined = ", ".join(f"`{tf}`" for tf in informative)
        line += f" You MAY use informative confirmation only on {joined}."
    if max_tf:
        line += f" Do NOT use any timeframe higher than `{max_tf}`."
    return line


def _pattern_guidance_text(
    *,
    preferred_patterns: Optional[List[str]] = None,
    avoid_patterns: Optional[List[str]] = None,
) -> str:
    blocks: list[str] = []
    preferred = [p for p in (preferred_patterns or []) if p]
    avoided = [p for p in (avoid_patterns or []) if p]
    if preferred:
        blocks.append("Preferred strategy families:\n" + "\n".join(f"   - {p}" for p in preferred))
    if avoided:
        blocks.append("Avoid low-frequency archetypes such as:\n" + "\n".join(f"   - {p}" for p in avoided))
    return "\n".join(blocks)


def _is_intraday_profile(
    *,
    base_timeframe: str,
    max_strategy_timeframe: Optional[str],
    target_trades: int,
) -> bool:
    tf = (base_timeframe or "").strip().lower()
    max_tf = (max_strategy_timeframe or base_timeframe or "").strip().lower()
    intraday_tf = tf.endswith("m") or max_tf.endswith("m")
    return intraday_tf or int(target_trades) >= 100


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
    market_context: Optional[str] = None,
    research_insights: Optional[str] = None,
    strategy_blueprints: Optional[str] = None,
    diversity_instructions: Optional[str] = None,
    previous_suggestions: Optional[List[str]] = None,
    base_timeframe: str = "1h",
    max_strategy_timeframe: Optional[str] = None,
    allowed_informative_timeframes: Optional[List[str]] = None,
    target_trades: int = 20,
    min_acceptable_trades: int = 10,
    roi_target_min_pct: float = 5.0,
    roi_target_max_pct: float = 10.0,
    stoploss_min_pct: float = 5.0,
    stoploss_max_pct: float = 10.0,
    preferred_patterns: Optional[List[str]] = None,
    avoid_patterns: Optional[List[str]] = None,
) -> str:
    # Type-aware history: only show rule-type candidates
    relevant_history = [
        h for h in history
        if str(h.get("candidate_type", "rule")).strip().lower() == "rule"
    ]
    history_section = ""
    if relevant_history:
        lines = []
        for h in relevant_history[-5:]:
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
        history_section = "\n## Previous Iterations (rule candidates, most recent 5)\n" + "\n".join(lines) + "\n"

    # Best strategy: scorecard only (no full code injection)
    best_section = ""
    if best_strategy_code and best_score > float("-inf"):
        best_section = (
            f"\n## Current Best Strategy (sharpe={best_score:.4f})\n"
            f"A rule-based strategy already scored {best_score:.4f}. "
            f"Do NOT copy its structure — explore a genuinely different approach.\n"
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

    # Slot-based diversity directive (expanded with DCA/grid/martingale archetypes)
    slot_idx = int(candidate_idx or 0)
    _slot_families = [
        "mean-reversion",
        "trend-pullback",
        "breakout",
        "dca-grid",           # DCA / Grid trading with adjust_trade_position
        "martingale-dca",     # Martingale / progressive DCA
        "basket-rotation",    # Cross-pair momentum rotation via informative_pairs
        "pair-relative",      # Relative strength / pair-based signals
    ]
    slot_directive = _slot_families[slot_idx % len(_slot_families)]

    # If diversity_instructions override the archetype, use that instead of slot_directive
    if diversity_instructions and diversity_instructions != slot_directive:
        slot_directive = diversity_instructions.split(".")[0].strip() if "." in diversity_instructions else diversity_instructions

    multi_header = ""
    multi_rules = ""
    if candidate_idx is not None and candidates_per_iteration:
        multi_header = f"\n## Candidate {int(candidate_idx) + 1}/{int(candidates_per_iteration)} — archetype: {slot_directive}\n"
        multi_rules = (
            "\n## Multi-candidate rules\n"
            "- This is one of several candidates generated in parallel for the same iteration.\n"
            "- Your assigned archetype is **{arch}**. Build the strategy around this family.\n"
            "- Make this strategy meaningfully different from other candidates (different indicators/logic).\n"
            "- Use a UNIQUE class name to avoid collisions. Recommended: suffix with `_cand{idx}`.\n"
        ).format(arch=slot_directive, idx=int(candidate_idx))

    market_section = ""
    if market_profile:
        market_section = f"\n## Market Profile\n{market_profile}\n"

    # External research sections (from opencli + paper research)
    research_section = ""
    if market_context:
        research_section += market_context
    if research_insights:
        research_section += "\n" + research_insights
    if strategy_blueprints:
        research_section += "\n" + strategy_blueprints

    # Diversity enforcement (P0-3)
    diversity_section = ""
    if diversity_instructions:
        diversity_section = f"\n## Diversity Requirement (MANDATORY)\n{diversity_instructions}\n"

    # Previous analysis suggestions feedback (P1-1) — SCOPE filtered
    suggestions_section = ""
    if previous_suggestions:
        rule_suggestions = [
            s for s in previous_suggestions[:5]
            if "SCOPE=rule" in s or "SCOPE=" not in s
        ]
        if rule_suggestions:
            sugg_lines = "\n".join(f"  - {s}" for s in rule_suggestions)
            suggestions_section = (
                "\n## Improvement Suggestions from Previous Iteration\n"
                "Apply these specific improvements (only SCOPE=rule suggestions apply here):\n"
                + sugg_lines + "\n"
            )

    # Tooling: simplified — always request a single code block
    tooling_section = """
## Output format
Reply with exactly one Python code block containing the complete strategy file.
"""

    trade_soft_cap = max(int(target_trades) * 2, 1200)

    timeframe_rule = _timeframe_policy_text(
        base_timeframe=base_timeframe,
        max_strategy_timeframe=max_strategy_timeframe,
        allowed_informative_timeframes=allowed_informative_timeframes,
    )
    pattern_guidance = _pattern_guidance_text(
        preferred_patterns=preferred_patterns,
        avoid_patterns=avoid_patterns,
    )
    intraday_profile = _is_intraday_profile(
        base_timeframe=base_timeframe,
        max_strategy_timeframe=max_strategy_timeframe,
        target_trades=target_trades,
    )

    trade_objective = (
        "Optimize for intraday frequency and turnover."
        if intraday_profile
        else "Ensure the strategy trades often enough to generate meaningful statistics."
    )
    trade_realism = (
        "- For the current market/pair universe, a realistic sweet spot is about 200-600 trades over the month. Do NOT optimize for rare home-run trades."
        if intraday_profile
        else "- Prefer simple, repeatable setups over rare signals."
    )
    risk_objective = (
        "Set intraday `minimal_roi`, `stoploss`, and time exits for fast recycling:"
        if intraday_profile
        else "Set reasonable `minimal_roi`, `stoploss`, and exit timing:"
    )
    importance_objective = (
        "IMPORTANT: This search values frequent, repeatable intraday edge with decent win rate."
        if intraday_profile
        else "IMPORTANT: This search values repeatable signals and robust trade count."
    )
    avoidance_objective = (
        "- Avoid multi-day swing logic, over-filtered EMA breakout designs, and trades that depend on rare large moves"
        if intraday_profile
        else "- Avoid over-filtered designs that starve trade count"
    )

    return f"""You are a quantitative trading strategy developer. Your goal is to create a FreqTrade strategy that maximizes risk-adjusted returns.

## Task
Create a complete FreqTrade IStrategy class in Python. The strategy file must be saved to:
  {sandbox_path}/user_data/strategies/

## Requirements
1. The class MUST inherit from `freqtrade.strategy.IStrategy`
2. MUST implement: `populate_indicators()`, `populate_entry_trend()`, `populate_exit_trend()`
3. Use Freqtrade v3 signal columns: set `enter_long` / `exit_long` (and optionally `enter_tag`/`exit_tag`). Do NOT use `buy`/`sell` columns
4. Use `pandas_ta` or manual indicator implementations. DO NOT import `talib` or `talib.abstract` — TA-Lib is NOT installed and will cause backtest failure. Use `import pandas_ta as ta` instead
4b. **IMPORTANT: Use HyperOptable parameters instead of hardcoded values!**
   Import: `from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter`
   Example:
   ```python
   rsi_period = IntParameter(10, 30, default=14, space="buy")
   rsi_threshold = IntParameter(25, 45, default=35, space="buy")
   exit_rsi = IntParameter(60, 80, default=70, space="sell")
   ```
   Then in `populate_indicators()`, precompute for all possible values:
   ```python
   for period in self.rsi_period.range:
       dataframe[f"rsi_{{period}}"] = ta.rsi(dataframe["close"], length=period)
   ```
   And in `populate_entry_trend()`, reference `.value`:
   ```python
   dataframe.loc[dataframe[f"rsi_{{self.rsi_period.value}}"] < self.rsi_threshold.value, "enter_long"] = 1
   ```
   This allows automatic parameter optimization via Hyperopt. Do NOT hardcode threshold/period values.
5. No look-ahead / future leakage (MUST NOT use):
   - `shift(-1)`, `diff(-1)`, `pct_change(-1)`
   - `rolling(..., center=True)`
   - `bfill()` / `backfill()` / `fillna(method='bfill')`
   - `np.roll(..., -1)`
6. {timeframe_rule}
7. Define complete order config using **entry/exit** wording (not buy/sell):
   - `order_types` must include keys: `entry`, `exit`, `stoploss`, `stoploss_on_exchange`
   - `order_time_in_force` must include keys: `entry`, `exit`
   - Use `"market"` for entry and exit order types (avoid limit orders which may result in 0 fills)
8. {trade_objective}
   - Target >= {int(target_trades)} trades over the timerange; fewer than {int(min_acceptable_trades)} trades is a failure
   {trade_realism}
   - Use relaxed thresholds: e.g. RSI 40-60 instead of extreme 20/80 values, mild pullback filters instead of deep oversold requirements
   - Combine at most 3-4 entry conditions (more conditions = fewer trades)
   - Prefer repeatable structures that can recycle multiple times over the timerange
9. {risk_objective}
   - Prefer total take-profit targets around { _fmt_pct(roi_target_min_pct) }% to { _fmt_pct(roi_target_max_pct) }%
   - Keep `stoploss` roughly between -{ _fmt_pct(stoploss_min_pct) }% and -{ _fmt_pct(stoploss_max_pct) }%
   - Add a time-based exit / max-hold rule so positions do not linger for days
10. Do NOT import os, subprocess, socket, requests, urllib, or use exec/eval/open
11. {importance_objective}
    - If unsure, make entry conditions LESS strict rather than more strict
    {avoidance_objective}
{pattern_guidance if pattern_guidance else ""}
{tooling_section}
## Objective Ranking
The search ranks candidates by a composite score that rewards:
  1. Positive profit after realistic fees
  2. Profit factor > 1.0
  3. Trade count inside [{int(min_acceptable_trades)}, {trade_soft_cap}]
  4. Per-pair robustness (median pair PF > 1.0 and worst-pair PF > 0.6)
  5. Drawdown-adjusted Sharpe (not annualized)

## Hard Gates
- trades < {int(min_acceptable_trades)} => immediate discard
- trades > {trade_soft_cap} => penalized (over-trading)
- profit_factor < 0.85 => immediate discard

## Pair Robustness
The strategy will be evaluated on ALL pairs in the config. A strategy that works on only one pair but loses on others will score poorly. Design entry logic that is pair-agnostic (e.g. based on relative price action, not hard-coded levels).

## Advanced Strategy Types
### DCA / Grid / Martingale
If your assigned archetype is `dca-grid` or `martingale-dca`, you MUST implement `adjust_trade_position()`:
```python
position_adjustment_enable = True
max_entry_position_adjustment = 3  # up to 3 additional entries

def adjust_trade_position(self, trade, current_time, current_rate: float,
                          current_profit: float, min_stake: float | None,
                          max_stake: float, current_entry_rate: float,
                          current_exit_rate: float, current_entry_profit: float,
                          current_exit_profit: float, **kwargs) -> float | None:
    # DCA: add to position when price drops by a threshold
    if current_profit > -0.02:  # only add when underwater
        return None
    if trade.nr_of_successful_entries >= self.max_entry_position_adjustment:
        return None  # max entries reached
    return trade.stake_amount  # add same size
```
- For **grid**: enter at fixed price intervals (e.g. every -1% drop), exit at target profit
- For **martingale**: increase stake on each DCA level (e.g. 1x, 1.5x, 2x)
- For **DCA**: equal-size additional entries at progressively lower prices
- Trade count may be lower (100-300) for DCA strategies — that is acceptable
- Keep `stoploss` wider for DCA strategies (e.g. -5% to -10%) since the strategy averages down

### Basket Trading / Cross-Pair Strategies
For cross-pair strategies (basket, momentum rotation, pair trading), use `informative_pairs()` to access other pairs' data:
```python
def informative_pairs(self):
    # Access BTC and ETH data for cross-pair signals
    return [("BTC/USDT", "5m"), ("ETH/USDT", "5m")]

def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    # Get BTC data for relative strength
    if self.dp:
        btc = self.dp.get_pair_dataframe("BTC/USDT", "5m")
        dataframe["btc_close"] = btc["close"]
        dataframe["btc_rsi"] = ta.rsi(btc["close"], length=14)
        # Relative strength: this pair vs BTC
        dataframe["relative_strength"] = dataframe["close"].pct_change(12) - btc["close"].pct_change(12)
    return dataframe
```
- **Basket rotation**: buy the strongest pair (highest relative_strength), sell the weakest
- **Pair trading**: long when spread is oversold, short when overbought (use `can_short=True` in futures mode)
- **Cross-pair momentum**: enter when this pair outperforms BTC/ETH

## Reference
- A reference strategy is at: {sandbox_path}/user_data/strategies/ExpressionLongStrategy_reference.py
- FreqTrade config: {freqtrade_config}
- Timerange for backtesting: {timerange}

## Iteration {iteration}
{multi_header}{history_section}{best_section}{kb_section}{market_section}{research_section}{diversity_section}{suggestions_section}{multi_rules}
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
    analysis_research_context: Optional[str] = None,
) -> str:
    m = metrics or {}

    # Truncate strategy code to reduce token usage
    truncated_code = strategy_code[:2600]
    if len(strategy_code) > 2600:
        truncated_code += "\n# ... (truncated for analysis) ..."

    research_block = ""
    if analysis_research_context:
        research_block = f"\n{analysis_research_context}\n"

    # Pair robustness section
    pair_robustness_block = ""
    results_per_pair = backtest_summary.get("results_per_pair") or backtest_summary.get("pair_results") or []
    if results_per_pair:
        pair_pfs = []
        # Handle both list and dict formats
        items = results_per_pair.items() if isinstance(results_per_pair, dict) else [(d.get("key", d.get("pair", "?")), d) for d in results_per_pair if isinstance(d, dict)]
        for pair_name, pair_data in items:
            if str(pair_name) == "TOTAL" or not isinstance(pair_data, dict):
                continue
            pf = pair_data.get("profit_factor", None)
            if pf is not None:
                try:
                    pair_pfs.append(float(pf))
                except (ValueError, TypeError):
                    pass
        if pair_pfs:
            pair_pfs_sorted = sorted(pair_pfs)
            n = len(pair_pfs_sorted)
            median_pf = pair_pfs_sorted[n // 2] if n % 2 == 1 else (pair_pfs_sorted[n // 2 - 1] + pair_pfs_sorted[n // 2]) / 2.0
            worst_pf = pair_pfs_sorted[0]
            pair_robustness_block = (
                f"\n## Pair Robustness\n"
                f"- Pairs evaluated: {n}\n"
                f"- Median pair PF: {median_pf:.3f}\n"
                f"- Worst pair PF: {worst_pf:.3f}\n"
            )

    return f"""You are analyzing a FreqTrade trading strategy's backtest results.

## Strategy Code (may be truncated)
```python
{truncated_code}
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
{pair_robustness_block}
## Objective Ranking
The search ranks candidates by a composite score that rewards:
  1. Positive profit after realistic fees
  2. Profit factor > 1.0
  3. Trade count inside target band (not too few, not too many)
  4. Per-pair robustness (median pair PF > 1.0, worst pair PF > 0.6)
  5. Drawdown-adjusted Sharpe (not annualized)

## Task
Analyze the strategy and provide a structured diagnosis.
{research_block}

## Output format (STRICT)
Return a single JSON object (no markdown fences) with these keys:
- strengths: list[string] — what worked well (1-3 items)
- weaknesses: list[string] — main weaknesses focusing on Sharpe, drawdown, profit factor, pair robustness (1-3 items)
- suggestions: list[string] — each suggestion MUST follow this tagged format:
    "SCOPE=<type>; AXIS=<axis>; CHANGE=<change>; WHY=<why>; PRIORITY=<p>"
    where <type> is one of rule|ml|dl|rl, <axis> is what to change, <change> is the concrete change, <why> is the reason, <p> is 1-3 (1=highest).
    If candidate is ml/dl/rl, at least 1 suggestion must change search space (features, model family, reward), not just thresholds.
    Provide 1-3 suggestions.
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
    base_timeframe: str = "1h",
    max_strategy_timeframe: Optional[str] = None,
    allowed_informative_timeframes: Optional[List[str]] = None,
    target_trades: int = 20,
    min_acceptable_trades: int = 10,
    roi_target_min_pct: float = 5.0,
    roi_target_max_pct: float = 10.0,
    stoploss_min_pct: float = 5.0,
    stoploss_max_pct: float = 10.0,
    preferred_patterns: Optional[List[str]] = None,
    avoid_patterns: Optional[List[str]] = None,
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

    timeframe_rule = _timeframe_policy_text(
        base_timeframe=base_timeframe,
        max_strategy_timeframe=max_strategy_timeframe,
        allowed_informative_timeframes=allowed_informative_timeframes,
    )
    pattern_guidance = _pattern_guidance_text(
        preferred_patterns=preferred_patterns,
        avoid_patterns=avoid_patterns,
    )
    intraday_profile = _is_intraday_profile(
        base_timeframe=base_timeframe,
        max_strategy_timeframe=max_strategy_timeframe,
        target_trades=target_trades,
    )
    objective_label = "intraday objective" if intraday_profile else "search objective"

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
4. {timeframe_rule}
5. Ensure `order_types` / `order_time_in_force` use entry/exit keys (not buy/sell) and include required keys
6. Do NOT import os, subprocess, socket, requests, urllib, talib, or use exec/eval/open
7. Use `pandas_ta` (NOT `talib`) for indicators. If the current code uses `talib.abstract as ta`, replace with `import pandas_ta as ta` and update ALL API calls (e.g. `ta.EMA(dataframe, timeperiod=N)` → `ta.ema(dataframe['close'], length=N)`, `ta.BBANDS(...)` → `ta.bbands(...)` etc.)
8. Keep the strategy aligned with the {objective_label}:
   - Target >= {int(target_trades)} trades over the timerange; fewer than {int(min_acceptable_trades)} trades is a failure
   - Prefer total take-profit targets around { _fmt_pct(roi_target_min_pct) }% to { _fmt_pct(roi_target_max_pct) }%
   - Keep `stoploss` roughly between -{ _fmt_pct(stoploss_min_pct) }% and -{ _fmt_pct(stoploss_max_pct) }%
   - Maintain or add time-based exits / max-hold logic for quick recycling
{pattern_guidance if pattern_guidance else ""}

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
    base_timeframe: str = "1h",
    max_strategy_timeframe: Optional[str] = None,
    allowed_informative_timeframes: Optional[List[str]] = None,
    target_trades: int = 20,
    min_acceptable_trades: int = 10,
    roi_target_min_pct: float = 5.0,
    roi_target_max_pct: float = 10.0,
    stoploss_min_pct: float = 5.0,
    stoploss_max_pct: float = 10.0,
    preferred_patterns: Optional[List[str]] = None,
    avoid_patterns: Optional[List[str]] = None,
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

    timeframe_rule = _timeframe_policy_text(
        base_timeframe=base_timeframe,
        max_strategy_timeframe=max_strategy_timeframe,
        allowed_informative_timeframes=allowed_informative_timeframes,
    )
    pattern_guidance = _pattern_guidance_text(
        preferred_patterns=preferred_patterns,
        avoid_patterns=avoid_patterns,
    )
    intraday_profile = _is_intraday_profile(
        base_timeframe=base_timeframe,
        max_strategy_timeframe=max_strategy_timeframe,
        target_trades=target_trades,
    )
    turnover_line = (
        "- Prefer repeatable intraday structures that can recycle across the month."
        if intraday_profile
        else "- Prefer repeatable structures over rare high-conviction setups."
    )

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
- The strategy should aim for >= {int(target_trades)} trades over the timerange. Below {int(min_acceptable_trades)} trades is a failure.
- Use at most 3-4 entry conditions. More conditions = fewer trades = likely failure.
- {turnover_line[2:]}
- Use relaxed indicator thresholds (e.g. RSI 40-60, mild pullback filters, shallow band/VWAP reversion).
- {timeframe_rule}
{pattern_guidance if pattern_guidance else ""}

## Output format (STRICT)
Return a single JSON object (no markdown fences) with keys:
- name_hint: string (CamelCase class name suggestion)
- timeframe: string (MUST be "{base_timeframe}")
- indicators: list[string]
- entry_rules: list[string] (keep to 3-4 rules max)
- exit_rules: list[string]
- risk_controls: list[string]
- parameters: object (key->value)

    Keep it short and actionable.
    """


def build_model_planner_prompt(
    *,
    iteration: int,
    candidate_idx: int,
    candidates_per_iteration: int,
    candidate_type: str,
    allowed_model_families: List[str],
    freqtrade_config: str,
    timerange: str,
    feature_file: str,
    expressions_file: Optional[str],
    pairs: List[str],
    history: List[Dict[str, Any]],
    target_trades: int = 20,
    min_acceptable_trades: int = 10,
) -> str:
    # Type-aware history: only show same-type candidates
    ct_lower = candidate_type.strip().lower()
    relevant_history = [
        h for h in history
        if str(h.get("candidate_type", "rule")).strip().lower() == ct_lower
    ]
    history_section = ""
    if relevant_history:
        lines = []
        for h in relevant_history[-5:]:
            lines.append(
                f"  - Iteration {h.get('iteration', '?')}: "
                f"type={h.get('candidate_type', 'rule')}, "
                f"sharpe={h.get('sharpe', 'N/A')}, "
                f"profit={h.get('profit_pct', 'N/A')}%, "
                f"trades={h.get('trades', 'N/A')}"
            )
        history_section = "\n## Recent history ({} candidates only)\n".format(ct_lower) + "\n".join(lines) + "\n"

    # Forced family / label scheduling for ML
    schedule_idx = int(candidate_idx)
    forced_family = ""
    forced_label = ""
    if ct_lower == "ml":
        ml_families = [f for f in allowed_model_families if f in ("lightgbm", "xgboost")]
        if ml_families:
            forced_family = ml_families[schedule_idx % len(ml_families)]
        label_periods = [3, 6, 12, 24]
        forced_label = str(label_periods[schedule_idx % len(label_periods)])

    # Trade cap by type
    if ct_lower in ("dl", "rl"):
        trade_cap = 1500
    else:
        trade_cap = 2000

    # RL-specific guidance
    rl_note = ""
    if ct_lower == "rl":
        rl_note = (
            "\n- IMPORTANT for RL: The action policy is the primary signal generator. "
            "Probability thresholds (`enter_prob_threshold`, `exit_prob_threshold`) are only confidence gates, "
            "not the main driver. Focus on reward shaping and episode design.\n"
        )

    expr_line = expressions_file or "null"
    model_families = ", ".join(f"`{item}`" for item in allowed_model_families)
    forced_family_line = f"\n- **Forced model family for this slot**: `{forced_family}`" if forced_family else ""
    forced_label_line = f"\n- **Forced label period for this slot**: {forced_label} candles" if forced_label else ""

    return f"""You are the PLANNER agent for a model-mining strategy search.

## Task
Design a concise experiment plan for one `{candidate_type}` candidate. Do NOT write code.

## Context
- Candidate: {int(candidate_idx) + 1}/{int(candidates_per_iteration)}
- Iteration: {int(iteration)}
- Freqtrade config: {freqtrade_config}
- Timerange: {timerange}
- Candidate type: {candidate_type}
- Allowed model families: {model_families}{forced_family_line}{forced_label_line}
- Feature file: {feature_file}
- Expressions file: {expr_line}
- Pairs: {", ".join(pairs)}
- Trade objective: target >= {int(target_trades)} trades, below {int(min_acceptable_trades)} is a failure
- Trade cap (soft upper bound): {trade_cap}
{history_section}{rl_note}
## Output
Return 4-6 short bullet points covering:
- chosen model family (must respect forced family if given)
- target horizon / label idea (must respect forced label period if given)
- signal threshold idea
- risk controls
- why this should produce frequent trades instead of sparse trades
"""


def build_model_candidate_prompt(
    *,
    iteration: int,
    candidate_idx: int,
    candidates_per_iteration: int,
    candidate_type: str,
    allowed_model_families: List[str],
    freqtrade_config: str,
    timerange: str,
    feature_file: str,
    expressions_file: Optional[str],
    pairs: List[str],
    history: List[Dict[str, Any]],
    planner_notes: Optional[str] = None,
    previous_suggestions: Optional[List[str]] = None,
    base_timeframe: str = "1h",
    max_strategy_timeframe: Optional[str] = None,
    target_trades: int = 20,
    min_acceptable_trades: int = 10,
    roi_target_min_pct: float = 5.0,
    roi_target_max_pct: float = 10.0,
    stoploss_min_pct: float = 5.0,
    stoploss_max_pct: float = 10.0,
    training_validation_ratio: float = 0.2,
    training_rolling_splits: int = 3,
    training_scaler: str = "robust",
    rl_total_timesteps: int = 5000,
) -> str:
    ct_lower = candidate_type.strip().lower()
    schedule_idx = int(candidate_idx)

    # Type-aware history
    relevant_history = [
        h for h in history
        if str(h.get("candidate_type", "rule")).strip().lower() == ct_lower
    ]
    history_section = ""
    if relevant_history:
        lines = []
        for h in relevant_history[-5:]:
            lines.append(
                f"  - Iteration {h.get('iteration', '?')}: "
                f"type={h.get('candidate_type', 'rule')}, "
                f"sharpe={h.get('sharpe', 'N/A')}, "
                f"profit={h.get('profit_pct', 'N/A')}%, "
                f"trades={h.get('trades', 'N/A')}, "
                f"profit_factor={h.get('profit_factor', 'N/A')}"
            )
        history_section = "\n## Recent history ({} candidates only)\n".format(ct_lower) + "\n".join(lines) + "\n"

    suggestions_section = ""
    if previous_suggestions:
        type_suggestions = [
            s for s in previous_suggestions[:5]
            if f"SCOPE={ct_lower}" in s or "SCOPE=" not in s
        ]
        if type_suggestions:
            suggestions_section = (
                "\n## Previous suggestions (SCOPE={} only)\n".format(ct_lower)
                + "\n".join(f"- {item}" for item in type_suggestions)
                + "\n"
            )

    planner_section = ""
    if planner_notes:
        planner_section = f"\n## Planner notes\n{planner_notes}\n"

    expr_line = expressions_file or "null"
    families = ", ".join(f"`{item}`" for item in allowed_model_families)
    timeframe_rule = _timeframe_policy_text(
        base_timeframe=base_timeframe,
        max_strategy_timeframe=max_strategy_timeframe,
        allowed_informative_timeframes=None,
    )

    # Forced scheduling per type
    forced_family = ""
    forced_label = ""
    forced_scaler = training_scaler
    forced_timesteps = rl_total_timesteps

    if ct_lower == "ml":
        ml_families = [f for f in allowed_model_families if f in ("lightgbm", "xgboost")]
        if ml_families:
            forced_family = ml_families[schedule_idx % len(ml_families)]
        label_periods = [3, 6, 12, 24]
        forced_label = str(label_periods[schedule_idx % len(label_periods)])
        scaler_options = ["robust", "standard", "minmax"]
        forced_scaler = scaler_options[schedule_idx % len(scaler_options)]
    elif ct_lower == "dl":
        forced_family = "pytorch_mlp"
        label_periods = [6, 12, 24]
        forced_label = str(label_periods[schedule_idx % len(label_periods)])
    elif ct_lower == "rl":
        forced_family = "ppo"
        forced_timesteps = max(15000, rl_total_timesteps)

    # Model-specific guidance sections
    model_specific = ""
    if ct_lower == "ml":
        model_specific = f"""
## ML-Specific Guidance
- Forced model family: `{forced_family}`
- Forced label period: {forced_label} candles
- Forced scaler: `{forced_scaler}`
- The training label is continuous `future_return` — use regression objectives only.
- `signal.enter_threshold` / `signal.exit_threshold` are return thresholds on the model prediction.
- `enter_prob_threshold` / `exit_prob_threshold` are NOT used for ML — set them to null or omit.
"""
    elif ct_lower == "dl":
        model_specific = f"""
## DL-Specific Guidance
- Model family: `{forced_family}`
- Forced label period: {forced_label} candles
- The training label is continuous `future_return` — use regression loss (MSE/Huber).
- Keep the network small (1-3 hidden layers, 32-128 units). Avoid over-parameterization.
- `signal.enter_threshold` / `signal.exit_threshold` are return thresholds on the prediction.
"""
    elif ct_lower == "rl":
        model_specific = f"""
## RL-Specific Guidance
- Model family: `{forced_family}`
- `total_timesteps` MUST be >= {forced_timesteps} (at least 15000 to allow policy convergence).
- The action policy IS the primary signal generator. Probability thresholds (`enter_prob_threshold`,
  `exit_prob_threshold`) are only confidence gates — they are NOT the main driver.
- Focus on reward shaping: `fee_bps`, `holding_penalty_bps`, `drawdown_penalty`.
- Set `enter_prob_threshold` and `exit_prob_threshold` as soft gates (e.g. 0.55 / 0.35).
"""

    # Required defaults section
    defaults_section = ""
    if forced_family or forced_label:
        defaults_lines = []
        if forced_family:
            defaults_lines.append(f'- `model_family`: `{forced_family}`')
        if forced_label:
            defaults_lines.append(f'- `label_period`: {forced_label}')
        if ct_lower == "ml":
            defaults_lines.append(f'- `training.scaler`: `{forced_scaler}`')
        if ct_lower == "rl":
            defaults_lines.append(f'- `training.total_timesteps`: {forced_timesteps}')
        defaults_section = "\n## Required Defaults For This Slot\n" + "\n".join(defaults_lines) + "\n"

    return f"""You are designing one `{candidate_type}` model candidate for an automated trading-model miner.

## Hard constraints
- Return ONLY one JSON object. No markdown fences. No prose before or after the JSON.
- `candidate_type` MUST be `{candidate_type}`.
- `model_family` MUST be one of: {families}.
- Base timeframe is `{base_timeframe}` and the strategy may not rely on any timeframe above `{max_strategy_timeframe or base_timeframe}`.
- {timeframe_rule}
- Target >= {int(target_trades)} trades over the timerange. Below {int(min_acceptable_trades)} trades is a failure.
- Prefer frequent, repeatable intraday signals over sparse high-conviction setups.
- Use realistic intraday risk: total ROI target about {_fmt_pct(roi_target_min_pct)}%-{_fmt_pct(roi_target_max_pct)}%, stoploss about -{_fmt_pct(stoploss_min_pct)}% to -{_fmt_pct(stoploss_max_pct)}%, and a time-based exit.
- Evaluation favors non-annualized daily edge, positive profit, positive-day consistency, profit factor, and controlled drawdown. Do not optimize for inflated native Sharpe/Calmar.
- A good candidate should still look sane after fees across all pairs, not just on one low-drawdown corner case.

## Objective Ranking
The search ranks candidates by a composite score that rewards:
  1. Positive profit after realistic fees
  2. Profit factor > 1.0
  3. Trade count inside target band
  4. Per-pair robustness (median pair PF > 1.0, worst pair PF > 0.6)
  5. Drawdown-adjusted Sharpe (not annualized)
{model_specific}{defaults_section}
## Context
- Candidate: {int(candidate_idx) + 1}/{int(candidates_per_iteration)}
- Iteration: {int(iteration)}
- Freqtrade config: {freqtrade_config}
- Timerange: {timerange}
- Feature file: {feature_file}
- Expressions file: {expr_line}
- Pairs: {", ".join(pairs)}
- Default validation_ratio: {training_validation_ratio}
- Default rolling_splits: {training_rolling_splits}
- Default scaler: {forced_scaler}
- Default RL total_timesteps: {forced_timesteps}
{history_section}{planner_section}{suggestions_section}
## Output schema
{{
  "name_hint": "CamelCaseName",
  "candidate_type": "{candidate_type}",
  "model_family": "{forced_family or allowed_model_families[0]}",
  "rationale": "short explanation",
  "feature_file": "{feature_file}",
  "expressions_file": {('"'+expressions_file+'"') if expressions_file else 'null'},
  "label_period": {forced_label or '6'},
  "training": {{
    "validation_ratio": {training_validation_ratio},
    "rolling_splits": {training_rolling_splits},
    "scaler": "{forced_scaler}",
    "purge": 6,
    "embargo": 2,
    "total_timesteps": {forced_timesteps}
  }},
  "model_params": {{}},
  "signal": {{
    "enter_threshold": 0.0025,
    "exit_threshold": -0.0005,
    "enter_prob_threshold": 0.55,
    "exit_prob_threshold": 0.35
  }},
  "risk": {{
    "minimal_roi": {{"0": 0.006, "30": 0.003, "90": 0.0}},
    "stoploss": -0.012,
    "max_hold_minutes": 180,
    "startup_candle_count": 120
  }},
  "reward": {{
    "fee_bps": 10.0,
    "holding_penalty_bps": 0.2,
    "drawdown_penalty": 0.05
  }}
}}

Notes:
- For `ml`, choose `lightgbm` or `xgboost` (respect forced family).
- For `dl`, choose `pytorch_mlp`.
- For `rl`, choose `ppo`, set meaningful `reward` coefficients, and use `total_timesteps` >= {forced_timesteps}.
- For `ml` and `dl`, the training label is continuous `future_return`, so use regression-style model parameters. Do not use binary/classification objectives, class weights, or probability-calibration assumptions.
- For `ml` and `dl`, `signal.enter_threshold` / `signal.exit_threshold` should be return thresholds on the model prediction. `enter_prob_threshold` / `exit_prob_threshold` are only relevant for `rl`.
- Favor conservative, fee-aware thresholds and regime filters over hyperactive always-in-market behavior.
- Avoid designs that rely on microscopic drawdown denominators to produce unrealistic Sharpe or Calmar.
- Keep `model_params` practical. Avoid giant models.
"""


def build_reviewer_prompt(
    *,
    strategy_code: str,
    freqtrade_config: str,
    timerange: str,
    base_timeframe: str = "1h",
    max_strategy_timeframe: Optional[str] = None,
    allowed_informative_timeframes: Optional[List[str]] = None,
    target_trades: int = 20,
    min_acceptable_trades: int = 10,
    roi_target_min_pct: float = 5.0,
    roi_target_max_pct: float = 10.0,
    stoploss_min_pct: float = 5.0,
    stoploss_max_pct: float = 10.0,
    preferred_patterns: Optional[List[str]] = None,
    avoid_patterns: Optional[List[str]] = None,
) -> str:
    """Reviewer role: review code for compatibility + miner validation rules."""

    # Truncate code to reduce token usage
    truncated_code = strategy_code[:3200]
    if len(strategy_code) > 3200:
        truncated_code += "\n# ... (truncated for review) ..."

    trade_soft_cap = max(int(target_trades) * 2, 1200)

    timeframe_rule = _timeframe_policy_text(
        base_timeframe=base_timeframe,
        max_strategy_timeframe=max_strategy_timeframe,
        allowed_informative_timeframes=allowed_informative_timeframes,
    )
    pattern_guidance = _pattern_guidance_text(
        preferred_patterns=preferred_patterns,
        avoid_patterns=avoid_patterns,
    )
    intraday_profile = _is_intraday_profile(
        base_timeframe=base_timeframe,
        max_strategy_timeframe=max_strategy_timeframe,
        target_trades=target_trades,
    )
    objective_line = (
        "Keep the design intraday and liquid enough"
        if intraday_profile
        else "Keep the design active enough"
    )

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

## Search objective (must preserve)
- {timeframe_rule}
- {objective_line} to target >= {int(target_trades)} trades over the timerange. Below {int(min_acceptable_trades)} trades is a failure.
- Trade soft cap (upper bound): {trade_soft_cap} — designs likely to exceed this will be penalized for over-trading.
- Prefer total take-profit targets around { _fmt_pct(roi_target_min_pct) }% to { _fmt_pct(roi_target_max_pct) }% and stoploss around -{ _fmt_pct(stoploss_min_pct) }% to -{ _fmt_pct(stoploss_max_pct) }%.
- Preserve or add time-based exits / max-hold rules for quick recycling.
{pattern_guidance if pattern_guidance else ""}

## Context
- FreqTrade config: {freqtrade_config}
- Backtest timerange: {timerange}

## Strategy code (may be truncated)
```python
{truncated_code}
```

## Output format (STRICT)
Return a single JSON object (no markdown fences) with keys:
- approved: boolean
- issues: list[string] — each issue MUST be tagged with one of: [hard_gate], [trade_band], [pair_robustness], [compat]
- fixed_code: string|null (if you can provide a corrected full file)

Tag meanings:
- [hard_gate]: import violation, look-ahead, missing method, wrong column names
- [trade_band]: trade count likely outside [{int(min_acceptable_trades)}, {trade_soft_cap}]
- [pair_robustness]: logic is pair-specific or uses hard-coded price levels
- [compat]: Freqtrade API / order config issues

If approved is false, fixed_code should include a complete corrected strategy when feasible.
"""


def build_backtester_prompt(
    *,
    strategy_code: str,
    freqtrade_config: str,
    timerange: str,
) -> str:
    """Backtester role: propose quick checks and likely failure modes."""

    # Truncate code to reduce token usage
    truncated_code = strategy_code[:3200]
    if len(strategy_code) > 3200:
        truncated_code += "\n# ... (truncated for pre-check) ..."

    return f"""You are the BACKTESTER agent for a Freqtrade strategy miner.

## Task
Given the strategy code, propose quick checks and likely failure modes BEFORE running a full backtest.

## Context
- FreqTrade config: {freqtrade_config}
- Backtest timerange: {timerange}

## Strategy code (may be truncated)
```python
{truncated_code}
```

## Required Coverage
When checking the strategy, ensure you cover:
- **trade-count band**: Will the strategy produce enough trades (not too few, not too many)?
- **fee-drag**: Are the take-profit targets large enough to survive realistic fees (e.g. 0.1% per side)?
- **pair-robustness**: Does the logic work across multiple pairs or is it hard-coded to specific price levels?

## Output format (STRICT)
Return a single JSON object (no markdown fences) with keys:
- quick_checks: list[string]   # commands or checks
- likely_failures: list[string]
- risk_notes: list[string]
"""
