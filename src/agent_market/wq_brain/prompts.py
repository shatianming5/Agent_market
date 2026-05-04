"""LLM prompt templates for WorldQuant BRAIN alpha generation."""
from __future__ import annotations

from typing import Any

from .operators import operators_prompt_block

PROMPT_VERSION = "wqb-v1.0"


def build_alpha_gen_prompt(
    *,
    batch_size: int = 5,
    iteration: int = 1,
    pool_summary: str = "",
    score_history_summary: str = "",
    region: str = "USA",
    universe: str = "TOP3000",
    kb_examples: list[str] | None = None,
) -> str:
    op_block = operators_prompt_block()
    pool_block = pool_summary or "Alpha pool is empty — generate fresh diverse alphas."
    history_block = (
        f"\n=== Score History (with failure reasons) ===\n{score_history_summary}"
        if score_history_summary
        else ""
    )
    kb_block = ""
    if kb_examples:
        kb_block = "\n=== Retrieved Similar Examples (from KB) ===\n"
        kb_block += "\n".join(f"  {ex}" for ex in kb_examples) + "\n"
    return f"""\
You are a quantitative researcher generating alpha factors for WorldQuant BRAIN.

{op_block}

=== Your Task ===
Generate exactly {batch_size} distinct WorldQuant FASTEXPR alpha expressions for:
  Region: {region}
  Universe: {universe}
  Delay: 1 (signal on T, fill at T+1 open — NO look-ahead)

=== Previously Submitted Alphas ===
{pool_block}
{history_block}
{kb_block}
=== VERIFIED RESULTS (real backtest — learn from these) ===
  rank(-ts_mean(returns, 60))                                      # sh=1.20 fi=0.62 to=0.10  FAIL (low return level)
  rank(-ts_delta(close,1)/ts_mean(volume,20) - ts_delta(close,5)/ts_mean(volume,60))  # sh=1.47 fi=0.78 to=0.39  FAIL
  rank(-ts_delta(close,1)/close*volume/adv20)                      # sh=1.77 fi=0.93 to=0.71  FAIL (to too high)
  rank(group_zscore(-returns, sector))                             # sh=1.70 fi=0.89 to=0.72  FAIL (to too high)
  rank(-ts_zscore(returns,20)*ts_mean(volume/adv20,20))            # sh=1.90 fi=0.87 to=0.69  FAIL
  rank(group_zscore(-ts_delta(close,1)/close*volume/adv20,sector)) # sh=1.79 fi=0.92 to=0.69  FAIL (sector adj barely helps)
  rank(-ts_mean(returns*volume/adv20, 10))                         # sh=1.11 fi=~0.6  FAIL (smoothing kills sharpe)

=== FITNESS FORMULA INSIGHT ===
  WQ fitness ≈ sqrt(|annualized_return|) * sharpe / sqrt(turnover)
  To pass fi>=1.0 with sh=1.8 and ret=20%/yr:  need to<=0.58
  CRITICAL OBSERVATION: All 1-day signals cluster at to=0.69-0.72 — adding sector neutralization barely moves to.
                        Smoothing Amihud with ts_mean(N>=5) reduces sharpe from 1.77 to <1.2 — NOT viable.
  STRATEGY — USE LONGER PRICE LAG (5d or 10d) instead of ts_delta(close,1):
    rank(-ts_delta(close,5)/close * volume/adv20)                  # expected to≈0.20-0.30, sh≈1.3-1.5
    rank(-ts_delta(close,5)/close * ts_mean(volume/adv20,5))       # 5d return * smoothed volume
    rank(-ts_delta(close,10)/close * ts_mean(volume/adv20,10))     # 10d price impact
  WHY: 5-day return Amihud: fi = sqrt(0.17)*1.4/sqrt(0.25) ≈ 1.15 → SHOULD PASS
  ALSO TRY medium-frequency momentum with low to:
    rank(ts_rank(close,252) - ts_rank(close,20))                   # 12m-1m momentum, to≈0.08
    rank(-ts_delta(close,60)/close)                                 # 60d reversal, to≈0.10
    rank(ts_mean(close/vwap-1,20))                                  # 20d close-to-VWAP

=== Rules ===
1. Each alpha MUST be a single expression that produces a per-stock score (float).
2. Use rank() or group_rank() as the outermost layer to normalise output.
3. Combine multiple signals for robustness (e.g. value + momentum, quality + mean-reversion).
4. Generate structurally DIVERSE alphas — different operator families, different fields.
5. AVOID repeating the same logic as already-submitted alphas shown above.
6. Do NOT use Python-only syntax (no lambda, import, def, class, print).
7. Each expression should be concise (< 200 characters ideally).
8. Provide brief economic intuition for each alpha.
9. ONLY use fields: open, close, high, low, vwap, volume, adv20, returns, sector, industry, subindustry.
10. DO NOT use adv60, adv120, adv180, cap, or any fundamental fields.
11. AVOID nesting group_* inside ts_* — e.g., ts_mean(group_zscore(...)) causes timeouts.
    OK: rank(group_zscore(ts_mean(returns,20), sector))  (group applied OUTSIDE ts)
    BAD: rank(ts_mean(group_zscore(returns, sector), 20))  (group INSIDE ts — slow!)

=== Output Format ===
Write ONLY a JSON file named `candidate.json` (no other text) with this schema:
{{
  "candidates": [
    {{
      "expr": "<FASTEXPR expression>",
      "rationale": "<one sentence: what market inefficiency this captures>"
    }},
    ...
  ],
  "prompt_version": "{PROMPT_VERSION}",
  "iteration": {iteration}
}}

IMPORTANT: Write only the JSON file. Do not output anything else.
Use the `file` or `terminal` tool to write `candidate.json` in your working directory.
"""


def build_repair_prompt(failed_expr: str, error_msg: str) -> str:
    op_block = operators_prompt_block()
    return f"""\
You are a quantitative researcher debugging a WorldQuant FASTEXPR expression.

{op_block}

=== Failed Expression ===
{failed_expr}

=== Error Message ===
{error_msg}

=== Task ===
Fix the expression so it compiles in WorldQuant BRAIN.
Common issues:
- Unsupported operators → replace with operators from the reference list
- Missing rank() wrapper → add rank() as the outermost call
- Python syntax → use only FASTEXPR operators

Output ONLY a JSON file `candidate.json`:
{{
  "candidates": [
    {{
      "expr": "<fixed expression>",
      "rationale": "Fixed: <brief description of what was wrong>"
    }}
  ],
  "prompt_version": "{PROMPT_VERSION}",
  "iteration": 0
}}
"""


def score_history_to_summary(score_history: list[dict], *, last_n: int = 10) -> str:
    """Convert score_history to a compact summary including failure reasons."""
    if not score_history:
        return ""
    recent = score_history[-last_n:]
    lines = []
    for h in recent:
        parts = [
            f"iter={h.get('iteration','?')} "
            f"passed={h.get('passed',0)}/{h.get('simulated','?')}"
        ]
        if h.get("top_sharpe") is not None:
            parts.append(f"sh={h['top_sharpe']:.2f}")
        if h.get("top_fitness") is not None:
            parts.append(f"fi={h['top_fitness']:.2f}")
        failures = h.get("failure_summary", {})
        if failures:
            top_fails = sorted(failures.items(), key=lambda x: -x[1])[:3]
            parts.append("fails=" + ",".join(f"{k}:{v}" for k, v in top_fails))
        if h.get("top_expr"):
            parts.append(f'best="{h["top_expr"][:60]}"')
        lines.append("  " + " | ".join(parts))
    return "\n".join(lines)
