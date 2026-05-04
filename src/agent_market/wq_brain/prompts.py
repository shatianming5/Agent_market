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
    forced_direction: str | None = None,
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

    direction_block = ""
    if forced_direction:
        direction_block = f"""
=== MANDATORY DIRECTION FOR THIS ITERATION ===
{forced_direction}
You MUST follow this direction for ALL {batch_size} candidates. Do NOT generate Amihud/ts_delta/ts_rank
variations — those have been exhausted. This direction is REQUIRED, not optional.

"""

    return f"""\
You are a quantitative researcher generating alpha factors for WorldQuant BRAIN.

{op_block}

=== Your Task ===
Generate exactly {batch_size} distinct WorldQuant FASTEXPR alpha expressions for:
  Region: {region}
  Universe: {universe}
  Delay: 1 (signal on T, fill at T+1 open — NO look-ahead)
{direction_block}
=== Previously Submitted Alphas ===
{pool_block}
{history_block}
{kb_block}
=== CALIBRATED RESULTS FROM ACTUAL WQ SIMULATION ===
  NOTE: These are REAL simulation results — the estimates in textbooks are wrong.
  rank(-ts_delta(close,5)/close * ts_mean(volume/adv20,5))          # sh=1.12 fi=0.66 to=0.35  FAIL
  rank(ts_rank(close,252) * (-ts_delta(close,5)/close))             # sh=1.35 fi=0.76 to=0.36  FAIL (best seen)
  rank(-ts_delta(close,5)/close * ts_mean(volume/adv20,60))         # sh=1.06 fi=0.62 to=0.35  FAIL
  rank(-ts_delta(close,10)/close * ts_mean(volume/adv20,10))        # sh=0.54 fi=0.27 to=0.23  FAIL
  rank(ts_rank(close,252) - ts_rank(close,20))                      # sh=0.57 fi=0.33 to=0.17  FAIL
  rank(-ts_delta(close,60)/close)                                    # sh=0.64 fi=0.54 to=0.09  FAIL
  rank(group_zscore(-ts_delta(close,5)/close, sector))              # sh=1.06 fi=0.62 to=0.35  FAIL

  CONCLUSION: All Amihud/ts_delta/ts_rank(close) variants FAIL. DO NOT generate more.
  To pass: need fi>=1.0 AND sh>=1.25. Gap to close: fi needs to reach 1.0 from current best 0.76.
  Try COMPLETELY DIFFERENT alpha families: ts_corr, (high-low)/close, open gap, volume rank patterns.

=== Fitness Calibration ===
  fi ≈ sqrt(|annual_return|) × sharpe / sqrt(turnover)
  Best seen: sh=1.35, fi=0.76, to=0.36, implied annual_return≈11%
  To reach fi=1.0 with sh=1.35: need annual_return≥20% OR turnover≤0.18

=== Rules ===
1. Each alpha MUST be a single expression that produces a per-stock score (float).
2. Use rank() or group_rank() as the outermost layer to normalise output.
3. AVOID all patterns listed in "CALIBRATED RESULTS" above — they are exhausted.
4. Follow the MANDATORY DIRECTION above — generate variations within that family only.
5. Do NOT use Python-only syntax (no lambda, import, def, class, print).
6. Each expression should be concise (< 200 characters ideally).
7. ONLY use fields: open, close, high, low, vwap, volume, adv20, returns, sector, industry, subindustry.
8. DO NOT use adv60, adv120, adv180, cap, or any fundamental fields.
9. AVOID nesting group_* inside ts_* — causes timeouts.
   OK: rank(group_zscore(ts_mean(returns,20), sector))
   BAD: rank(ts_mean(group_zscore(returns, sector), 20))

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
