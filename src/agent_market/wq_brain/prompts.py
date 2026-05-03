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
) -> str:
    op_block = operators_prompt_block()
    pool_block = pool_summary or "Alpha pool is empty — generate fresh diverse alphas."
    history_block = (
        f"\n=== Score History ===\n{score_history_summary}"
        if score_history_summary
        else ""
    )
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

=== Rules ===
1. Each alpha MUST be a single expression that produces a per-stock score (float).
2. Use rank() or group_rank() as the outermost layer to normalise output.
3. Combine multiple signals for robustness (e.g. value + momentum, quality + mean-reversion).
4. Generate structurally DIVERSE alphas — different operator families, different fields.
5. AVOID repeating the same logic as already-submitted alphas shown above.
6. Do NOT use Python-only syntax (no lambda, import, def, class, print).
7. Each expression should be concise (< 200 characters ideally).
8. Provide brief economic intuition for each alpha.

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
    """Convert score_history list to a compact text summary for the LLM."""
    if not score_history:
        return ""
    recent = score_history[-last_n:]
    lines = []
    for h in recent:
        it = h.get("iteration", "?")
        passed = h.get("passed", 0)
        submitted = h.get("submitted", 0)
        top_sharpe = h.get("top_sharpe")
        top_fitness = h.get("top_fitness")
        parts = [f"iter={it} passed={passed}/{h.get('simulated', '?')} submitted={submitted}"]
        if top_sharpe is not None:
            parts.append(f"top_sharpe={top_sharpe:.2f}")
        if top_fitness is not None:
            parts.append(f"top_fitness={top_fitness:.2f}")
        lines.append("  " + " | ".join(parts))
    return "\n".join(lines)
