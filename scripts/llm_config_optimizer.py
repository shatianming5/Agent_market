#!/usr/bin/env python3
"""LLM-driven configuration optimizer for Agent Market.

Uses the LLM to analyze backtest results and suggest parameter improvements.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _llm_request(prompt: str, *, system: str = "") -> str:
    """Call OpenAI-compatible LLM."""
    import requests  # noqa: PLC0415

    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:4141/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "_")
    model = os.environ.get("OPENAI_MODEL", "gpt-5.2")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 2048},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LLM config optimizer")
    parser.add_argument("--config", required=True, help="Current config JSON")
    parser.add_argument("--output", required=True, help="Output optimized config JSON")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    # Load latest backtest summary
    feedback_path = ROOT / "user_data" / "llm_feedback" / "latest_backtest_summary.json"
    feedback = {}
    if feedback_path.exists():
        feedback = json.loads(feedback_path.read_text(encoding="utf-8"))

    system = """You are a quantitative trading strategy optimizer.
You analyze backtest results and suggest parameter improvements for LightGBM models and expression generation.
You MUST output valid JSON only, no markdown, no explanation."""

    prompt = f"""Current agent_flow config:
{json.dumps(config, indent=2)}

Latest backtest results:
{json.dumps(feedback, indent=2)}

Based on these results, suggest an optimized config. Focus on:
1. ml_training params: learning_rate, num_leaves, num_boost_round, subsample, feature_fraction
2. expression args: --top-n, --llm-count, --llm-loops, --llm-temperature
3. training params: validation_ratio, purge, embargo

Rules:
- Keep the same JSON structure exactly
- Only change numeric parameters, not paths or script names
- learning_rate should be between 0.01 and 0.1
- num_leaves should be between 15 and 127
- label_period should stay at 12 (tested, others are worse)
- Be conservative: small changes from current values

Output the complete optimized config as valid JSON:"""

    print("[llm_optimizer] Requesting optimization suggestions...", file=sys.stderr)
    response = _llm_request(prompt, system=system)

    # Extract JSON from response
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        optimized = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            optimized = json.loads(text[start:end])
        else:
            print(f"[llm_optimizer] Failed to parse LLM response as JSON", file=sys.stderr)
            sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(optimized, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[llm_optimizer] Wrote optimized config to {out_path}", file=sys.stderr)

    # Show diff
    for section in ["expression", "ml_training"]:
        old = config.get(section, {})
        new = optimized.get(section, {})
        if old != new:
            print(f"[llm_optimizer] Changed in {section}:", file=sys.stderr)


if __name__ == "__main__":
    main()
