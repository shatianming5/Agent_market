#!/usr/bin/env python
"""
Idea Agent
----------

Given a high level trading idea or market observation, craft a structured
factor-design brief that can be refined and handed over to the expression
generation agent.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict

from agent_market.freqai import llm as llm_utils


def _load_idea_text(args: argparse.Namespace) -> str:
    if args.idea:
        return args.idea.strip()
    if args.idea_file:
        return Path(args.idea_file).read_text(encoding="utf-8").strip()
    raise ValueError("Idea text is required via --idea or --idea-file.")


def _build_prompt(idea_text: str, context: str | None = None) -> str:
    addon = f"\nAdditional context:\n{context}" if context else ""
    return textwrap.dedent(
        f"""
        You are a senior quantitative researcher. Convert the following trading idea into a structured design brief.

        Base idea:
        \"\"\"
        {idea_text}
        \"\"\"
        {addon}

        Return a JSON object with the following keys:
          - idea_title: concise summary (< 12 words).
          - narrative: 3-5 sentence overview of the intuition.
          - market_regime: bullet list (array of strings) describing regimes where the idea should perform well or fail.
          - target_horizon: dictionary with keys `label_period_candles`, `timeframe`, `lookback_candles`.
          - candidate_features: array of objects with keys `name`, `description`, `data_source`, `reasoning`.
          - expected_signals: array of objects with keys `signal_name`, `mechanism`, `leading_or_lagging`, `expected_direction`.
          - risk_flags: array of strings enumerating pitfalls, overfitting risks, or data issues.
          - evaluation_focus: array of objects with keys `metric`, `target`, and `reason`, capturing quantitative thresholds (e.g. ic_pearson >= 0.04).

        Constraints:
          - Use valid JSON (double quotes).
          - Keep numeric values realistic and include units where relevant.
          - Be explicit about data frequency and pairing.
        """
    ).strip()


def _parse_json_response(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # attempt to locate first/last braces
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


def run(args: argparse.Namespace) -> Dict[str, Any]:
    idea_text = _load_idea_text(args)
    prompt = _build_prompt(idea_text, args.context)
    config = llm_utils.LLMConfig.from_args(args)
    raw_content, usage = llm_utils.request_completion(prompt, config)
    payload = _parse_json_response(raw_content)
    payload["_idea_source"] = idea_text
    payload["_llm_usage"] = usage
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate structured factor design brief from a free-form idea.")
    parser.add_argument("--idea", help="Idea text supplied inline.")
    parser.add_argument("--idea-file", help="Path to a text/markdown file containing the idea.")
    parser.add_argument("--context", help="Optional supplementary context injected into the prompt.")
    parser.add_argument("--output", default="user_data/reports/idea_raw.json")

    parser.add_argument("--llm-base-url", default=llm_utils.DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=llm_utils.DEFAULT_MODEL)
    parser.add_argument("--llm-api-key", default=llm_utils.DEFAULT_API_KEY)
    parser.add_argument("--llm-count", type=int, default=1)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--llm-timeout", type=float, default=llm_utils.DEFAULT_TIMEOUT)
    parser.add_argument("--llm-temperature", type=float, default=0.15)
    parser.add_argument("--llm-max-tokens", type=int, default=800)

    args = parser.parse_args()
    result = run(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    title = result.get("idea_title") or "Untitled Idea"
    print(f"[idea-agent] Wrote design brief '{title}' -> {output_path}")


if __name__ == "__main__":
    main()
