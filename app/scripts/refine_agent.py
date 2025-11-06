#!/usr/bin/env python
"""
Refiner Agent
-------------

Consume the structured brief from idea_agent and translate it into a
detailed specification that the factor expression generator can follow.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict

from agent_market.freqai import llm as llm_utils


def _build_prompt(idea_payload: Dict[str, Any], context: str | None = None) -> str:
    idea_json = json.dumps(idea_payload, ensure_ascii=False, indent=2)
    extra = f"\nAdditional context:\n{context}" if context else ""
    return textwrap.dedent(
        f"""
        You are the lead architect of a quantitative research workflow.
        Refine the following design brief into a specification the expression generation agent can obey.

        Original brief (JSON):
        ```json
        {idea_json}
        ```
        {extra}

        Respond with valid JSON containing:
          - spec_id: short identifier (snake_case).
          - primary_objective: string.
          - time_horizon: object with `timeframe`, `label_period_candles`, `lookback_windows` (array).
          - feature_blocks: array where each item contains `name`, `features` (array of feature hints), and `rationale`.
          - signal_blueprints: array with each item containing
              * name
              * narrative
              * operations: ordered list describing the transforms (e.g. ["z(feat_x)", "rolling_mean(...)", "ratio(...)"])
              * required_inputs: array of feature names.
              * failure_modes: array of strings.
          - optimisation_guidelines: object with fields `metrics` (array), `target_ranges` (object), `stress_tests` (array of strings).
          - expression_constraints: object containing
              * allowed_functions: array of helper names the factor agent may use.
              * disallowed_patterns: array of regex/fragments to avoid.
              * normalisation_hint: string.
              * quality_thresholds: object mapping metric names to minimum/maximum targets (e.g. ic_pearson_min, ic_ir_min).
              * forbidden_constructs: array of string descriptions for expressions that should be excluded (e.g. "pure price-only signals").
          - backtest_plan: object with `timerange`, `pairs`, `freqaimodel_candidates`, `notes`.

        Rules:
          - Produce strict JSON (double quotes, no trailing commas).
          - Keep arrays concise but informative (<=5 items where possible).
          - Ensure allowed_functions align with FreqAI helper names (z, abs, clip, ema, roll_mean, pct_change, etc.).
        """
    ).strip()


def _load_payload(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Idea payload must be a JSON object.")
    return data


def _parse_json_response(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


def run(args: argparse.Namespace) -> Dict[str, Any]:
    idea_path = Path(args.input)
    idea_payload = _load_payload(idea_path)
    prompt = _build_prompt(idea_payload, args.context)
    config = llm_utils.LLMConfig.from_args(args)
    raw_content, usage = llm_utils.request_completion(prompt, config)
    spec = _parse_json_response(raw_content)
    spec["_idea_file"] = str(idea_path)
    spec["_llm_usage"] = usage
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine idea brief into structured design specification.")
    parser.add_argument("--input", required=True, help="Idea JSON produced by scripts/idea_agent.py.")
    parser.add_argument("--context", help="Optional supplemental context.")
    parser.add_argument("--output", default="user_data/reports/idea_spec.json")

    parser.add_argument("--llm-base-url", default=llm_utils.DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=llm_utils.DEFAULT_MODEL)
    parser.add_argument("--llm-api-key", default=llm_utils.DEFAULT_API_KEY)
    parser.add_argument("--llm-count", type=int, default=1)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--llm-timeout", type=float, default=llm_utils.DEFAULT_TIMEOUT)
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--llm-max-tokens", type=int, default=1000)

    args = parser.parse_args()
    result = run(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[refine-agent] Wrote spec {result.get('spec_id', '<unknown>')} -> {output_path}")


if __name__ == "__main__":
    main()
