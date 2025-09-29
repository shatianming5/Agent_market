from __future__ import annotations

import json
import os
import textwrap
import time
from pathlib import Path
from dataclasses import dataclass
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if load_dotenv:
    load_dotenv(PROJECT_ROOT / '.env')

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from requests import Response
from requests.exceptions import RequestException

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_API_KEY",
    "DEFAULT_TIMEOUT",
    "ALLOWED_FUNCTIONS",
    "LLMConfig",
    "build_prompt",
    "request_completion",
    "extract_candidates",
]


DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.zhizengzeng.com/v1")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
DEFAULT_API_KEY = (
    os.environ.get("LLM_API_KEY")
    or os.environ.get("ZHIZENGZENG_API_KEY")
    or os.environ.get("ZHIZENGZENG_APIKEY")
    or ""
)
DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "45"))

ALLOWED_FUNCTIONS = [
    ("z(column)", "z-score ???"),
    ("abs(series)", "????"),
    ("shift(series, n)", "???? n ?"),
    ("roll_mean(series, window)", "????"),
    ("roll_std(series, window)", "?????"),
    ("pct_change(series, n)", "n ???????"),
    ("sign(series)", "???? -1/0/1"),
    ("clip(series, lower, upper)", "??????"),
    ("ema(series, span)", "??????"),
    ("rolling_max(series, window)", "?????"),
    ("rolling_min(series, window)", "?????"),
    ("log1p(series)", "log(1+x)"),
    ("tanh(series)", "??????"),
]


@dataclass
class LLMConfig:
    """???? LLM ?????"""

    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    model: str = DEFAULT_MODEL
    temperature: float = 0.2
    max_tokens: int = 1024
    count: int = 50
    retries: int = 3
    timeout: float = DEFAULT_TIMEOUT

    @classmethod
    def from_args(cls, args: Any) -> "LLMConfig":
        return cls(
            base_url=getattr(args, "llm_base_url", DEFAULT_BASE_URL) or DEFAULT_BASE_URL,
            api_key=getattr(args, "llm_api_key", DEFAULT_API_KEY) or DEFAULT_API_KEY,
            model=getattr(args, "llm_model", DEFAULT_MODEL) or DEFAULT_MODEL,
            temperature=float(getattr(args, "llm_temperature", 0.2)),
            max_tokens=int(getattr(args, "llm_max_tokens", 1024)),
            count=int(getattr(args, "llm_count", 50)),
            retries=int(getattr(args, "llm_retries", 3)),
            timeout=float(getattr(args, "llm_timeout", DEFAULT_TIMEOUT)),
        )


def _feature_metadata_map(feature_cfg: Dict) -> Dict[str, Dict]:
    return {item.get("name"): item for item in feature_cfg.get("features", []) if item.get("name")}


def build_feature_glossary(
    feature_cfg: Dict,
    feature_cols: Sequence[str],
    combos: Sequence[Dict],
    max_items: int = 60,
) -> str:
    meta_map = _feature_metadata_map(feature_cfg)
    lines: List[str] = []
    for col in feature_cols:
        info = meta_map.get(col, {})
        parts: List[str] = [col]
        if info.get("type"):
            parts.append(f"type={info['type']}")
        if info.get("period") is not None:
            parts.append(f"period={info['period']}")
        metrics: List[str] = []
        for key in ("score", "correlation", "mutual_info"):
            value = info.get(key)
            if isinstance(value, (int, float)):
                metrics.append(f"{key}={value:.4f}")
        if metrics:
            parts.append(", ".join(metrics))
        description = info.get("description")
        if description:
            parts.append(description)
        lines.append("- " + " | ".join(parts))

    combo_lines: List[str] = []
    for combo in combos:
        name = combo.get("name")
        if not name or name not in feature_cols:
            continue
        formula = combo.get("formula") or " / ".join(combo.get("sources", []))
        combo_lines.append(f"- {name} | {combo.get('type')} | {formula}")

    all_lines = lines + combo_lines
    if not all_lines:
        return "No engineered features found."
    if len(all_lines) > max_items:
        remaining = len(all_lines) - max_items
        return "\n".join(all_lines[:max_items]) + f"\n- ...(truncated {remaining} lines)"
    return "\n".join(all_lines)


def _format_allowed_functions() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in ALLOWED_FUNCTIONS)


def build_prompt(
    feature_cfg: Dict,
    feature_cols: Sequence[str],
    combos: Sequence[Dict],
    timeframe: str,
    label_period: int,
    request_count: int,
    avoid_expressions: Optional[Sequence[str]] = None,
    feedback: Optional[str] = None,
) -> str:
    glossary = build_feature_glossary(feature_cfg, feature_cols, combos)
    functions_doc = _format_allowed_functions()
    pairs = feature_cfg.get("pairs", [])
    exchange = feature_cfg.get("exchange", "unknown")
    selection_methods = feature_cfg.get("selection_methods", [])
    prompt = textwrap.dedent(
        f"""
        Role: Senior quantitative factor engineer responsible for discovering predictive expressions.
        Goal: Propose {request_count} composite expressions that help forecast forward returns for the next {label_period} candles.

        Data context:
        - Exchange: {exchange}
        - Pairs: {', '.join(pairs) if pairs else 'unspecified'}
        - Timeframe: {timeframe}
        - Label horizon: {label_period} candles
        - Feature selection methods: {', '.join(selection_methods) if selection_methods else 'unspecified'}

        Available feature columns:
        {glossary}

        Allowed helper functions:
        {functions_doc}

        Output policy:
        - Use only the listed column names. Do not invent new variables or rely on external data.
        - Only use the allowed helper functions plus basic arithmetic operations.
        - Respond with a single JSON object shaped as {{"expressions": [ ... ]}}.
        - Each item must contain:
            * name: short snake_case identifier (< 15 chars).
            * expression: executable Python expression string.
            * description: concise human explanation (Chinese or English).
            * reason: rationale for predictive power (Chinese or English).
            * category: choose one of ['trend','momentum','volatility','volume','mean_reversion','other'].
        - Expressions should remain numerically stable (avoid divide by zero, use +1e-6 if needed).
        - Prefer combinations that complement each other (diversified signals).
        """
    )

    if avoid_expressions:
        avoid_list = [expr for expr in avoid_expressions if expr]
        if avoid_list:
            listed = '\n'.join(f'- {expr}' for expr in avoid_list[:50])
            prompt += (
                '\n\n        Previously generated expressions to avoid:'
                f'\n{listed}'
                '\n        - ??????????'
            )
    if feedback:
        prompt += textwrap.dedent(
            f"""
        Recent backtest feedback (use these observations to improve predictive coverage and robustness):
        {feedback}
        """
        )
    return prompt.strip()


def request_completion(prompt: str, config: LLMConfig) -> Tuple[str, Optional[Dict[str, Any]]]:
    if not config.api_key:
        raise ValueError("?? LLM ???????? API Key?")

    url = config.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert quantitative factor engineer. Always reply with valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    last_exc: Optional[Exception] = None
    for attempt in range(1, max(config.retries, 1) + 1):
        try:
            response: Response = requests.post(url, headers=headers, json=payload, timeout=config.timeout)
            if response.status_code >= 400:
                raise ValueError(f"LLM request failed {response.status_code}: {response.text[:200]}")
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise ValueError("LLM response missing choices")
            content = choices[0].get("message", {}).get("content")
            if not content:
                raise ValueError("LLM response missing message.content")
            return content, data.get("usage")
        except (RequestException, ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
            print(f"[llm] call failed ({attempt}/{config.retries}): {exc}")
            if attempt == config.retries:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise last_exc  # type: ignore[misc]


def _extract_json_object(payload: str) -> Dict[str, Any]:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        match = None
        try:
            start = payload.index("{")
            end = payload.rindex("}")
            match = payload[start : end + 1]
        except ValueError as exc:  # noqa: BLE001
            raise json.JSONDecodeError("No JSON object found", payload, 0) from exc
        return json.loads(match)


def extract_candidates(raw_content: str) -> List[Dict[str, Any]]:
    payload = _extract_json_object(raw_content)
    expressions_raw = payload.get("expressions")
    if not isinstance(expressions_raw, list):
        raise ValueError("LLM response missing expressions list")

    cleaned: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for idx, item in enumerate(expressions_raw):
        if isinstance(item, str):
            expr = item.strip()
            if not expr:
                continue
            name = f"llm_expr_{idx}"
            meta: Dict[str, Any] = {}
        elif isinstance(item, dict):
            expr = str(item.get("expression") or item.get("formula") or "").strip()
            if not expr:
                continue
            name = str(item.get("name") or f"llm_expr_{idx}")
            meta = {
                "description": item.get("description"),
                "reason": item.get("reason") or item.get("rationale"),
                "category": item.get("category"),
            }
        else:
            continue
        if expr in seen:
            continue
        seen.add(expr)
        cleaned.append({"expression": expr, "name": name, **meta})
    return cleaned

