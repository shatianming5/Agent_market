from __future__ import annotations

import json
import os
import re
import textwrap
import time
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_dotenv_fallback(path: Path) -> None:
    """
    Minimal `.env` loader (no external deps).
    - Ignores comments / blank lines.
    - Supports `export KEY=VALUE`.
    - Supports quoted values (single/double).
    - Uses `os.environ.setdefault` (shell env wins).
    """

    if not path.exists():
        return
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("export "):
            s = s[7:].strip()
        if "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and ((value[0] == value[-1] == "'") or (value[0] == value[-1] == '"')):
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_dotenv_fallback(PROJECT_ROOT / ".env")

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


DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
DEFAULT_MODEL = os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-3.5-turbo"
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
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

    base_url = config.base_url.rstrip("/")
    parsed = urlparse(base_url)
    path = (parsed.path or "").rstrip("/")
    if not (path.endswith("/v1") or "/v1/" in (path + "/")):
        base_url = base_url + "/v1"
    url = base_url + "/chat/completions"
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
    payload_json_mode = {**payload, "response_format": {"type": "json_object"}}

    last_exc: Optional[Exception] = None
    json_mode_supported = True
    for attempt in range(1, max(config.retries, 1) + 1):
        try:
            req_payload = payload_json_mode if json_mode_supported else payload
            response: Response = requests.post(url, headers=headers, json=req_payload, timeout=config.timeout)
            if response.status_code >= 400:
                text = response.text[:400]
                if (
                    json_mode_supported
                    and response.status_code == 400
                    and ("response_format" in text or "Unknown parameter" in text or "unknown parameter" in text)
                ):
                    json_mode_supported = False
                    raise ValueError("LLM json_mode not supported; retry without response_format")
                raise ValueError(f"LLM request failed {response.status_code}: {text}")
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
    variants: list[str] = []
    text = payload.strip()
    if text.startswith("```"):
        # Strip fenced blocks (```json ... ```)
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
        text = text.strip()
    variants.append(text)
    try:
        start = text.index("{")
        end = text.rindex("}")
        variants.append(text[start : end + 1])
    except ValueError:
        pass

    last_exc: Optional[Exception] = None
    for candidate in variants:
        for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError as exc:
                last_exc = exc
                continue
    if last_exc is not None:
        raise last_exc
    raise json.JSONDecodeError("No JSON object found", payload, 0)


def extract_candidates(raw_content: str) -> List[Dict[str, Any]]:
    try:
        payload = _extract_json_object(raw_content)
        expressions_raw = payload.get("expressions")
        if not isinstance(expressions_raw, list):
            raise ValueError("LLM response missing expressions list")
    except (json.JSONDecodeError, ValueError):
        return _extract_candidates_fallback(raw_content)

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


def _extract_candidates_fallback(raw_content: str) -> List[Dict[str, Any]]:
    """
    Fallback parser when the model doesn't return strict JSON.
    Extracts `"expression": "..."` patterns and returns them as candidates.
    """

    text = raw_content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
        text = text.strip()

    expr_pat = re.compile(r'"expression"\s*:\s*"((?:\\.|[^"\\])*)"', flags=re.IGNORECASE)
    name_pat = re.compile(r'"name"\s*:\s*"((?:\\.|[^"\\])*)"', flags=re.IGNORECASE)
    cat_pat = re.compile(r'"category"\s*:\s*"((?:\\.|[^"\\])*)"', flags=re.IGNORECASE)
    desc_pat = re.compile(r'"description"\s*:\s*"((?:\\.|[^"\\])*)"', flags=re.IGNORECASE)
    reason_pat = re.compile(r'"reason"\s*:\s*"((?:\\.|[^"\\])*)"', flags=re.IGNORECASE)

    def _unescape(raw: str) -> str:
        try:
            return json.loads(f'"{raw}"')
        except Exception:
            return raw.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")

    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for idx, match in enumerate(expr_pat.finditer(text)):
        expr = _unescape(match.group(1)).strip()
        if not expr or expr in seen:
            continue
        seen.add(expr)

        window = text[max(0, match.start() - 500) : match.end() + 500]
        name_match = None
        for m in name_pat.finditer(window):
            name_match = m
        name = _unescape(name_match.group(1)).strip() if name_match else f"llm_expr_{idx}"
        desc_match = None
        for m in desc_pat.finditer(window):
            desc_match = m
        reason_match = None
        for m in reason_pat.finditer(window):
            reason_match = m
        cat_match = None
        for m in cat_pat.finditer(window):
            cat_match = m
        meta = {
            "description": _unescape(desc_match.group(1)).strip() if desc_match else None,
            "reason": _unescape(reason_match.group(1)).strip() if reason_match else None,
            "category": _unescape(cat_match.group(1)).strip() if cat_match else None,
        }
        candidates.append({"expression": expr, "name": name, **meta})
        if len(candidates) >= 100:
            break
    if not candidates:
        raise ValueError("LLM response missing expressions list")
    return candidates
