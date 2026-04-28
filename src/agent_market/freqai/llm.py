from __future__ import annotations

import json
import os
import re
import signal
import shutil
import threading
import textwrap
import time
from contextlib import contextmanager
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
    "extract_factor_spec_payload",
    "load_factor_spec_prompt_assets",
    "parse_factor_spec",
    "request_factor_spec_completion",
]


DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
DEFAULT_MODEL = os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-3.5-turbo"
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "45"))
DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER") or "openai_compatible"
DEFAULT_AGENT_URL = os.environ.get("OPENCODE_URL") or ""
DEFAULT_WORKSPACE = os.environ.get("LLM_WORKSPACE") or str(PROJECT_ROOT)
DEFAULT_AGENT_MAX_TURNS = int(os.environ.get("LLM_AGENT_MAX_TURNS", "12"))
DEFAULT_AGENT_STALE_TIMEOUT = float(os.environ.get("LLM_AGENT_STALE_TIMEOUT", "180"))
DEFAULT_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT") or os.environ.get("OPENAI_REASONING_EFFORT") or ""
DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4096"))

ALLOWED_FUNCTIONS = [
    ("z(column)", "z-score normalization"),
    ("ts_z(series, window)", "time-series z-score over a rolling window"),
    ("abs(series)", "absolute value"),
    ("shift(series, n)", "shift series by n bars"),
    ("diff(series, n)", "difference: series - shift(series, n)"),
    ("roll_mean(series, window)", "rolling mean"),
    ("roll_std(series, window)", "rolling standard deviation"),
    ("rolling_mean(series, window)", "rolling mean (alias for roll_mean)"),
    ("rolling_max(series, window)", "rolling maximum"),
    ("rolling_min(series, window)", "rolling minimum"),
    ("rolling_sum(series, window)", "rolling sum"),
    ("pct_change(series, n)", "n-bar percentage change"),
    ("sign(series)", "sign function: -1/0/1"),
    ("clip(series, lower, upper)", "clip to [lower, upper]"),
    ("ema(series, span)", "exponential moving average"),
    ("log1p(series)", "log(1+x), safe for near-zero values"),
    ("log(series)", "natural log (use log1p for near-zero safety)"),
    ("sqrt(series)", "square root"),
    ("exp(series)", "exponential"),
    ("tanh(series)", "hyperbolic tangent squash to [-1, 1]"),
    ("decay_linear(series, window)", "linearly-weighted rolling mean (recent bars weighted more)"),
    ("robust_z(series, window)", "robust z-score using median/MAD instead of mean/std"),
    ("winsorize(series, lower_pct, upper_pct)", "winsorize outliers at given percentiles"),
    ("rank_xs(series)", "cross-sectional rank in [0, 1]"),
    ("zscore(series)", "z-score (1 arg) or time-series z-score (2 args with window)"),
    ("ifelse(condition, true_val, false_val)", "conditional: like np.where"),
]


@dataclass
class LLMConfig:
    """???? LLM ?????"""

    base_url: str = DEFAULT_BASE_URL
    agent_url: str = DEFAULT_AGENT_URL
    api_key: str = DEFAULT_API_KEY
    model: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER
    workspace: str = DEFAULT_WORKSPACE
    temperature: float = 0.2
    max_tokens: int = DEFAULT_MAX_TOKENS
    count: int = 50
    retries: int = 3
    timeout: float = DEFAULT_TIMEOUT
    max_turns: int = DEFAULT_AGENT_MAX_TURNS
    stale_timeout: float = DEFAULT_AGENT_STALE_TIMEOUT
    reasoning_effort: str = DEFAULT_REASONING_EFFORT

    @classmethod
    def from_args(cls, args: Any) -> "LLMConfig":
        return cls(
            base_url=getattr(args, "llm_base_url", DEFAULT_BASE_URL) or DEFAULT_BASE_URL,
            agent_url=getattr(args, "llm_agent_url", DEFAULT_AGENT_URL) or DEFAULT_AGENT_URL,
            api_key=getattr(args, "llm_api_key", DEFAULT_API_KEY) or DEFAULT_API_KEY,
            model=getattr(args, "llm_model", DEFAULT_MODEL) or DEFAULT_MODEL,
            provider=getattr(args, "llm_provider", DEFAULT_PROVIDER) or DEFAULT_PROVIDER,
            workspace=getattr(args, "llm_workspace", DEFAULT_WORKSPACE) or DEFAULT_WORKSPACE,
            temperature=float(getattr(args, "llm_temperature", 0.2)),
            max_tokens=int(getattr(args, "llm_max_tokens", DEFAULT_MAX_TOKENS)),
            count=int(getattr(args, "llm_count", 50)),
            retries=int(getattr(args, "llm_retries", 3)),
            timeout=float(getattr(args, "llm_timeout", DEFAULT_TIMEOUT)),
            max_turns=int(getattr(args, "llm_max_turns", DEFAULT_AGENT_MAX_TURNS)),
            stale_timeout=float(getattr(args, "llm_stale_timeout", DEFAULT_AGENT_STALE_TIMEOUT)),
            reasoning_effort=(
                getattr(args, "llm_reasoning_effort", DEFAULT_REASONING_EFFORT)
                or DEFAULT_REASONING_EFFORT
            ),
        )


def _request_completion_via_opencode(
    prompt: str,
    config: LLMConfig,
    *,
    system_prompt: str,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    from agent_market.agents.executor import OpenCodeExecutor  # noqa: WPS433

    model = str(config.model or os.environ.get("OPENCODE_MODEL") or "").strip()
    if not model:
        raise ValueError("OpenCode provider requires llm_model or OPENCODE_MODEL.")

    workspace = Path(str(config.workspace or DEFAULT_WORKSPACE)).expanduser()
    if not workspace.is_absolute():
        workspace = (PROJECT_ROOT / workspace).resolve()
    else:
        workspace = workspace.resolve()

    if not workspace.exists():
        raise ValueError(f"OpenCode workspace does not exist: {workspace}")

    agent_url = str(config.agent_url or "").strip() or None
    if agent_url is None and not shutil.which("opencode"):
        raise ValueError("OpenCode provider requested but `opencode` CLI is not available and no llm_agent_url/OPENCODE_URL was provided.")

    full_prompt = textwrap.dedent(
        f"""
        {system_prompt}

        You are operating as the factor-mining agent for this repository.
        You may inspect repo files when helpful.
        Respond with exactly one JSON object and no surrounding prose.

        User task:
        {prompt}
        """
    ).strip()

    executor = OpenCodeExecutor(
        repo=workspace,
        model=model,
        base_url=agent_url,
        max_turns=max(1, int(config.max_turns)),
        stale_timeout=max(30.0, float(config.stale_timeout)),
        max_retries=max(0, int(config.retries)),
    )
    try:
        result = executor.run(full_prompt)
        return result.assistant_text, result.usage
    finally:
        executor.close()


def _feature_metadata_map(feature_cfg: Dict) -> Dict[str, Dict]:
    return {item.get("name"): item for item in feature_cfg.get("features", []) if item.get("name")}


def build_feature_glossary(
    feature_cfg: Dict,
    feature_cols: Sequence[str],
    combos: Sequence[Dict],
    max_items: int = 200,
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


def _content_to_text(content: Any) -> str:
    """Normalize OpenAI/Claude-compatible content payloads into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    return ""


def _extract_choice_text(choice: Dict[str, Any]) -> str:
    msg = choice.get("message") or {}
    if not isinstance(msg, dict):
        msg = {}
    content = _content_to_text(msg.get("content"))
    if content:
        return content
    for key in ("reasoning_content", "reasoning", "text"):
        content = _content_to_text(msg.get(key))
        if '"expressions"' in content:
            return content
    return _content_to_text(choice.get("text"))


def _token_fallbacks(max_tokens: int) -> List[int]:
    values = [max(1, int(max_tokens))]
    for value in (20000, 16000, 12000, 8192, 4096):
        if 0 < value < values[0] and value not in values:
            values.append(value)
    return values


@contextmanager
def _wall_timeout(seconds: float):
    seconds = float(seconds or 0)
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"LLM request exceeded wall timeout {seconds:.1f}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *old_timer)


def _post_chat_with_token_fallback(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: float,
) -> Response:
    response: Optional[Response] = None
    for max_tokens in _token_fallbacks(int(payload.get("max_tokens") or 4096)):
        req_payload = {**payload, "max_tokens": max_tokens}
        with _wall_timeout(float(timeout) + 5.0):
            response = requests.post(
                url,
                headers=headers,
                json=req_payload,
                timeout=timeout,
                proxies={"http": None, "https": None},
            )
        if response.status_code < 400:
            if max_tokens != int(payload.get("max_tokens") or max_tokens):
                print(f"[llm] max_tokens fallback accepted: {payload.get('max_tokens')} -> {max_tokens}", flush=True)
            return response
        text = response.text[:500].lower()
        if not any(marker in text for marker in ("max_tokens", "maximum", "too many", "context", "token")):
            return response
    assert response is not None
    return response


def build_prompt(
    feature_cfg: Dict,
    feature_cols: Sequence[str],
    combos: Sequence[Dict],
    timeframe: str,
    label_period: int,
    request_count: int,
    avoid_expressions: Optional[Sequence[str]] = None,
    feedback: Optional[str] = None,
    loop_feedback: Optional[Dict[str, Any]] = None,
) -> str:
    glossary = build_feature_glossary(feature_cfg, feature_cols, combos)
    functions_doc = _format_allowed_functions()
    pairs = feature_cfg.get("pairs", [])
    exchange = feature_cfg.get("exchange", "unknown")
    selection_methods = feature_cfg.get("selection_methods", [])

    # Category quota guidance from loop_feedback
    category_guidance = ""
    if loop_feedback:
        cat_dist = loop_feedback.get("category_distribution", {})
        if cat_dist:
            total = sum(cat_dist.values())
            saturated = [c for c, n in cat_dist.items() if n / max(total, 1) > 0.3]
            starved = [c for c in ["trend", "momentum", "volatility", "volume", "mean_reversion"]
                       if cat_dist.get(c, 0) / max(total, 1) < 0.1]
            if saturated or starved:
                category_guidance = "\n        Category balance guidance:\n"
                if saturated:
                    category_guidance += f"        - SATURATED (already have many): {', '.join(saturated)}. Generate fewer of these.\n"
                if starved:
                    category_guidance += f"        - NEEDED (under-represented): {', '.join(starved)}. Prioritize these categories.\n"

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
        - You may use comparisons (>, <, >=, <=) as regime filters and multiply booleans with signals (True=1, False=0).
        - Respond with a single JSON object shaped as {{"expressions": [ ... ]}}.
        - Each item must contain:
            * name: short snake_case identifier (< 15 chars).
            * expression: executable Python expression string.
            * description: concise human explanation (Chinese or English).
            * reason: rationale for predictive power (Chinese or English).
            * category: choose one of ['trend','momentum','volatility','volume','mean_reversion','other'].
        - Expressions should remain numerically stable (avoid divide by zero, use +1e-6 if needed).
        - Prefer combinations that complement each other (diversified signals).
        {category_guidance}"""
    )

    # Smart avoid: sample diverse subset instead of truncating to first 50
    if avoid_expressions:
        avoid_list = list(expr for expr in avoid_expressions if expr)
        if avoid_list:
            if len(avoid_list) <= 80:
                sampled = avoid_list
            else:
                # Take recent 30 + random 30 from the rest for diversity
                import random as _rng
                recent = avoid_list[-30:]
                older = avoid_list[:-30]
                random_sample = _rng.sample(older, min(30, len(older)))
                sampled = random_sample + recent
            listed = '\n'.join(f'- {expr}' for expr in sampled)
            prompt += (
                f'\n\n        Previously generated expressions ({len(avoid_list)} total, showing {len(sampled)} samples) — do NOT repeat these or trivial variants:'
                f'\n{listed}'
            )

    # Dynamic loop feedback: show top performers + what worked
    if loop_feedback:
        top_factors = loop_feedback.get("top_factors", [])
        if top_factors:
            top_lines = '\n'.join(
                f'- IC={f["ic"]:.3f} category={f["category"]} expr={f["expression"]}'
                for f in top_factors[:10]
            )
            prompt += textwrap.dedent(
                f"""

        Current top-performing factors (learn from these patterns and create DIFFERENT but similarly effective expressions):
{top_lines}
        - Focus on finding NEW signal sources, not minor variations of the above.
        """
            )
        weak_categories = loop_feedback.get("weak_categories", [])
        if weak_categories:
            prompt += f"\n        Categories needing more exploration: {', '.join(weak_categories)}\n"

    if feedback:
        prompt += textwrap.dedent(
            f"""
        Recent backtest feedback (use these observations to improve predictive coverage and robustness):
        {feedback}
        """
        )
    return prompt.strip()


def build_refine_prompt(
    *,
    factor: Dict[str, Any],
    feature_cols: Sequence[str],
    feature_cfg: Dict,
    per_pair_ic: Dict[str, float],
    timeframe: str,
    label_period: int,
    autocorr: float = 0.0,
    turnover: float = 0.0,
) -> str:
    """Build a prompt that asks the LLM to improve a specific factor."""
    glossary = build_feature_glossary(feature_cfg, list(feature_cols), feature_cfg.get("feature_combos", []))
    functions_doc = _format_allowed_functions()
    expr = factor.get("expression", "")
    ic = factor.get("metric_abs_ic", 0)
    category = factor.get("category", "other")

    # Per-pair analysis
    pair_lines = []
    strong_pairs = []
    weak_pairs = []
    for pair, pic in sorted(per_pair_ic.items()):
        tag = "✅" if abs(pic) > 0.1 else "⚠️" if abs(pic) > 0.05 else "❌"
        pair_lines.append(f"  {pair}: IC={pic:+.4f} {tag}")
        if abs(pic) > 0.1:
            strong_pairs.append(pair)
        elif abs(pic) < 0.05:
            weak_pairs.append(pair)

    pair_report = "\n".join(pair_lines)

    # Diagnose issues
    issues = []
    if weak_pairs:
        issues.append(f"Weak on {len(weak_pairs)} pairs: {', '.join(weak_pairs[:5])}")
    if autocorr > 0.95:
        issues.append(f"Signal changes too slowly (autocorr={autocorr:.3f}), need more responsive signal")
    if autocorr < 0.1:
        issues.append(f"Signal too noisy (autocorr={autocorr:.3f}), need more smoothing")
    if turnover > 0.4:
        issues.append(f"Too many trades (turnover={turnover:.2f}), add regime filter or smoothing")
    if turnover < 0.05:
        issues.append(f"Too few trades (turnover={turnover:.2f}), signal barely changes")

    issues_text = "\n".join(f"  - {i}" for i in issues) if issues else "  - No critical issues, but room for improvement"

    return textwrap.dedent(f"""
        Role: Senior quantitative factor engineer specializing in factor refinement.
        Task: Improve the following factor expression to achieve better cross-asset consistency and higher IC.

        ORIGINAL FACTOR:
          Expression: {expr}
          Category: {category}
          Mean |IC|: {ic:.4f}
          Autocorrelation(1): {autocorr:.3f}
          Turnover: {turnover:.2f}

        PER-PAIR IC BREAKDOWN:
        {pair_report}

        DIAGNOSED ISSUES:
        {issues_text}

        Available feature columns:
        {glossary}

        Allowed helper functions:
        {functions_doc}

        INSTRUCTIONS:
        Generate 5 improved variants of this factor. Each variant should:
        1. Address the diagnosed issues (especially weak pairs)
        2. Maintain the core signal idea but improve robustness
        3. Try different normalization, window sizes, or regime filters
        4. Keep expressions numerically stable

        Strategies to try:
        - If weak on some pairs: add cross-asset normalization (ts_z, robust_z, rank_xs)
        - If autocorrelation too high: use shorter windows or pct_change
        - If autocorrelation too low: add ema smoothing
        - If turnover too high: add regime filter (adx > threshold)
        - Combine with orthogonal signals (volume, volatility) for confirmation

        Respond with a single JSON object: {{"expressions": [...]}}
        Each item: name, expression, description, reason, category
    """).strip()


def request_completion(
    prompt: str,
    config: LLMConfig,
    *,
    system_prompt: str = "You are an expert quantitative factor engineer. Always reply with valid JSON.",
) -> Tuple[str, Optional[Dict[str, Any]]]:
    provider = str(config.provider or DEFAULT_PROVIDER).strip().lower() or "openai_compatible"

    if provider in ("opencode", "auto"):
        try:
            return _request_completion_via_opencode(prompt, config, system_prompt=system_prompt)
        except Exception:
            if provider == "opencode":
                raise

    if not config.api_key:
        raise ValueError("?? LLM ???????? API Key?")

    # Allow OPENAI_API_BASE for OpenAI-compatible gateways.
    if not config.base_url:
        config.base_url = os.environ.get("OPENAI_API_BASE") or config.base_url

    base_url = config.base_url.rstrip("/")
    parsed = urlparse(base_url)
    path = (parsed.path or "").rstrip("/")
    # Keep explicit versioned endpoints intact (e.g. BigModel /api/paas/v4).
    if not (
        path.endswith("/v1")
        or "/v1/" in (path + "/")
        or path.endswith("/v4")
        or "/v4/" in (path + "/")
        or "/api/paas/v4" in path
    ):
        base_url = base_url + "/v1"
    url = base_url + "/chat/completions"
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": str(system_prompt),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if str(config.reasoning_effort or "").strip():
        payload["reasoning_effort"] = str(config.reasoning_effort).strip()
    payload_json_mode = {**payload, "response_format": {"type": "json_object"}}

    last_exc: Optional[Exception] = None
    json_mode_supported = True
    retries = max(int(config.retries), 1)
    for attempt in range(1, retries + 1):
        try:
            req_payload = payload_json_mode if json_mode_supported else payload
            response: Response = _post_chat_with_token_fallback(url, headers, req_payload, config.timeout)
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
            content = _extract_choice_text(choices[0])
            if not content:
                if json_mode_supported and req_payload is payload_json_mode:
                    json_mode_supported = False
                    response = _post_chat_with_token_fallback(url, headers, payload, config.timeout)
                    if response.status_code >= 400:
                        raise ValueError(f"LLM request failed {response.status_code}: {response.text[:400]}")
                    data = response.json()
                    choices = data.get("choices") or []
                    if choices:
                        content = _extract_choice_text(choices[0])
                if not content:
                    finish_reason = choices[0].get("finish_reason") if choices else None
                    raise ValueError(f"LLM response missing message.content finish_reason={finish_reason!r}")
            return content, data.get("usage")
        except (RequestException, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
            print(f"[llm] call failed ({attempt}/{retries}): {exc}")
            if attempt == retries:
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


_FACTOR_SPEC_SYSTEM_PATH = (
    PROJECT_ROOT / "src" / "agent_market" / "factor_compiler" / "prompts" / "factor_spec.system.md"
)
_FACTOR_SPEC_FEWSHOT_PATH = (
    PROJECT_ROOT / "src" / "agent_market" / "factor_compiler" / "prompts" / "factor_spec.fewshot.json"
)


def load_factor_spec_prompt_assets() -> Dict[str, Any]:
    """
    Load FactorSpec prompt assets from `src/agent_market/factor_compiler/prompts/`.

    This function is offline-friendly and does not require any API key.
    """

    system_prompt = _FACTOR_SPEC_SYSTEM_PATH.read_text(encoding="utf-8")
    fewshot = json.loads(_FACTOR_SPEC_FEWSHOT_PATH.read_text(encoding="utf-8"))
    return {
        "system_prompt": system_prompt,
        "fewshot": fewshot,
        "paths": {
            "system_prompt": str(_FACTOR_SPEC_SYSTEM_PATH),
            "fewshot": str(_FACTOR_SPEC_FEWSHOT_PATH),
        },
    }


def extract_factor_spec_payload(raw_content: str) -> Dict[str, Any]:
    """
    Extract a JSON object payload that should validate as `FactorSpec`.
    """

    payload = _extract_json_object(raw_content)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("LLM response missing FactorSpec JSON object")
    return payload


def parse_factor_spec(raw_content: str) -> Any:
    """
    Parse and validate a FactorSpec from raw model output (or any JSON string).
    """

    from agent_market.factor_compiler import FactorSpec  # noqa: WPS433

    payload = extract_factor_spec_payload(raw_content)
    return FactorSpec.model_validate(payload)


def request_factor_spec_completion(prompt: str, config: LLMConfig) -> Tuple[str, Optional[Dict[str, Any]]]:
    assets = load_factor_spec_prompt_assets()
    return request_completion(prompt, config, system_prompt=str(assets.get("system_prompt") or ""))
