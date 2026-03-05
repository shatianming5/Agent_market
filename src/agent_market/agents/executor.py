from __future__ import annotations

import json
import hashlib
import re
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol
from urllib.parse import urlparse

import requests
from requests import Response
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentRunResult:
    assistant_text: str
    provider: str
    model: str | None = None
    tool_trace: list[dict[str, Any]] | None = None
    raw: Any | None = None


class AgentExecutor(Protocol):
    def run(
        self,
        prompt: str,
        *,
        on_turn: Optional[Callable[[Any], None]] = None,
    ) -> AgentRunResult:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class OpenCodeExecutor:
    def __init__(
        self,
        *,
        repo: Path,
        model: str,
        base_url: Optional[str] = None,
        max_turns: int = 30,
        stale_timeout: float = 180.0,
        max_retries: int = 2,
        timeout_seconds: int = 600,
        unattended: str = "strict",
        auto_compact: bool = True,
        context_length: int = 128_000,
        tool_policy: Any | None = None,
        permission_overrides: Optional[dict[str, Any]] = None,
    ) -> None:
        from runner_fsm.opencode.client import OpenCodeClient  # noqa: WPS433

        resolved_model = (model or os.environ.get("OPENCODE_MODEL", "")).strip()
        if not resolved_model:
            raise ValueError("OpenCode model is required (model= or OPENCODE_MODEL).")

        resolved_url = (base_url or os.environ.get("OPENCODE_URL") or "").strip() or None
        self._provider = "opencode"
        self._model = resolved_model

        self._client: OpenCodeClient = OpenCodeClient(
            repo=repo,
            model=resolved_model,
            base_url=resolved_url,
            timeout_seconds=timeout_seconds,
            unattended=unattended,
            max_turns=max_turns,
            auto_compact=auto_compact,
            context_length=context_length,
            stale_timeout=stale_timeout,
            request_retry_attempts=max(0, int(max_retries)),
            session_recover_attempts=max(0, int(max_retries)),
            tool_policy=tool_policy,
            permission_overrides=permission_overrides or {},
        )

    def run(
        self,
        prompt: str,
        *,
        on_turn: Optional[Callable[[Any], None]] = None,
    ) -> AgentRunResult:
        from runner_fsm.dtypes import AgentResult  # noqa: WPS433

        result: AgentResult = self._client.run(prompt, on_turn=on_turn)
        return AgentRunResult(
            assistant_text=result.assistant_text,
            provider=self._provider,
            model=self._model,
            tool_trace=list(result.tool_trace or []),
            raw=result.raw,
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            logger.debug("OpenCodeExecutor.close failed", exc_info=True)


class OpenCodeCliExecutor:
    """OpenCode executor backed by `opencode run` (CLI, no local server).

    This is more robust in environments where `opencode serve` may hang or be
    blocked, at the cost of tool-loop capabilities.
    """

    def __init__(
        self,
        *,
        repo: Path,
        model: str,
        max_retries: int = 2,
        timeout_seconds: int = 600,
        log_level: str = "ERROR",
    ) -> None:
        resolved_model = (model or os.environ.get("OPENCODE_MODEL", "")).strip()
        if not resolved_model:
            raise ValueError("OpenCode model is required (model= or OPENCODE_MODEL).")

        self._provider = "opencode_cli"
        self._model = resolved_model
        self._repo = Path(repo)
        self._max_retries = max(0, int(max_retries or 0))
        self._timeout_seconds = int(timeout_seconds or 600)
        self._cmd = [
            "opencode",
            "run",
            "-m",
            resolved_model,
            "--log-level",
            str(log_level or "ERROR"),
        ]

    def run(
        self,
        prompt: str,
        *,
        on_turn: Optional[Callable[[Any], None]] = None,
    ) -> AgentRunResult:
        _ = on_turn  # opencode CLI does not stream turn events here
        last_err: Optional[BaseException] = None

        for attempt in range(self._max_retries + 1):
            try:
                proc = subprocess.run(
                    self._cmd,
                    input=str(prompt or ""),
                    text=True,
                    capture_output=True,
                    cwd=str(self._repo),
                    timeout=self._timeout_seconds,
                    check=False,
                )
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""

                if proc.returncode != 0:
                    tail = (stderr or stdout).strip()[-400:]
                    raise RuntimeError(f"opencode_run_failed rc={proc.returncode} tail={tail}")

                assistant = stdout.strip() or stderr.strip()
                if not assistant:
                    raise RuntimeError("opencode_run_empty_output")

                return AgentRunResult(
                    assistant_text=assistant,
                    provider=self._provider,
                    model=self._model,
                    tool_trace=None,
                    raw={
                        "returncode": proc.returncode,
                        "stdout": stdout,
                        "stderr": stderr,
                        "cmd": list(self._cmd),
                    },
                )
            except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
                last_err = exc
                if attempt >= self._max_retries:
                    break
                time.sleep(0.5 * (attempt + 1))

        raise RuntimeError(f"OpenCodeCliExecutor failed: {last_err}") from last_err

    def close(self) -> None:
        return


class HeuristicExecutor:
    """Local heuristic executor (no external LLM calls).

    Deterministically generates strategy candidates and JSON role outputs
    (planner/reviewer/backtester) to keep the miner runnable when external
    providers are unavailable (e.g. rate limited).
    """

    def __init__(self, *, repo: Path) -> None:
        self._provider = "heuristic"
        self._model = None
        self._repo = Path(repo)

    @staticmethod
    def _stable_seed(prompt: str) -> int:
        h = hashlib.sha256((prompt or "").encode("utf-8", errors="ignore")).hexdigest()
        return int(h[:8], 16)

    @staticmethod
    def _extract_iteration(prompt: str) -> int:
        m = re.search(r"##\s*Iteration\s+(\d+)", prompt or "")
        if not m:
            return 0
        try:
            return int(m.group(1))
        except Exception:
            return 0

    @staticmethod
    def _extract_candidate_idx(prompt: str) -> int:
        m = re.search(r"##\s*Candidate\s+(\d+)\s*/\s*(\d+)", prompt or "")
        if not m:
            return 0
        try:
            return max(0, int(m.group(1)) - 1)
        except Exception:
            return 0

    @staticmethod
    def _pick_family(seed: int, candidate_idx: int) -> str:
        families = ["EmaRsiTrend", "BollingerReversion", "MacdMomentum", "DonchianBreakout"]
        return families[(seed + int(candidate_idx)) % len(families)]

    @classmethod
    def _build_strategy_code(cls, *, prompt: str) -> str:
        seed = cls._stable_seed(prompt)
        iteration = cls._extract_iteration(prompt)
        candidate_idx = cls._extract_candidate_idx(prompt)
        family = cls._pick_family(seed, candidate_idx)

        class_name = f"Heuristic{family}I{iteration}C{candidate_idx}"

        # Candidate-specific parameters (kept moderate to generate trades).
        ema_fast = 10 + (seed % 6)
        ema_slow = 22 + (seed % 10)
        rsi_period = 14

        buy_rsi = 48 + (seed % 10)
        sell_rsi = 45 - (seed % 8)

        bb_period = 18 + (seed % 8)
        bb_std = 1.8 + ((seed % 5) * 0.2)

        dc_period = 18 + (seed % 10)

        minimal_roi = {"0": 0.03, "120": 0.01, "240": 0.0}
        stoploss = -0.06

        indicators = ""
        entry_logic = ""
        exit_logic = ""

        if family == "EmaRsiTrend":
            indicators = """
        df["ema_fast"] = _ema(df["close"], int(self.buy_ema_fast))
        df["ema_slow"] = _ema(df["close"], int(self.buy_ema_slow))
        df["rsi"] = _rsi(df["close"], int(self.buy_rsi_period))
"""
            entry_logic = """
        cross_up = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
        cond = (df["volume"] > 0) & cross_up & (df["rsi"] > float(self.buy_rsi))
        df.loc[cond, ["enter_long", "enter_tag"]] = (1, "ema_rsi")
"""
            exit_logic = """
        cross_dn = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))
        cond = (df["volume"] > 0) & (cross_dn | (df["rsi"] < float(self.sell_rsi)))
        df.loc[cond, "exit_long"] = 1
"""
        elif family == "BollingerReversion":
            indicators = """
        df["rsi"] = _rsi(df["close"], int(self.buy_rsi_period))
        mid, upper, lower = _bbands(df["close"], int(self.buy_bb_period), float(self.buy_bb_std))
        df["bb_mid"] = mid
        df["bb_upper"] = upper
        df["bb_lower"] = lower
"""
            entry_logic = """
        cond = (df["volume"] > 0) & (df["close"] < df["bb_lower"] * (1.0 - float(self.buy_bb_buffer))) & (df["rsi"] < float(self.buy_rsi))
        df.loc[cond, ["enter_long", "enter_tag"]] = (1, "bb_revert")
"""
            exit_logic = """
        cond = (df["volume"] > 0) & ((df["close"] > df["bb_mid"]) | (df["rsi"] > float(self.sell_rsi)))
        df.loc[cond, "exit_long"] = 1
"""
        elif family == "MacdMomentum":
            indicators = """
        df["rsi"] = _rsi(df["close"], int(self.buy_rsi_period))
        macd, macd_sig, hist = _macd(df["close"], int(self.buy_ema_fast), int(self.buy_ema_slow), 9)
        df["macd"] = macd
        df["macd_signal"] = macd_sig
        df["macd_hist"] = hist
"""
            entry_logic = """
        cross_up = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)
        cond = (df["volume"] > 0) & cross_up & (df["rsi"] > float(self.buy_rsi))
        df.loc[cond, ["enter_long", "enter_tag"]] = (1, "macd_momo")
"""
            exit_logic = """
        cross_dn = (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0)
        cond = (df["volume"] > 0) & (cross_dn | (df["rsi"] < float(self.sell_rsi)))
        df.loc[cond, "exit_long"] = 1
"""
        else:  # DonchianBreakout
            indicators = """
        df["rsi"] = _rsi(df["close"], int(self.buy_rsi_period))
        df["dc_high"] = df["high"].rolling(int(self.buy_dc_period)).max().shift(1)
        df["dc_low"] = df["low"].rolling(int(self.buy_dc_period)).min().shift(1)
"""
            entry_logic = """
        cond = (df["volume"] > 0) & (df["close"] > df["dc_high"]) & (df["rsi"] > float(self.buy_rsi))
        df.loc[cond, ["enter_long", "enter_tag"]] = (1, "donchian")
"""
            exit_logic = """
        cond = (df["volume"] > 0) & ((df["close"] < df["dc_low"]) | (df["rsi"] < float(self.sell_rsi)))
        df.loc[cond, "exit_long"] = 1
"""

        code = f"""from __future__ import annotations

import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy


def _ema(series, period: int):
    return series.ewm(span=int(period), adjust=False).mean()


def _rsi(series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / float(period), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / float(period), adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(0.0)


def _bbands(series, period: int = 20, stddev: float = 2.0):
    mid = series.rolling(int(period)).mean()
    std = series.rolling(int(period)).std()
    upper = mid + float(stddev) * std
    lower = mid - float(stddev) * std
    return mid, upper, lower


def _macd(series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=int(signal), adjust=False).mean()
    hist = macd - macd_signal
    return macd, macd_signal, hist


class {class_name}(IStrategy):
    timeframe = "1h"
    process_only_new_candles = True
    startup_candle_count: int = 60
    can_short = False
    use_exit_signal = True

    minimal_roi = {minimal_roi!r}
    stoploss = {stoploss}

    order_types = {{
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }}
    order_time_in_force = {{
        "entry": "GTC",
        "exit": "GTC",
    }}

    # Tunable params (named buy_/sell_ to enable evolve.py mutations)
    buy_ema_fast = {ema_fast}
    buy_ema_slow = {ema_slow}
    buy_rsi_period = {rsi_period}
    buy_rsi = {buy_rsi}
    sell_rsi = {sell_rsi}

    buy_bb_period = {bb_period}
    buy_bb_std = {bb_std:.2f}
    buy_bb_buffer = 0.002

    buy_dc_period = {dc_period}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
{indicators}
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
{entry_logic}
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
{exit_logic}
        return df
"""
        return code.strip() + "\n"

    def run(
        self,
        prompt: str,
        *,
        on_turn: Optional[Callable[[Any], None]] = None,
    ) -> AgentRunResult:
        _ = on_turn
        p = str(prompt or "")
        pl = p.lower()

        if "you are the planner agent" in pl:
            seed = self._stable_seed(p)
            cand = self._extract_candidate_idx(p)
            fam = self._pick_family(seed, cand)
            payload = {
                "name_hint": f"Heuristic{fam}",
                "timeframe": "1h",
                "indicators": ["EMA", "RSI"] + (["BBANDS"] if fam == "BollingerReversion" else []),
                "entry_rules": ["volume>0", "family-specific trigger"],
                "exit_rules": ["family-specific exit"],
                "risk_controls": ["stoploss", "roi"],
                "parameters": {"family": fam},
            }
            return AgentRunResult(
                assistant_text=json.dumps(payload, ensure_ascii=False),
                provider=self._provider,
                model=self._model,
                tool_trace=None,
                raw=payload,
            )

        if "you are the reviewer agent" in pl:
            payload = {"approved": True, "issues": [], "fixed_code": None}
            return AgentRunResult(
                assistant_text=json.dumps(payload, ensure_ascii=False),
                provider=self._provider,
                model=self._model,
                tool_trace=None,
                raw=payload,
            )

        if "you are the backtester agent" in pl:
            payload = {
                "quick_checks": ["python -m py_compile <strategy>.py"],
                "likely_failures": ["missing required methods", "order_types keys"],
                "risk_notes": ["ensure enough trades", "watch drawdown"],
            }
            return AgentRunResult(
                assistant_text=json.dumps(payload, ensure_ascii=False),
                provider=self._provider,
                model=self._model,
                tool_trace=None,
                raw=payload,
            )

        code = self._build_strategy_code(prompt=p)
        wrapped = f"```python\n{code}\n```"
        return AgentRunResult(
            assistant_text=wrapped,
            provider=self._provider,
            model=self._model,
            tool_trace=None,
            raw=None,
        )

    def close(self) -> None:
        return


class OpenAIChatExecutor:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout_seconds: float = 60.0,
        retries: int = 2,
    ) -> None:
        self._provider = "openai_compatible"
        self._base_url = str(base_url or "").strip() or "https://api.openai.com/v1"
        self._api_key = str(api_key or "").strip()
        self._model = str(model or "").strip()
        self._system_prompt = system_prompt
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._timeout = float(timeout_seconds)
        self._retries = max(0, int(retries))

        if not self._api_key:
            raise ValueError("OpenAI-compatible api_key is required.")
        if not self._model:
            raise ValueError("OpenAI-compatible model is required.")

    @staticmethod
    def _normalize_base_url(raw: str) -> str:
        """Normalize OpenAI-compatible base URL.

        Rules:
        - Keep explicit API version paths intact (e.g. /v1, /api/paas/v4).
        - Append /v1 only for plain host-style URLs.
        """
        base_url = raw.rstrip("/")
        parsed = urlparse(base_url)
        path = (parsed.path or "").rstrip("/")

        # Already versioned (OpenAI-compatible custom gateways, BigModel, etc.).
        if path.endswith("/v1") or "/v1/" in (path + "/"):
            return base_url
        if path.endswith("/v4") or "/v4/" in (path + "/"):
            return base_url
        if "/api/paas/v4" in path:
            return base_url

        # Default OpenAI-compatible convention.
        return base_url + "/v1"

    def _post(self, url: str, payload: dict[str, Any]) -> Response:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        return requests.post(
            url,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=self._timeout,
        )

    def run(self, prompt: str, *, on_turn: Optional[Callable[[Any], None]] = None) -> AgentRunResult:
        _ = on_turn  # unsupported in this executor
        base_url = self._normalize_base_url(self._base_url)
        url = base_url + "/chat/completions"

        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        last_err: Optional[BaseException] = None
        for attempt in range(self._retries + 1):
            try:
                resp = self._post(url, payload)
                if resp.status_code >= 400:
                    raise RuntimeError(f"llm_http_{resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                choice0 = (data.get("choices") or [{}])[0]
                msg = choice0.get("message") or {}
                content = msg.get("content")
                if not isinstance(content, str):
                    content = "" if content is None else str(content)
                if not content.strip():
                    raise RuntimeError("llm_empty_response")
                return AgentRunResult(
                    assistant_text=content,
                    provider=self._provider,
                    model=self._model,
                    tool_trace=None,
                    raw=data,
                )
            except (RequestException, RuntimeError, ValueError) as exc:
                last_err = exc
                if attempt >= self._retries:
                    break
                time.sleep(0.5 * (attempt + 1))

        raise RuntimeError(f"OpenAIChatExecutor failed: {last_err}")

    def close(self) -> None:
        return


class TemplateExecutor:
    """Deterministic no-LLM executor.

    This is intended as a graceful fallback when external dependencies (opencode/LLM)
    are unavailable.
    """

    def __init__(self, *, text: str = "") -> None:
        self._provider = "template"
        self._text = text

    def run(self, prompt: str, *, on_turn: Optional[Callable[[Any], None]] = None) -> AgentRunResult:
        _ = prompt
        _ = on_turn
        return AgentRunResult(
            assistant_text=self._text or "(template_executor)",
            provider=self._provider,
            model=None,
            tool_trace=None,
            raw=None,
        )

    def close(self) -> None:
        return
