from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Protocol
from urllib.parse import urlparse

import requests
from filelock import FileLock, Timeout as FileLockTimeout
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentRunResult:
    assistant_text: str
    provider: str
    model: str | None = None
    tool_trace: list[dict[str, Any]] | None = None
    raw: Any | None = None
    # D13: Token usage tracking
    usage: dict[str, int] | None = None


@dataclass(frozen=True)
class _BufferedResponse:
    status_code: int
    headers: dict[str, str]
    text: str

    def json(self) -> Any:
        return json.loads(self.text)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return value if value > 0 else float(default)


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except Exception:
        return int(default)
    return value if value >= 0 else int(default)


@contextmanager
def _llm_request_slot(max_concurrent: int, wait_timeout: float) -> Iterator[None]:
    """Best-effort cross-process semaphore for outbound LLM requests."""
    if max_concurrent <= 0:
        yield
        return

    slot_dir = Path(tempfile.gettempdir()) / "agent_market_llm_slots"
    slot_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(1.0, float(wait_timeout))
    lock: FileLock | None = None
    while time.monotonic() < deadline:
        for idx in range(max_concurrent):
            candidate = FileLock(str(slot_dir / f"slot_{idx:02d}.lock"))
            try:
                candidate.acquire(timeout=0.01)
                lock = candidate
                break
            except FileLockTimeout:
                continue
        if lock is not None:
            break
        time.sleep(0.25)

    if lock is None:
        raise RuntimeError(f"llm_slot_wait_timeout_{max_concurrent}")

    try:
        yield
    finally:
        try:
            lock.release()
        except Exception:
            logger.debug("Failed to release LLM slot lock", exc_info=True)


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
        usage = self._extract_usage_from_tool_trace(result.tool_trace)
        return AgentRunResult(
            assistant_text=result.assistant_text,
            provider=self._provider,
            model=self._model,
            tool_trace=list(result.tool_trace or []),
            raw=result.raw,
            usage=usage,
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            logger.debug("OpenCodeExecutor.close failed", exc_info=True)

    @staticmethod
    def _extract_usage_from_tool_trace(tool_trace: list[dict[str, Any]] | None) -> dict[str, Any] | None:
        """Aggregate token/cost metadata emitted by OpenCode turns."""

        if not tool_trace:
            return None

        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = 0
        cache_read_tokens = 0
        cache_write_tokens = 0
        total_tokens = 0
        cost_usd = 0.0
        seen = False

        for entry in tool_trace:
            if not isinstance(entry, dict):
                continue
            tokens = entry.get("tokens")
            if not isinstance(tokens, dict):
                continue
            seen = True
            try:
                _prompt = int(tokens.get("input") or 0)
            except Exception:
                _prompt = 0
            try:
                _completion = int(tokens.get("output") or 0)
            except Exception:
                _completion = 0
            try:
                _reasoning = int(tokens.get("reasoning") or 0)
            except Exception:
                _reasoning = 0
            try:
                _cache_read = int(tokens.get("cache_read") or 0)
            except Exception:
                _cache_read = 0
            try:
                _cache_write = int(tokens.get("cache_write") or 0)
            except Exception:
                _cache_write = 0
            try:
                _total = int(tokens.get("total") or 0)
            except Exception:
                _total = 0
            try:
                _cost = float(tokens.get("cost") or 0.0)
            except Exception:
                _cost = 0.0

            if _total <= 0:
                _total = _prompt + _completion + _reasoning

            prompt_tokens += _prompt
            completion_tokens += _completion
            reasoning_tokens += _reasoning
            cache_read_tokens += _cache_read
            cache_write_tokens += _cache_write
            total_tokens += _total
            cost_usd += _cost

        if not seen:
            return None

        return {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "cost_usd": round(cost_usd, 6),
        }


class OpenAIChatExecutor:
    def __init__(
        self,
        *,
        base_url: str = "",
        base_urls: list[str] | None = None,
        api_key: str,
        model: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout_seconds: float = 60.0,
        retries: int = 2,
    ) -> None:
        self._provider = "openai_compatible"
        # Multi-URL support: base_urls takes priority over base_url
        if base_urls:
            self._base_urls = [u.strip() for u in base_urls if u.strip()]
        else:
            self._base_urls = [str(base_url or "").strip() or "https://api.openai.com/v1"]
        self._url_index = 0
        self._api_key = str(api_key or "").strip()
        self._model = str(model or "").strip()
        self._system_prompt = system_prompt
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._timeout = _env_float("AGENT_MARKET_LLM_REQUEST_TIMEOUT", float(timeout_seconds))
        self._retries = max(0, int(retries))
        self._max_concurrent = _env_int("AGENT_MARKET_LLM_MAX_CONCURRENT", 0)
        self._slot_wait_timeout = _env_float(
            "AGENT_MARKET_LLM_SLOT_WAIT_SECONDS",
            max(300.0, self._timeout * 6.0),
        )

        if not self._api_key:
            raise ValueError("OpenAI-compatible api_key is required.")
        if not self._model:
            raise ValueError("OpenAI-compatible model is required.")

        if len(self._base_urls) > 1:
            logger.info("OpenAIChatExecutor multi-URL mode: %d endpoints", len(self._base_urls))

    @staticmethod
    def _normalize_base_url(raw: str) -> str:
        base_url = raw.rstrip("/")
        parsed = urlparse(base_url)
        path = (parsed.path or "").rstrip("/")
        if path.endswith("/v1") or "/v1/" in (path + "/"):
            return base_url
        if path.endswith("/v4") or "/v4/" in (path + "/"):
            return base_url
        if "/api/paas/v4" in path:
            return base_url
        return base_url + "/v1"

    def _post(self, url: str, payload: dict[str, Any]) -> _BufferedResponse:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Connection": "close",
        }
        connect_timeout = min(10.0, max(1.0, self._timeout))
        read_timeout = max(10.0, self._timeout)
        with _llm_request_slot(self._max_concurrent, self._slot_wait_timeout):
            started = time.monotonic()
            resp = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=(connect_timeout, read_timeout),
                stream=True,
            )
            try:
                chunks: list[bytes] = []
                for chunk in resp.iter_content(chunk_size=65536):
                    if time.monotonic() - started > self._timeout:
                        raise RuntimeError(f"llm_wall_timeout_{int(self._timeout)}")
                    if chunk:
                        chunks.append(chunk)
                if time.monotonic() - started > self._timeout:
                    raise RuntimeError(f"llm_wall_timeout_{int(self._timeout)}")
                encoding = resp.encoding or "utf-8"
                body = b"".join(chunks).decode(encoding, errors="replace")
                return _BufferedResponse(
                    status_code=int(resp.status_code),
                    headers={str(k): str(v) for k, v in resp.headers.items()},
                    text=body,
                )
            finally:
                resp.close()

    def _pick_url(self) -> str:
        """Return the next base URL in round-robin order."""
        url = self._base_urls[self._url_index % len(self._base_urls)]
        return self._normalize_base_url(url)

    def _advance_url(self) -> None:
        """Rotate to the next base URL."""
        if len(self._base_urls) > 1:
            self._url_index = (self._url_index + 1) % len(self._base_urls)

    def run(self, prompt: str, *, on_turn: Optional[Callable[[Any], None]] = None) -> AgentRunResult:
        _ = on_turn
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        n_urls = len(self._base_urls)
        # Outer rounds: try all endpoints per round, up to 6 rounds (wait between rounds)
        max_rounds = max(3, self._retries + 1)
        last_err: Optional[BaseException] = None
        for rnd in range(max_rounds):
            for idx in range(n_urls):
                base_url = self._pick_url()
                url = base_url + "/chat/completions"
                try:
                    resp = self._post(url, payload)
                    if resp.status_code >= 400:
                        retryable = resp.status_code in (429, 500, 502, 503, 504)
                        if retryable:
                            self._advance_url()
                            if resp.status_code == 429:
                                ra = resp.headers.get("Retry-After")
                                sleep_s = 2.0
                                if ra:
                                    try:
                                        sleep_s = max(sleep_s, float(ra))
                                    except Exception:
                                        pass
                                time.sleep(min(30.0, sleep_s))
                            if rnd == 0 and idx < 3:
                                logger.warning(
                                    "HTTP %d from %s, trying next endpoint (round %d, %d/%d)",
                                    resp.status_code, base_url, rnd + 1, idx + 1, n_urls,
                                )
                        raise RuntimeError(f"llm_http_{resp.status_code}: {resp.text[:200]}")
                    data = resp.json()
                    choice0 = (data.get("choices") or [{}])[0]
                    msg = choice0.get("message") or {}
                    content = msg.get("content")
                    if not isinstance(content, str):
                        content = "" if content is None else str(content)
                    if not content.strip():
                        raise RuntimeError("llm_empty_response")
                    # D13: Extract token usage from response
                    _usage = None
                    if isinstance(data.get("usage"), dict):
                        _usage = {
                            k: int(v) for k, v in data["usage"].items()
                            if isinstance(v, (int, float))
                        }
                    return AgentRunResult(
                        assistant_text=content,
                        provider=self._provider,
                        model=self._model,
                        tool_trace=None,
                        raw=data,
                        usage=_usage,
                    )
                except (RequestException, RuntimeError, ValueError) as exc:
                    last_err = exc
                    if isinstance(exc, RequestException) or "llm_wall_timeout" in str(exc) or "llm_slot_wait_timeout" in str(exc):
                        self._advance_url()
                    continue
            # All endpoints failed this round — wait before next round
            if rnd < max_rounds - 1:
                wait_s = min(30, 10 * (rnd + 1))
                logger.warning(
                    "All %d endpoints failed (round %d/%d, last_err=%s). Waiting %ds before retry...",
                    n_urls, rnd + 1, max_rounds, last_err, wait_s,
                )
                time.sleep(wait_s)

        raise RuntimeError(
            f"OpenAIChatExecutor failed after {max_rounds} rounds x {n_urls} endpoints: {last_err}"
        )

    def close(self) -> None:
        return
