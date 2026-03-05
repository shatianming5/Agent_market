from __future__ import annotations

import json
import logging
import os
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
        base_url = raw.rstrip("/")
        parsed = urlparse(base_url)
        path = (parsed.path or "").rstrip("/")
        if not (path.endswith("/v1") or "/v1" in path.split("/")):
            base_url = base_url + "/v1"
        return base_url

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
                    content = str(content)
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
