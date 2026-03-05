"""Adapter wrapping OpenCodeClient for strategy mining."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

from runner_fsm.dtypes import AgentResult, TurnEvent
from runner_fsm.opencode.client import OpenCodeClient

logger = logging.getLogger(__name__)


class StrategyAgent:
    """Thin wrapper around OpenCodeClient tailored for strategy generation."""

    def __init__(
        self,
        workspace: Path,
        model: str = "",
        base_url: Optional[str] = None,
        max_turns: int = 30,
        stale_timeout: float = 180.0,
        max_retries: int = 2,
    ) -> None:
        resolved_model = model or os.environ.get("OPENCODE_MODEL", "")
        if not resolved_model:
            raise ValueError(
                "No LLM model specified. Set model= or OPENCODE_MODEL env var."
            )
        resolved_url = base_url or os.environ.get("OPENCODE_URL") or None

        self._max_retries = max(0, max_retries)
        self._client: Optional[OpenCodeClient] = OpenCodeClient(
            repo=workspace,
            model=resolved_model,
            base_url=resolved_url,
            timeout_seconds=600,
            unattended="strict",
            max_turns=max_turns,
            auto_compact=True,
            context_length=128000,
            stale_timeout=stale_timeout,
            request_retry_attempts=self._max_retries,
            session_recover_attempts=self._max_retries,
            permission_overrides={"external_directory": {"*": "allow"}},
        )
        logger.info(
            "StrategyAgent initialized: model=%s workspace=%s max_retries=%d",
            resolved_model,
            workspace,
            self._max_retries,
        )

    def run(
        self,
        prompt: str,
        on_turn: Optional[Callable[[TurnEvent], None]] = None,
    ) -> str:
        if self._client is None:
            raise RuntimeError("StrategyAgent is already closed")
        result: AgentResult = self._client.run(prompt, on_turn=on_turn)
        return result.assistant_text

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.debug("Error closing OpenCodeClient", exc_info=True)
            finally:
                self._client = None
