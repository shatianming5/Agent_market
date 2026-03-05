"""Provider-agnostic agent adapter for strategy mining.

Primary provider: OpenCode client (tool-calling loop).
Fallbacks:
- OpenAI-compatible chat completion (no tools)
- Deterministic template strategy (no external deps)

The adapter is intentionally lightweight: phases own the state machine.
"""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

from agent_market.agents.executor import (
    AgentExecutor,
    AgentRunResult,
    OpenAIChatExecutor,
    OpenCodeExecutor,
    TemplateExecutor,
)

logger = logging.getLogger(__name__)


def _extract_code_block(text: str) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None

    fence = "```"
    s = text
    best: str | None = None
    pos = 0
    while True:
        i = s.find(fence, pos)
        if i < 0:
            break
        j = s.find(fence, i + len(fence))
        if j < 0:
            break

        header = s[i + len(fence) : s.find("\n", i + len(fence))]
        body_start = s.find("\n", i + len(fence))
        if body_start < 0:
            break
        body = s[body_start + 1 : j]
        lang = (header or "").strip().lower()
        if not lang or "python" in lang or "py" == lang:
            cand = body.strip()
            if cand and (best is None or len(cand) > len(best)):
                best = cand
        pos = j + len(fence)

    if best:
        return best

    # Fallback: if it looks like raw Python, accept whole text.
    if "class" in s and "IStrategy" in s and "populate_entry_trend" in s:
        return s.strip()

    return None


def _infer_strategy_class_name(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if base_name == "IStrategy":
                return node.name
    return None


_TEMPLATE_STRATEGY = """\
from freqtrade.strategy import IStrategy


class TemplateRsiStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.02}
    stoploss = -0.10

    def populate_indicators(self, dataframe, metadata):
        # Minimal indicator set to keep template dependency-light.
        try:
            import talib.abstract as ta

            dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        except Exception:
            dataframe["rsi"] = 50
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe.loc[dataframe["rsi"] < 30, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe.loc[dataframe["rsi"] > 70, "exit_long"] = 1
        return dataframe
"""


class StrategyAgent:
    """Strategy-miner agent wrapper with graceful fallback."""

    def __init__(
        self,
        workspace: Path,
        model: str = "",
        base_url: Optional[str] = None,
        max_turns: int = 30,
        stale_timeout: float = 180.0,
        max_retries: int = 2,
        provider: str = "auto",
        tool_policy: Any | None = None,
    ) -> None:
        self._workspace = Path(workspace)
        self._provider = (provider or "auto").strip().lower()
        self._max_retries = max(0, int(max_retries))
        self._closed = False

        self._executor: AgentExecutor
        self._executor_info: dict[str, Any] = {}

        # 1) Prefer OpenCode (tool loop) if possible.
        if self._provider in ("auto", "opencode"):
            try:
                self._executor = OpenCodeExecutor(
                    repo=self._workspace,
                    model=model,
                    base_url=base_url,
                    max_turns=max_turns,
                    stale_timeout=stale_timeout,
                    max_retries=max_retries,
                    tool_policy=tool_policy,
                    permission_overrides={"external_directory": {"*": "allow"}},
                )
                self._executor_info = {"provider": "opencode"}
                logger.info("StrategyAgent provider=opencode workspace=%s", self._workspace)
                return
            except Exception as exc:
                if self._provider == "opencode":
                    raise
                logger.warning("OpenCode unavailable, falling back: %s", exc)

        # 2) Fallback: OpenAI-compatible chat completion.
        if self._provider in ("auto", "openai", "openai_compatible"):
            api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
            llm_base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or ""
            llm_model = os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or ""
            if api_key.strip() and llm_model.strip():
                try:
                    self._executor = OpenAIChatExecutor(
                        base_url=llm_base_url or "https://api.openai.com/v1",
                        api_key=api_key,
                        model=llm_model,
                        system_prompt=(
                            "You are a senior quantitative strategy engineer. "
                            "Reply with concise, correct output."
                        ),
                        retries=max_retries,
                        timeout_seconds=60,
                    )
                    self._executor_info = {"provider": "openai_compatible"}
                    logger.info("StrategyAgent provider=openai_compatible workspace=%s", self._workspace)
                    return
                except Exception as exc:
                    if self._provider in ("openai", "openai_compatible"):
                        raise
                    logger.warning("OpenAI-compatible fallback unavailable: %s", exc)

        # 3) Final fallback: deterministic template.
        self._executor = TemplateExecutor(text=_TEMPLATE_STRATEGY)
        self._executor_info = {"provider": "template"}
        logger.info("StrategyAgent provider=template workspace=%s", self._workspace)

    def run(
        self,
        prompt: str,
        on_turn: Optional[Callable[[Any], None]] = None,
    ) -> str:
        """Run the underlying provider once.

        Note: for non-tooling providers, this does **not** automatically write files.
        Use ``generate_strategy`` for strategy generation.
        """
        if self._closed:
            raise RuntimeError("StrategyAgent is already closed")
        try:
            result: AgentRunResult = self._executor.run(prompt, on_turn=on_turn)
            return result.assistant_text
        except Exception as exc:
            # Graceful fallback when OpenCode is selected but unavailable at runtime
            # (e.g. `opencode` missing from PATH, server crash, stale timeout).
            if self._provider == "opencode":
                raise
            if not isinstance(self._executor, OpenCodeExecutor):
                raise
            logger.warning("OpenCode run failed, falling back: %s", exc)

            api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
            llm_base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or ""
            llm_model = os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or ""
            if api_key.strip() and llm_model.strip():
                try:
                    self._executor = OpenAIChatExecutor(
                        base_url=llm_base_url or "https://api.openai.com/v1",
                        api_key=api_key,
                        model=llm_model,
                        system_prompt=(
                            "You are a senior quantitative strategy engineer. "
                            "Reply with concise, correct output."
                        ),
                        retries=self._max_retries,
                        timeout_seconds=60,
                    )
                    result2: AgentRunResult = self._executor.run(prompt, on_turn=on_turn)
                    return result2.assistant_text
                except Exception as exc2:
                    logger.warning("OpenAI-compatible fallback failed: %s", exc2)

            self._executor = TemplateExecutor(text=_TEMPLATE_STRATEGY)
            result3: AgentRunResult = self._executor.run(prompt, on_turn=on_turn)
            return result3.assistant_text

    def generate_strategy(
        self,
        prompt: str,
        *,
        on_turn: Optional[Callable[[Any], None]] = None,
        filename_hint: str | None = None,
    ) -> Path | None:
        """Generate a strategy and ensure a .py file is created in the sandbox."""
        if self._closed:
            raise RuntimeError("StrategyAgent is already closed")
        strategies_dir = self._workspace / "user_data" / "strategies"
        strategies_dir.mkdir(parents=True, exist_ok=True)

        before_mtime = {
            p: p.stat().st_mtime
            for p in strategies_dir.glob("*.py")
            if p.is_file()
            and not p.name.startswith("_")
            and "reference" not in p.name.lower()
        }

        text = self.run(prompt, on_turn=on_turn)

        candidates = sorted(
            [
                p
                for p in strategies_dir.glob("*.py")
                if p.is_file()
                and not p.name.startswith("_")
                and "reference" not in p.name.lower()
            ],
            key=lambda p: p.stat().st_mtime,
        )

        changed = False
        for p in candidates:
            try:
                mt = p.stat().st_mtime
            except Exception:
                continue
            if p not in before_mtime or before_mtime.get(p) != mt:
                changed = True
                break

        # If tool-capable provider wrote files, prefer newest.
        if candidates and changed:
            return candidates[-1]

        code = _extract_code_block(text)
        if code:
            class_name = _infer_strategy_class_name(code) or "MinedStrategy"
            file_name = filename_hint or f"{class_name}.py"
            out_path = strategies_dir / file_name
            out_path.write_text(code, encoding="utf-8")
            return out_path

        # If no code was provided, fall back to whatever is present.
        if candidates:
            return candidates[-1]

        out_path = strategies_dir / (filename_hint or "MinedStrategy.py")
        out_path.write_text(_TEMPLATE_STRATEGY, encoding="utf-8")
        return out_path

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._executor.close()
        except Exception:
            logger.debug("StrategyAgent.close failed", exc_info=True)
