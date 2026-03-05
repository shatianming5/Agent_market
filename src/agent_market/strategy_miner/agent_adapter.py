"""Provider-agnostic agent adapter for strategy mining.

Primary provider: OpenCode client (tool-calling loop).
Fallbacks:
- OpenAI-compatible chat completion (no tools)

The adapter is intentionally lightweight: phases own the state machine.

Recovery hardening:
- Robust code extraction from tool-tag / markdown outputs.
- Provider fallback chain: opencode -> openai-compatible (glm-4-flash).
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
)

logger = logging.getLogger(__name__)


def _extract_opencode_write_blocks(text: str) -> list[str]:
    """Extract bodies inside `<write ...>...</write>` blocks."""
    if not isinstance(text, str) or not text:
        return []

    s = text
    lower = s.lower()
    out: list[str] = []
    pos = 0
    while True:
        i = lower.find("<write", pos)
        if i < 0:
            break
        j = lower.find(">", i)
        if j < 0:
            break
        k = lower.find("</write>", j)
        if k < 0:
            break
        body = s[j + 1 : k]
        body = body.strip("\n")
        if body.strip():
            out.append(body)
        pos = k + len("</write>")

    return out


def _strip_opencode_tool_lines(text: str) -> str:
    """Remove common OpenCode tool-tag lines from *text* (best-effort)."""
    if not isinstance(text, str) or not text:
        return ""

    cleaned: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            cleaned.append(line)
            continue
        # Common tool tags
        if s.startswith("<read") and s.endswith("/>"):
            continue
        if s.startswith("<bash") and s.endswith("/>"):
            continue
        if s.startswith("<edit") and s.endswith("/>"):
            continue
        if s.startswith("<write"):
            continue
        if s.startswith("</write"):
            continue
        if s.startswith("<final") or s.startswith("</final"):
            continue
        if s.startswith("<tool") or s.startswith("</tool"):
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def _extract_code_block(text: str) -> str | None:
    """Extract a python strategy body from LLM output.

    Supports:
    - OpenCode `<write ...>...</write>` tool blocks
    - Markdown fences ```python ...```
    - Raw python (with tool-tag lines stripped)
    """

    if not isinstance(text, str) or not text.strip():
        return None

    # 1) Prefer explicit tool-write blocks.
    write_blocks = _extract_opencode_write_blocks(text)
    if write_blocks:
        best = max(write_blocks, key=len).strip()
        best = _strip_opencode_tool_lines(best)
        if best:
            return best

    # 2) Markdown code fences.
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
        return _strip_opencode_tool_lines(best)

    # 3) Raw python fallback (strip obvious tool lines first).
    stripped = _strip_opencode_tool_lines(s)
    if "class" in stripped and "IStrategy" in stripped and "populate_entry_trend" in stripped:
        return stripped.strip()

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
        openai_fallback_model: str = "glm-4-flash",
    ) -> None:
        self._workspace = Path(workspace)
        self._provider = (provider or "auto").strip().lower()
        self._max_retries = max(0, int(max_retries))
        self._openai_fallback_model = str(openai_fallback_model or "").strip() or "glm-4-flash"
        self._closed = False

        self._executor: AgentExecutor
        self._executor_info: dict[str, Any] = {}

        if self._provider == "template":
            raise ValueError(
                "provider=template is disabled for strategy_miner (no-template enforced). "
                "Use provider=opencode or provider=openai_compatible."
            )

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
            openai_exec = self._build_openai_executor()
            if openai_exec is not None:
                self._executor = openai_exec
                self._executor_info = {"provider": "openai_compatible"}
                logger.info("StrategyAgent provider=openai_compatible workspace=%s", self._workspace)
                return

            if self._provider in ("openai", "openai_compatible"):
                raise ValueError(
                    "OpenAI-compatible provider requested but missing credentials. "
                    "Set LLM_API_KEY (or OPENAI_API_KEY)."
                )
        raise ValueError(
            "No usable LLM provider available for strategy_miner (no-template enforced). "
            "Provide an OpenCode model (MinerConfig.model / OPENCODE_MODEL) "
            "or set OpenAI-compatible credentials (LLM_API_KEY / OPENAI_API_KEY)."
        )

    def _build_openai_executor(self) -> OpenAIChatExecutor | None:
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        if not api_key.strip():
            return None

        llm_base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or ""
        llm_model = os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or ""
        model = (llm_model or self._openai_fallback_model).strip() or self._openai_fallback_model

        try:
            return OpenAIChatExecutor(
                base_url=llm_base_url or "https://api.openai.com/v1",
                api_key=api_key,
                model=model,
                system_prompt=(
                    "You are a senior quantitative strategy engineer. "
                    "Reply with concise, correct output."
                ),
                retries=self._max_retries,
                timeout_seconds=60,
            )
        except Exception as exc:
            logger.warning("OpenAI-compatible fallback unavailable: %s", exc)
            return None

    def run_result(
        self,
        prompt: str,
        on_turn: Optional[Callable[[Any], None]] = None,
    ) -> AgentRunResult:
        """Run the underlying provider once and return a rich result."""
        if self._closed:
            raise RuntimeError("StrategyAgent is already closed")
        try:
            result: AgentRunResult = self._executor.run(prompt, on_turn=on_turn)
            return result
        except Exception as exc:
            # Graceful fallback when OpenCode is selected but unavailable at runtime
            # (e.g. server crash, stale timeout).
            if self._provider == "opencode":
                raise
            if not isinstance(self._executor, OpenCodeExecutor):
                raise
            logger.warning("OpenCode run failed, falling back: %s", exc)

            openai_exec = self._build_openai_executor()
            if openai_exec is not None:
                try:
                    self._executor = openai_exec
                    result2: AgentRunResult = self._executor.run(prompt, on_turn=on_turn)
                    return result2
                except Exception as exc2:
                    logger.warning("OpenAI-compatible fallback failed: %s", exc2)

            raise RuntimeError(f"StrategyAgent.run failed: {exc}") from exc

    def run(
        self,
        prompt: str,
        on_turn: Optional[Callable[[Any], None]] = None,
    ) -> str:
        """Run the underlying provider once."""
        return self.run_result(prompt, on_turn=on_turn).assistant_text

    def generate_strategy(
        self,
        prompt: str,
        *,
        on_turn: Optional[Callable[[Any], None]] = None,
        on_result: Optional[Callable[[AgentRunResult], None]] = None,
        filename_hint: str | None = None,
    ) -> Path | None:
        """Generate a strategy and ensure a .py file is created in the sandbox.

        If the active provider cannot produce a usable artifact, fall back in order:
        opencode -> openai-compatible.
        """
        if self._closed:
            raise RuntimeError("StrategyAgent is already closed")
        strategies_dir = self._workspace / "user_data" / "strategies"
        strategies_dir.mkdir(parents=True, exist_ok=True)

        def _snapshot_mtime() -> dict[Path, float]:
            out: dict[Path, float] = {}
            for p in strategies_dir.glob("*.py"):
                if not p.is_file():
                    continue
                if p.name.startswith("_") or "reference" in p.name.lower():
                    continue
                try:
                    out[p] = p.stat().st_mtime
                except Exception:
                    continue
            return out

        def _list_candidates() -> list[Path]:
            return sorted(
                [
                    p
                    for p in strategies_dir.glob("*.py")
                    if p.is_file()
                    and not p.name.startswith("_")
                    and "reference" not in p.name.lower()
                ],
                key=lambda p: p.stat().st_mtime,
            )

        attempted_openai = False

        for _ in range(3):
            before_mtime = _snapshot_mtime()
            try:
                result = self.run_result(prompt, on_turn=on_turn)
                if on_result is not None:
                    on_result(result)
                text = result.assistant_text
            except Exception as exc:
                logger.warning("StrategyAgent.run failed: %s", exc)
                text = ""

            candidates = _list_candidates()

            # If tool-capable provider wrote files, prefer newest.
            changed = False
            for p in candidates:
                try:
                    mt = p.stat().st_mtime
                except Exception:
                    continue
                if p not in before_mtime or before_mtime.get(p) != mt:
                    changed = True
                    break

            if candidates and changed:
                return candidates[-1]

            code = _extract_code_block(text)
            if code:
                class_name = _infer_strategy_class_name(code) or "MinedStrategy"
                file_name = filename_hint or f"{class_name}.py"
                out_path = strategies_dir / file_name
                out_path.write_text(code, encoding="utf-8")
                return out_path

            # No usable artifact produced: fall back provider chain.
            if isinstance(self._executor, OpenCodeExecutor) and not attempted_openai:
                openai_exec = self._build_openai_executor()
                attempted_openai = True
                if openai_exec is not None:
                    self._executor = openai_exec
                    self._executor_info = {"provider": "openai_compatible"}
                    continue

            break

        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._executor.close()
        except Exception:
            logger.debug("StrategyAgent.close failed", exc_info=True)
