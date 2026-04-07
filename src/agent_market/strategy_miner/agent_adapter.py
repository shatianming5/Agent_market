"""Provider-agnostic agent adapter for strategy mining.

Providers:
- OpenCode client (tool-calling loop)
- OpenAI-compatible chat completion (no tools)

The adapter is intentionally lightweight: phases own the state machine.

Recovery hardening:
- Robust code extraction from tool-tag / markdown outputs.
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

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_dotenv_fallback(path: Path) -> None:
    """Load simple KEY=VALUE pairs without overriding existing env."""
    try:
        if not path.exists():
            return
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)
    except Exception:
        logger.debug("Failed to load .env from %s", path, exc_info=True)


_load_dotenv_fallback(_PROJECT_ROOT / ".env")


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
    """Strategy-miner agent wrapper (OpenCode or OpenAI-compatible)."""

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

        # 1) OpenAI-compatible chat completion
        if self._provider in ("openai", "openai_compatible"):
            self._executor = self._build_openai_executor(model, base_url)
            self._executor_info = {"provider": "openai_compatible"}
            logger.info("StrategyAgent provider=openai_compatible workspace=%s", self._workspace)
            return

        # 2) OpenCode (tool loop) — explicit or auto
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
                    permission_overrides={"external_directory": {str(self._workspace): "allow"}},
                )
                self._executor_info = {"provider": "opencode"}
                logger.info("StrategyAgent provider=opencode workspace=%s", self._workspace)
                return
            except Exception as exc:
                if self._provider == "opencode":
                    raise
                logger.warning("OpenCode unavailable, trying openai_compatible: %s", exc)

        # 3) Auto fallback to OpenAI-compatible
        if self._provider == "auto":
            try:
                self._executor = self._build_openai_executor(model, base_url)
                self._executor_info = {"provider": "openai_compatible"}
                logger.info("StrategyAgent provider=openai_compatible (fallback) workspace=%s", self._workspace)
                return
            except Exception as exc:
                logger.warning("OpenAI-compatible also unavailable: %s", exc)

        raise ValueError(
            "No usable LLM provider available. "
            "Provide an OpenCode model (OPENCODE_MODEL) "
            "or set OpenAI-compatible credentials (OPENAI_API_KEY / LLM_API_KEY)."
        )

    def _build_openai_executor(self, model: str = "", base_url: Optional[str] = None) -> OpenAIChatExecutor:
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        if not api_key.strip():
            raise ValueError("OpenAI-compatible api_key is required (LLM_API_KEY or OPENAI_API_KEY).")

        # Multi-URL support: LLM_BASE_URLS (comma-separated) takes priority
        base_urls_env = os.environ.get("LLM_BASE_URLS", "").strip()
        base_urls: list[str] | None = None
        if base_urls_env:
            base_urls = [u.strip() for u in base_urls_env.split(",") if u.strip()]

        llm_base_url = (
            base_url
            or os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )
        llm_model = model or os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or ""
        if not llm_model.strip():
            raise ValueError("OpenAI-compatible model is required.")

        return OpenAIChatExecutor(
            base_url=llm_base_url,
            base_urls=base_urls,
            api_key=api_key,
            model=llm_model,
            system_prompt=(
                "You are a senior quantitative strategy engineer. "
                "Reply with concise, correct output."
            ),
            retries=self._max_retries,
            timeout_seconds=60,
        )

    def run_result(
        self,
        prompt: str,
        on_turn: Optional[Callable[[Any], None]] = None,
    ) -> AgentRunResult:
        """Run the underlying provider once and return a rich result."""
        if self._closed:
            raise RuntimeError("StrategyAgent is already closed")
        return self._executor.run(prompt, on_turn=on_turn)

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
        """Generate a strategy and ensure a .py file is created in the sandbox."""
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

            # Retry once with a stricter "write a python file" instruction.
            if text and "IStrategy" not in text:
                strict_prompt = (
                    prompt
                    + "\n\nIMPORTANT:\n"
                    + "- You MUST output a complete Freqtrade strategy as a single Python file.\n"
                    + "- The code MUST define exactly one class inheriting from IStrategy.\n"
                    + "- Output ONLY a single ```python ...``` code block (no extra text).\n"
                )
                try:
                    result2 = self.run_result(strict_prompt, on_turn=on_turn)
                    if on_result is not None:
                        on_result(result2)
                    code2 = _extract_code_block(result2.assistant_text)
                    if code2:
                        class_name = _infer_strategy_class_name(code2) or "MinedStrategy"
                        file_name = filename_hint or f"{class_name}.py"
                        out_path = strategies_dir / file_name
                        out_path.write_text(code2, encoding="utf-8")
                        return out_path
                except Exception:
                    logger.debug("Strict generate retry failed", exc_info=True)

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
