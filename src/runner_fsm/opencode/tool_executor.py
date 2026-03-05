from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..utils.security import cmd_allowed, looks_interactive, safe_env
from ..utils.subprocess import STDIO_TAIL_CHARS, run_cmd_capture, tail
from .tool_parser import ToolCall, ToolResult

# Allowlist of env var prefixes/names safe to pass to tool subprocesses.
_ENV_ALLOWLIST_PREFIXES = (
    "PATH", "HOME", "LANG", "LC_", "TERM", "SHELL",
    "PYTHONPATH", "PYTHONUNBUFFERED", "VIRTUAL_ENV",
    "CUDA", "NCCL", "NVIDIA", "GPU", "TORCH", "HF_",
    "TRANSFORMERS", "TOKENIZERS", "WANDB_", "MLFLOW_",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "CONDA", "PIP_", "XDG_",
    # Pipeline-specific vars needed by agent bash tool subprocesses
    "DATA_PATH", "MODEL_PATH", "OUTPUT_DIR", "GRADING_SERVER_URL",
)

# Secret env var names that should NEVER be forwarded even if their prefix
# is in the allowlist (e.g. HF_TOKEN matches HF_ prefix).
_ENV_SECRET_DENYLIST = frozenset((
    "HF_TOKEN", "HF_API_KEY", "HUGGING_FACE_HUB_TOKEN",
    "WANDB_API_KEY",
    "MLFLOW_TRACKING_TOKEN",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
))


def _is_env_like(path: Path) -> bool:
    name = path.name.lower()
    return name == ".env" or name.startswith(".env.") or name.endswith(".env") or ".env." in name


def _within_root(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


def _sanitized_env(*, unattended: str) -> dict[str, str]:
    """Build a minimal env for tool subprocesses using an allowlist."""
    base: dict[str, str] = {}
    for k, v in os.environ.items():
        ku = k.upper()
        if ku in _ENV_SECRET_DENYLIST:
            continue
        if any(ku == pfx or ku.startswith(pfx) for pfx in _ENV_ALLOWLIST_PREFIXES):
            base[k] = v
    return safe_env(base, {}, unattended=unattended)


@dataclass(frozen=True)
class ToolPolicy:
    repo: Path
    unattended: str

    def _allow_file(self, path: Path, op: str) -> tuple[bool, str | None]:
        if _is_env_like(path):
            return False, f"{op}_env_files_is_blocked"
        if not _within_root(self.repo, path):
            return False, "path_outside_repo"
        return True, None

    def allow_file_read(self, path: Path) -> tuple[bool, str | None]:
        return self._allow_file(path, "reading")

    def allow_file_write(self, path: Path) -> tuple[bool, str | None]:
        return self._allow_file(path, "writing")

    def allow_bash(self, cmd: str) -> tuple[bool, str | None]:
        cmd = cmd.strip()
        if not cmd:
            return False, "empty_command"
        allowed, reason = cmd_allowed(cmd)
        if not allowed:
            return False, reason or "blocked"
        if self.unattended == "strict" and looks_interactive(cmd):
            return False, "likely_interactive_command_disallowed_in_strict_mode"
        return True, None


def execute_tool_calls(
    calls: Iterable[ToolCall],
    *,
    policy: ToolPolicy,
) -> list[ToolResult]:
    results: list[ToolResult] = []

    for call in calls:
        if call.kind == "file":
            data = call.payload
            file_path_raw = str(data.get("filePath") or "").strip()
            content = data.get("content")

            if not file_path_raw:
                results.append(ToolResult(kind="file", ok=False, detail={"error": "missing_filePath"}))
                continue

            file_path = Path(file_path_raw).expanduser()
            if not file_path.is_absolute():
                file_path = (policy.repo / file_path).resolve()
            else:
                file_path = file_path.resolve()

            if content is None and ("oldString" in data or "newString" in data):
                old = data.get("oldString")
                new = data.get("newString")
                if new is None:
                    new_s = ""
                elif isinstance(new, str):
                    new_s = new
                else:
                    results.append(
                        ToolResult(kind="edit", ok=False,
                                   detail={"filePath": str(file_path), "error": "invalid_newString"})
                    )
                    continue

                if old is None or (isinstance(old, str) and old == ""):
                    ok, reason = policy.allow_file_write(file_path)
                    if not ok:
                        results.append(
                            ToolResult(kind="edit", ok=False,
                                       detail={"filePath": str(file_path), "error": reason or "blocked"})
                        )
                        continue
                    try:
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_text(new_s, encoding="utf-8", errors="replace")
                    except Exception as e:
                        results.append(
                            ToolResult(kind="edit", ok=False, detail={
                                "filePath": str(file_path), "error": "write_failed",
                                "exception": type(e).__name__, "message": str(e)[:200],
                            })
                        )
                        continue
                    results.append(
                        ToolResult(kind="edit", ok=True,
                                   detail={"filePath": str(file_path), "bytes": len(new_s), "mode": "replace"})
                    )
                    continue

                if not isinstance(old, str):
                    results.append(
                        ToolResult(kind="edit", ok=False,
                                   detail={"filePath": str(file_path), "error": "invalid_oldString"})
                    )
                    continue

                ok, reason = policy.allow_file_write(file_path)
                if not ok:
                    results.append(
                        ToolResult(kind="edit", ok=False,
                                   detail={"filePath": str(file_path), "error": reason or "blocked"})
                    )
                    continue
                if not file_path.exists():
                    results.append(
                        ToolResult(kind="edit", ok=False,
                                   detail={"filePath": str(file_path), "error": "not_found"})
                    )
                    continue
                try:
                    raw = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    results.append(
                        ToolResult(kind="edit", ok=False, detail={
                            "filePath": str(file_path), "error": "read_failed",
                            "exception": type(e).__name__, "message": str(e)[:200],
                        })
                    )
                    continue
                matches = raw.count(old)
                if matches <= 0:
                    results.append(
                        ToolResult(kind="edit", ok=False,
                                   detail={"filePath": str(file_path), "error": "oldString_not_found"})
                    )
                    continue
                if matches != 1:
                    results.append(
                        ToolResult(kind="edit", ok=False,
                                   detail={"filePath": str(file_path), "error": "oldString_not_unique", "matches": matches})
                    )
                    continue
                updated = raw.replace(old, new_s, 1)
                try:
                    file_path.write_text(updated, encoding="utf-8", errors="replace")
                except Exception as e:
                    results.append(
                        ToolResult(kind="edit", ok=False, detail={
                            "filePath": str(file_path), "error": "write_failed",
                            "exception": type(e).__name__, "message": str(e)[:200],
                        })
                    )
                    continue
                results.append(
                    ToolResult(kind="edit", ok=True,
                               detail={"filePath": str(file_path), "bytes": len(updated), "mode": "replace_once"})
                )
                continue

            if content is None:
                ok, reason = policy.allow_file_read(file_path)
                if not ok:
                    results.append(
                        ToolResult(kind="read", ok=False,
                                   detail={"filePath": str(file_path), "error": reason or "blocked"})
                    )
                    continue
                if not file_path.exists():
                    results.append(
                        ToolResult(kind="read", ok=False,
                                   detail={"filePath": str(file_path), "error": "not_found"})
                    )
                    continue
                try:
                    raw = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    results.append(
                        ToolResult(kind="read", ok=False, detail={
                            "filePath": str(file_path), "error": "read_failed",
                            "exception": type(e).__name__, "message": str(e)[:200],
                        })
                    )
                    continue
                results.append(
                    ToolResult(kind="read", ok=True,
                               detail={"filePath": str(file_path), "content": tail(raw, 20000)})
                )
                continue

            if not isinstance(content, str):
                results.append(
                    ToolResult(kind="write", ok=False,
                               detail={"filePath": str(file_path), "error": "invalid_content"})
                )
                continue

            ok, reason = policy.allow_file_write(file_path)
            if not ok:
                results.append(
                    ToolResult(kind="write", ok=False,
                               detail={"filePath": str(file_path), "error": reason or "blocked"})
                )
                continue

            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8", errors="replace")
            except Exception as e:
                results.append(
                    ToolResult(kind="write", ok=False, detail={
                        "filePath": str(file_path), "error": "write_failed",
                        "exception": type(e).__name__, "message": str(e)[:200],
                    })
                )
                continue
            results.append(
                ToolResult(kind="write", ok=True,
                           detail={"filePath": str(file_path), "bytes": len(content)})
            )
            continue

        if call.kind == "bash":
            data = call.payload
            cmd = str(data.get("command") or "").strip()
            ok, reason = policy.allow_bash(cmd)
            if not ok:
                results.append(ToolResult(kind="bash", ok=False, detail={"command": cmd, "error": reason or "blocked"}))
                continue

            env = _sanitized_env(unattended=str(policy.unattended or "strict"))
            res = run_cmd_capture(cmd, policy.repo, timeout_seconds=60, env=env)
            results.append(
                ToolResult(
                    kind="bash",
                    ok=(res.rc == 0),
                    detail={
                        "command": cmd, "rc": res.rc, "timed_out": res.timed_out,
                        "stdout": tail(res.stdout or "", STDIO_TAIL_CHARS),
                        "stderr": tail(res.stderr or "", STDIO_TAIL_CHARS),
                    },
                )
            )
            continue

        results.append(ToolResult(kind=str(call.kind), ok=False, detail={"error": "unsupported_tool"}))

    return results
