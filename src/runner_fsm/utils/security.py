from __future__ import annotations

import re

_HARD_DENY_PATTERNS: tuple[str, ...] = (
    # rm -rf / (with optional trailing slash, args, --no-preserve-root)
    r"(^|[;&|\n]\s*)\s*rm\s+(-\w*\s+)*-rf\s+/\s*($|[\s;&|\n])",
    r"(^|[;&|\n]\s*)\s*rm\s+(-\w*\s+)*-rf\s+/\*",
    # rm -rf ~ or ~/ or $HOME or $HOME/
    r"(^|[;&|\n]\s*)\s*rm\s+(-\w*\s+)*-rf\s+~(/|\s|$)",
    r"(^|[;&|\n]\s*)\s*rm\s+(-\w*\s+)*-rf\s+\$HOME(/|\s|$)",
    r"(^|[;&|\n]\s*)\s*:\(\)\s*\{\s*:\|\:\s*&\s*\}\s*;\s*:",  # fork bomb
)

_SAFE_DENY_PATTERNS: tuple[str, ...] = (
    r"\bsudo\b",
    r"\bbrew\s+uninstall\b",
    r"\bdocker\s+system\s+prune\b",
    r"\bdocker\s+volume\s+prune\b",
    r"\bmkfs\b",
    r"\bdd\b",
    r"\bshutdown\b",
    r"\breboot\b",
)


def _compile_patterns(patterns: tuple[str, ...]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns if p.strip()]


# Pre-compile once at module level to avoid recompilation on every cmd_allowed() call.
_HARD_DENY_COMPILED = _compile_patterns(_HARD_DENY_PATTERNS)
_SAFE_DENY_COMPILED = _compile_patterns(_SAFE_DENY_PATTERNS)


def _matches_any(patterns: list[re.Pattern[str]], text: str) -> str | None:
    for p in patterns:
        if p.search(text):
            return p.pattern
    return None


def looks_interactive(cmd: str) -> bool:
    s = cmd.strip().lower()
    if not s:
        return False
    if s.startswith("docker login") and "--password-stdin" not in s and " -p " not in s and " --password " not in s:
        return True
    if " gh auth login" in f" {s}" and "--with-token" not in s:
        return True
    return False


def safe_env(base: dict[str, str], extra: dict[str, str], *, unattended: str) -> dict[str, str]:
    env = dict(base)
    env.update({k: str(v) for k, v in extra.items()})
    if unattended == "strict":
        env.setdefault("CI", "1")
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def cmd_allowed(cmd: str) -> tuple[bool, str | None]:
    cmd = cmd.strip()
    if not cmd:
        return False, "empty_command"

    hit = _matches_any(_HARD_DENY_COMPILED, cmd)
    if hit:
        return False, f"blocked_by_hard_deny: {hit}"

    hit = _matches_any(_SAFE_DENY_COMPILED, cmd)
    if hit:
        return False, f"blocked_by_safe_deny: {hit}"

    return True, None
