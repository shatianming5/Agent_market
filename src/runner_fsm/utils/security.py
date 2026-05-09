from __future__ import annotations

import re

_HARD_DENY_PATTERNS: tuple[str, ...] = (
    # rm -rf / (with optional trailing slash, args, --no-preserve-root) —
    # broadened terminator class so the pattern matches inside quoted
    # interpreter args (e.g. python3 -c "os.system('rm -rf /')").
    r"\brm\s+(-\w*\s+)*-rf\s+/\s*($|[\s;&|\n\"'])",
    r"\brm\s+(-\w*\s+)*-rf\s+/\*",
    # rm -rf ~ or ~/ or $HOME or $HOME/
    r"\brm\s+(-\w*\s+)*-rf\s+~(/|\s|$)",
    r"\brm\s+(-\w*\s+)*-rf\s+\$HOME(/|\s|$)",
    r":\(\)\s*\{\s*:\|\:\s*&\s*\}\s*;\s*:",  # fork bomb
    # Pipe-to-shell remote-execution patterns: curl/wget URL | sh|bash|zsh
    r"\b(curl|wget|fetch)\b[^|]*\|\s*(sudo\s+)?(sh|bash|zsh|fish|python3?|perl|ruby|node)\b",
    # Eval / exec a command-substitution expansion: eval "$(...)" / exec `...`
    r"\b(eval|exec)\s+[\"']?\$\(",
    r"\b(eval|exec)\s+[\"']?`",
    # Writes / dumps to OS-owned paths (NOT just reads — those have legit
    # uses; writes/redirects to system dirs almost never do for a research agent)
    r"(>|>>)\s*/(etc|usr|sys|proc|boot|root|var/log|bin|sbin|lib(64)?)/",
    r"\btee\s+(-a\s+)?/(etc|usr|sys|proc|boot|root|var/log|bin|sbin|lib(64)?)/",
    r"\bchmod\s+\+s\b",  # setuid/setgid escalation
    r"\bchown\s+root\b",

    # CRITICAL FILE PATH references regardless of leading command.
    # Catches `python3 -c "open('/etc/shadow').read()"` and similar
    # interpreter-escape attempts that bypass the (cat|less|cp|...) prefix.
    r"['\"\s]/etc/(passwd|shadow|sudoers|gshadow|sudoers\.d/)",
    r"['\"\s](~|\$HOME|/home/[^\s/'\"]+)/\.ssh/(id_|authorized_keys|known_hosts|config)",
    r"['\"\s](~|\$HOME|/home/[^\s/'\"]+)/\.aws/credentials",
    r"['\"\s](~|\$HOME|/home/[^\s/'\"]+)/\.config/gcloud/",

    # Python / Ruby / Perl interpreter escape — code that explicitly opens
    # subshells or sockets bypasses the shell denylist. Pattern fires on
    # well-known escape APIs in any -c / -e payload, regardless of quoting.
    # Trailing `\b` removed because the last alternative (`exec\s*\(`) ends
    # in `(` which is non-word and never satisfies a word boundary.
    r"\b(python3?|ruby|perl|node)\s+-[ec]\s+[^\n]*(os\.system|subprocess\.(run|Popen|call|check_output)|socket\.|pty\.spawn|\bexec\s*\()",
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
    # SSH credentials + cloud creds — leaking these is the highest-value
    # exfil target for a compromised agent
    r"(cat|less|more|head|tail|cp|mv|rsync)\s+[^\n]*(~|\$HOME|/home/[^\s/]+)/\.ssh/(id_|authorized_keys|known_hosts|config)",
    r"(cat|less|more|head|tail|cp|mv|rsync)\s+[^\n]*(~|\$HOME|/home/[^\s/]+)/\.aws/credentials",
    r"(cat|less|more|head|tail|cp|mv|rsync)\s+[^\n]*(~|\$HOME|/home/[^\s/]+)/\.config/gcloud/",
    r"(cat|less|more|head|tail|cp|mv|rsync)\s+[^\n]*/etc/(passwd|shadow|sudoers)\b",
    # Writing a private key into the cred store
    r">\s*(~|\$HOME|/home/[^\s/]+)/\.ssh/(id_|authorized_keys)",
    # Network exfil of files: curl/wget POSTing a file payload
    r"\b(curl|wget)\b[^\n]*(--data|-d|--data-binary)\s+[\"']?@",
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
