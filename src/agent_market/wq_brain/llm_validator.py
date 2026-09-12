"""LLM output validator + structured recovery hints.

The agentic loop (``agent_runner.run_agent``) hands control to an LLM CLI
(opencode/hermes) and lets it drive the workflow via shell tools. When the
LLM mis-formats a CLI invocation, fabricates an unknown subcommand, or
emits malformed JSON, the failure today is a generic ``Exception`` caught
in ``run_agent`` — the LLM gets no signal it can act on.

This module gives ``run_agent`` (and any future caller) three structured
checks:

  1. :func:`validate_expression` — is this a parseable FASTEXPR?
  2. :func:`validate_cli_invocation` — is this a recognised
     ``python3 scripts/wq_brain.py <sub>`` call?
  3. :func:`validate_emit_response` — is this a wq_brain ``_emit`` JSON
     blob (possibly wrapped in markdown / surrounding text)?

Each check returns a :class:`ValidationResult`; a failed result includes
both the underlying error and a recovery hint phrased so an LLM can read
it and fix the next attempt.
"""
from __future__ import annotations

import json
import re
import shlex
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# ── Result type ────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    ok: bool
    kind: str = ""           # "expression" | "cli" | "json" | ...
    error: Optional[str] = None
    hint: Optional[str] = None
    parsed: Any = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def __bool__(self) -> bool:
        return self.ok


# ── 1. FASTEXPR validation ─────────────────────────────────────────────


def validate_expression(expr: str) -> ValidationResult:
    """Run the expr through the bundled tokenizer + parser.

    Catches the failure modes the LLM hits most often:

      * unbalanced parens / brackets
      * unknown identifier passed to a context that needs a panel
      * mistyped operator name (``ts_corrwild(...)`` instead of ``ts_corr``)

    Does *not* check operator availability against WQ's actual whitelist —
    that's :func:`agent_market.wq_brain.operators.is_operator_available`'s
    job. We just verify it's a syntactically valid FASTEXPR program.
    """
    if not expr or not expr.strip():
        return ValidationResult(
            ok=False, kind="expression",
            error="empty expression",
            hint="The candidate expression is empty. Re-emit a non-empty FASTEXPR.",
        )
    try:
        from .expr_parser import Parser, tokenize
        ast = Parser(tokenize(expr)).parse_program()
    except Exception as exc:
        return ValidationResult(
            ok=False, kind="expression",
            error=str(exc),
            hint=(
                f"FASTEXPR parser rejected the expression: {exc}. "
                "Common causes: mismatched parens, unknown operator (typo?), "
                "missing comma between arguments, or stray text before/after "
                "the expression. Re-emit a clean FASTEXPR with balanced parens."
            ),
        )
    return ValidationResult(ok=True, kind="expression", parsed=ast)


# ── 2. CLI invocation validation ───────────────────────────────────────


# Mirrors `sub.add_parser("...")` calls in scripts/wq_brain.py + pool subs.
KNOWN_SUBCOMMANDS: frozenset[str] = frozenset({
    "auth", "validate", "mutate", "simulate", "submit",
    "pre-check", "pre-check-local",
    "fetch-data", "update-data", "local-simulate",
    "anti-overfit", "score",
    "pool", "corr",
    "search-arxiv", "search-papers", "math", "docs", "web-search", "fetch-url",
    "skill-search", "skill-list",
    "scan", "agent", "report", "review",
    "audit-data", "calibrate-local", "seed-calibration",
    "kaggle-fetch", "kaggle-import",
})


_POOL_SUBCOMMANDS: frozenset[str] = frozenset({
    "list", "backfill-exprs", "dedup", "resubmit-all", "status", "sync-status",
})


_CLI_PATTERN = re.compile(
    r"(?:python3?\s+)?(?:scripts/)?wq_brain(?:\.py)?\s+(?P<rest>.+)",
)


def validate_cli_invocation(cmd: str) -> ValidationResult:
    """Parse a ``python3 scripts/wq_brain.py <sub> ...`` shell line.

    Accepts two surface forms:

      * ``python3 scripts/wq_brain.py simulate --expr 'rank(close)'``
      * ``wq_brain simulate --expr 'rank(close)'`` (when on $PATH)

    Returns ok=True with parsed = ``{"subcommand": ..., "args": [...]}``
    when the shape is recognised; ok=False otherwise with a hint
    enumerating the closest known subcommand.
    """
    if not cmd or not cmd.strip():
        return ValidationResult(
            ok=False, kind="cli", error="empty command",
            hint="No CLI invocation detected. Emit a `python3 scripts/wq_brain.py <sub> ...` line.",
        )
    match = _CLI_PATTERN.search(cmd.strip())
    if not match:
        return ValidationResult(
            ok=False, kind="cli",
            error="not a wq_brain CLI invocation",
            hint=(
                "Expected `python3 scripts/wq_brain.py <subcommand> ...` "
                "(or `wq_brain <subcommand> ...` if on $PATH)."
            ),
        )
    try:
        argv = shlex.split(match.group("rest"))
    except ValueError as exc:
        return ValidationResult(
            ok=False, kind="cli",
            error=f"shell-tokenisation failed: {exc}",
            hint="Mismatched quotes in the CLI line. Re-emit with balanced single/double quotes.",
        )
    if not argv:
        return ValidationResult(
            ok=False, kind="cli", error="missing subcommand",
            hint=f"No subcommand. Pick one of: {sorted(KNOWN_SUBCOMMANDS)}",
        )
    sub = argv[0]
    if sub not in KNOWN_SUBCOMMANDS:
        suggestion = _closest_subcommand(sub)
        hint = f"Unknown subcommand `{sub}`."
        if suggestion:
            hint += f" Did you mean `{suggestion}`?"
        hint += f" Known: {sorted(KNOWN_SUBCOMMANDS)}"
        return ValidationResult(ok=False, kind="cli", error=hint, hint=hint)
    if sub == "pool":
        if len(argv) < 2 or argv[1] not in _POOL_SUBCOMMANDS:
            return ValidationResult(
                ok=False, kind="cli",
                error="`pool` requires a sub-subcommand",
                hint=(
                    "`pool` needs one of: "
                    f"{sorted(_POOL_SUBCOMMANDS)}. e.g. `pool list --tag X`."
                ),
            )
    return ValidationResult(
        ok=True, kind="cli",
        parsed={"subcommand": sub, "args": argv[1:]},
    )


def _closest_subcommand(name: str) -> Optional[str]:
    """Levenshtein-1 / prefix match against KNOWN_SUBCOMMANDS."""
    name = name.lower()
    candidates = sorted(KNOWN_SUBCOMMANDS)
    # Prefix match wins
    prefix = [c for c in candidates if c.startswith(name) or name.startswith(c)]
    if prefix:
        return prefix[0]
    # Distance-1 (single typo) — tiny, no need for full Levenshtein
    for c in candidates:
        if abs(len(c) - len(name)) > 2:
            continue
        diffs = sum(1 for a, b in zip(c, name) if a != b)
        if diffs + abs(len(c) - len(name)) <= 2:
            return c
    return None


# ── 3. JSON / _emit response validation ────────────────────────────────


_DECODER = json.JSONDecoder()


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """Pull the first valid ``{...}`` object from ``text`` and decode it.

    LLMs frequently wrap JSON in markdown fences or a chatty preamble. We
    use ``json.JSONDecoder().raw_decode()`` to find the first SYNTACTICALLY
    VALID object, which correctly handles:

      * markdown ``” fences (handled by stripping)
      * preamble before the JSON
      * trailing prose after the JSON
      * multiple ``{...}`` blocks (returns the first valid one)

    The previous greedy regex ``\\{[\\s\\S]*\\}`` matched outermost braces
    and could incorrectly merge two adjacent ``{}`` blocks.
    """
    if not text:
        return None
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[len("json"):]
            text = inner.strip()
    # Try fast path: entire text is JSON. Only return dicts — top-level
    # arrays / scalars aren't `_emit` payloads.
    try:
        fast = json.loads(text)
        if isinstance(fast, dict):
            return fast
    except (ValueError, TypeError):
        pass
    # Slow path: scan for the first '{' that begins a syntactically valid
    # object. raw_decode tells us where the parse ended so we can resume
    # past invalid leading braces.
    pos = 0
    while True:
        i = text.find("{", pos)
        if i < 0:
            return None
        try:
            obj, _ = _DECODER.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
            pos = i + 1
        except ValueError:
            pos = i + 1


def validate_emit_response(
    text: str,
    *,
    required_fields: Optional[list[str]] = None,
) -> ValidationResult:
    """Verify an `_emit` response from a wq_brain CLI invocation.

    Returns ok=True with parsed=<dict> when the JSON parses and contains
    the expected ``ok`` boolean (plus any caller-required fields). Falls
    back to ok=False with a hint that the LLM can use to reformat.
    """
    parsed = _extract_json(text)
    if parsed is None:
        return ValidationResult(
            ok=False, kind="json",
            error="no JSON object found in response",
            hint=(
                "The response did not contain a valid JSON object. "
                "wq_brain CLI commands ALWAYS print a single JSON object "
                "via `_emit`. Re-run the CLI command and capture stdout."
            ),
        )
    if "ok" not in parsed:
        return ValidationResult(
            ok=False, kind="json",
            error="response missing 'ok' field",
            hint=(
                "Every wq_brain `_emit` payload includes an `ok` boolean. "
                "Got: " + ", ".join(sorted(parsed.keys())[:8]) +
                ". Likely you captured shell output unrelated to wq_brain."
            ),
            parsed=parsed,
        )
    missing = [f for f in (required_fields or []) if f not in parsed]
    if missing:
        return ValidationResult(
            ok=False, kind="json",
            error=f"response missing required fields: {missing}",
            hint=(
                f"The CLI emitted JSON but it lacks {missing}. "
                "Either the subcommand failed (check `ok` and `error`) or "
                "you parsed the wrong stdout block."
            ),
            parsed=parsed,
        )
    return ValidationResult(ok=True, kind="json", parsed=parsed)


# ── 4. Recovery hint formatter ─────────────────────────────────────────


def format_recovery_hint(result: ValidationResult) -> str:
    """Render a markdown block the agent can paste back into the next prompt.

    Empty string when the result is OK.
    """
    if result.ok:
        return ""
    return (
        f"### ❌ {result.kind.title()} validation failed\n"
        f"\n"
        f"- error: `{result.error}`\n"
        f"- hint: {result.hint}\n"
    )
