"""LLM endpoint probe + failover.

Multiple production incidents traced back to a single configured endpoint
returning ``503 / model_not_found`` and the agent loop silently dying at the
title-generation step. This module enumerates candidate ``(base_url,
api_key, model)`` triples and probes each via a minimal chat-completions
request so the colony controller can switch to the first healthy combo.

Candidates are read from one of (in priority order):

  1. ``--candidates-file <path>`` JSON list.
  2. ``OPENAI_FALLBACK_ENDPOINTS`` env var (same JSON shape).
  3. The single endpoint configured via ``OPENAI_BASE_URL`` /
     ``OPENAI_API_KEY`` / ``OPENAI_MODEL`` — useful as a smoke check.

Each candidate JSON object has shape::

    {"base_url": "https://example/v1", "api_key": "sk-...",
     "model": "gpt-5-something"}

``api_key`` may be omitted; ``OPENAI_API_KEY`` is used as a fallback so the
candidate file can stay in version control without leaking secrets.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


@dataclass
class EndpointCandidate:
    base_url: str
    model: str
    api_key: Optional[str] = None
    label: str = ""

    def materialise(self) -> tuple[str, str, str]:
        """Return (base_url, api_key, model) with env fallbacks applied."""
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY") or ""
        return self.base_url, api_key, self.model


@dataclass
class ProbeResult:
    candidate: EndpointCandidate
    ok: bool
    http_status: Optional[int] = None
    elapsed_ms: int = 0
    error: Optional[str] = None
    body_excerpt: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _resolve_url(base: str) -> str:
    """Normalise BASE_URL → chat-completions URL (mirrors cmd_ping_llm)."""
    base = base.rstrip("/")
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def probe_candidate(
    candidate: EndpointCandidate, *, timeout: float = 10.0
) -> ProbeResult:
    """Issue a minimal chat-completions request and report the outcome."""
    base, api_key, model = candidate.materialise()
    if not base or not api_key or not model:
        return ProbeResult(
            candidate=candidate, ok=False,
            error="missing base_url / api_key / model",
        )
    url = _resolve_url(base)
    body = json.dumps({
        "model": model,
        "max_tokens": 4,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "ping"}],
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return ProbeResult(
            candidate=candidate, ok=True,
            elapsed_ms=elapsed_ms, http_status=200,
            body_excerpt=payload[:300],
        )
    except urllib.error.HTTPError as exc:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        body = exc.read().decode("utf-8", errors="replace")[:300]
        return ProbeResult(
            candidate=candidate, ok=False,
            elapsed_ms=elapsed_ms, http_status=exc.code,
            error=exc.reason, body_excerpt=body,
        )
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return ProbeResult(
            candidate=candidate, ok=False,
            elapsed_ms=elapsed_ms,
            error=f"{type(exc).__name__}: {exc}",
        )


def probe_candidates(
    candidates: Sequence[EndpointCandidate], *, timeout: float = 10.0
) -> list[ProbeResult]:
    return [probe_candidate(c, timeout=timeout) for c in candidates]


def first_healthy(probes: Iterable[ProbeResult]) -> Optional[ProbeResult]:
    for probe in probes:
        if probe.ok:
            return probe
    return None


def load_candidates_from_file(path: Path) -> list[EndpointCandidate]:
    """Load a JSON list of candidates from disk.

    Tolerates a single-object file as a convenience for one-shot smoke checks.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    items: list[dict[str, Any]]
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = [raw]
    else:
        raise ValueError(f"candidates file must be JSON object or list, got {type(raw).__name__}")
    return [
        EndpointCandidate(
            base_url=str(i.get("base_url") or "").strip(),
            model=str(i.get("model") or "").strip(),
            api_key=i.get("api_key"),
            label=str(i.get("label") or "").strip(),
        )
        for i in items
        if (i.get("base_url") and i.get("model"))
    ]


def load_candidates_from_env() -> list[EndpointCandidate]:
    """Read ``OPENAI_FALLBACK_ENDPOINTS`` JSON or fall back to env triple."""
    raw = os.environ.get("OPENAI_FALLBACK_ENDPOINTS")
    if raw:
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(items, list):
            return []
        return [
            EndpointCandidate(
                base_url=str(i.get("base_url") or "").strip(),
                model=str(i.get("model") or "").strip(),
                api_key=i.get("api_key"),
                label=str(i.get("label") or "").strip(),
            )
            for i in items
            if isinstance(i, dict) and i.get("base_url") and i.get("model")
        ]
    base = os.environ.get("OPENAI_BASE_URL") or ""
    model = os.environ.get("OPENAI_MODEL") or ""
    if base and model:
        return [
            EndpointCandidate(base_url=base, model=model, label="env_default")
        ]
    return []


def write_env_local(path: Path, candidate: EndpointCandidate) -> None:
    """Persist the chosen endpoint into ``.env.local`` (next to ``.env``).

    The agent_runner loader reads ``.env.local`` *before* ``.env`` so this
    file wins. Existing variables in ``.env.local`` are preserved unless we
    overwrite them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            existing[k.strip()] = v.strip()
    base, api_key, model = candidate.materialise()
    existing["OPENAI_BASE_URL"] = base
    if api_key:
        existing["OPENAI_API_KEY"] = api_key
    existing["OPENAI_MODEL"] = model
    lines = [
        "# Written by `wq_brain endpoint failover` — auto-managed.",
        "# Remove or edit to override the colony's chosen endpoint.",
    ]
    for k, v in existing.items():
        lines.append(f"{k}={v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
