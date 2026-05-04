"""Per-tag alpha pool with jaccard duplicate detection.

Atomic JSON write (tmp + rename + fsync). Thread-safe.
"""
from __future__ import annotations

import json
import os
import re
import threading
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

from .dtypes import AlphaCandidate, AlphaPoolEntry

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_SAVE_LOCK = threading.Lock()


def _tokenize(expr: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(expr.lower()))


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


class DuplicateDetector:
    """In-memory rolling window of recently-seen exprs (default 200)."""

    def __init__(self, *, threshold: float = 0.95, maxlen: int = 200) -> None:
        self._seen: deque[tuple[str, frozenset[str]]] = deque(maxlen=maxlen)
        self._threshold = threshold

    def is_duplicate(self, expr: str) -> bool:
        toks = _tokenize(expr)
        for prior_expr, prior_toks in self._seen:
            if prior_expr == expr:
                return True
            if _jaccard(toks, prior_toks) >= self._threshold:
                return True
        return False

    def add(self, expr: str) -> None:
        self._seen.append((expr, _tokenize(expr)))


class AlphaPool:
    """Persistent JSON-backed pool of submitted alphas, scoped by tag."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._entries: list[AlphaPoolEntry] = []
        if self._path.exists():
            self._load()

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterable[AlphaPoolEntry]:
        return iter(self._entries)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def entries(self) -> list[AlphaPoolEntry]:
        return list(self._entries)

    def add(self, entry: AlphaPoolEntry) -> bool:
        if any(e.alpha_id == entry.alpha_id for e in self._entries):
            return False
        self._entries.append(entry)
        self._save()
        return True

    def add_from_candidate(self, c: AlphaCandidate, *, tag: str) -> Optional[AlphaPoolEntry]:
        if not c.sim_result or not c.sim_result.alpha_id:
            return None
        r = c.sim_result
        if r.sharpe is None or r.fitness is None or r.returns is None or r.turnover is None:
            return None
        entry = AlphaPoolEntry(
            alpha_id=r.alpha_id,
            expr=c.expr,
            settings_dict=c.settings.to_api_dict() if hasattr(c.settings, "to_api_dict") else dict(c.settings.__dict__),
            sharpe=float(r.sharpe),
            fitness=float(r.fitness),
            returns=float(r.returns),
            turnover=float(r.turnover),
            tag=tag,
            source=c.source,
        )
        if self.add(entry):
            return entry
        return None

    def top_n_by_fitness(self, n: int = 5) -> list[AlphaPoolEntry]:
        return sorted(self._entries, key=lambda e: -e.fitness)[:n]

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            self._entries = []
            return
        self._entries = [AlphaPoolEntry.from_dict(d) for d in data]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with _SAVE_LOCK:
            tmp = self._path.with_suffix(".json.tmp")
            payload = json.dumps([e.to_dict() for e in self._entries], indent=2)
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._path)
