"""Alpha pool management: persistence, dedup, and diversity tracking."""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

from .dtypes import AlphaPoolEntry, AlphaCandidate

logger = logging.getLogger(__name__)


def _tokenset(expr: str) -> frozenset[str]:
    import re
    return frozenset(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr.lower()))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union)


class DuplicateDetector:
    """Fast local dedup using Jaccard similarity on token sets.

    Two expressions are considered duplicates if their token-set Jaccard
    similarity is ≥ threshold (default 0.95).
    """

    def __init__(self, threshold: float = 0.95, maxlen: int = 200) -> None:
        self._threshold = threshold
        self._cache: deque[tuple[str, frozenset]] = deque(maxlen=maxlen)

    def is_duplicate(self, expr: str) -> bool:
        ts = _tokenset(expr)
        for _, cached_ts in self._cache:
            if jaccard(ts, cached_ts) >= self._threshold:
                return True
        return False

    def add(self, expr: str) -> None:
        self._cache.append((expr, _tokenset(expr)))

    def seen_count(self) -> int:
        return len(self._cache)


class AlphaPool:
    """Persistent alpha pool for a given tag.

    Persists to a JSON file and keeps a DuplicateDetector in memory for
    fast pre-check before submitting to the WQ API.
    """

    def __init__(self, pool_path: Path, *, corr_threshold: float = 0.70) -> None:
        self._path = pool_path
        self._corr_threshold = corr_threshold
        self._entries: list[AlphaPoolEntry] = []
        self._detector = DuplicateDetector()
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        if pool_path.exists():
            self._load()
            for e in self._entries:
                self._detector.add(e.expr)

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._entries = [AlphaPoolEntry.from_dict(d) for d in data.get("entries", [])]
            logger.info("Loaded %d alphas from pool %s", len(self._entries), self._path)
        except Exception as exc:
            logger.warning("Failed to load pool from %s: %s", self._path, exc)
            self._entries = []

    def save(self) -> None:
        data = json.dumps(
            {"entries": [e.to_dict() for e in self._entries], "saved_at": time.time()},
            ensure_ascii=False,
            indent=2,
        )
        tmp = self._path.with_suffix(".tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            os.write(fd, data.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        tmp.rename(self._path)

    # ── Query ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def is_local_duplicate(self, expr: str) -> bool:
        return self._detector.is_duplicate(expr)

    def all_exprs(self) -> list[str]:
        return [e.expr for e in self._entries]

    def top_n_by_fitness(self, n: int = 10) -> list[AlphaPoolEntry]:
        return sorted(self._entries, key=lambda e: e.fitness, reverse=True)[:n]

    def summary_for_prompt(self, n: int = 20) -> str:
        """Return a compact text summary of the pool for LLM context."""
        top = self.top_n_by_fitness(n)
        if not top:
            return "Alpha pool is empty — generate fresh diverse alphas."
        lines = [f"Pool has {len(self._entries)} submitted alphas. Top {len(top)} by fitness:"]
        for i, e in enumerate(top, 1):
            lines.append(
                f"  {i}. [sharpe={e.sharpe:.2f} fitness={e.fitness:.2f}] {e.expr[:80]}"
            )
        lines.append(
            "\nIMPORTANT: Generate structurally DIFFERENT alphas — avoid reusing the "
            "same operators/fields combination as the above."
        )
        return "\n".join(lines)

    # ── Mutation ──────────────────────────────────────────────────────────

    def add(self, entry: AlphaPoolEntry) -> None:
        self._entries.append(entry)
        self._detector.add(entry.expr)
        self.save()

    def add_from_candidate(
        self, candidate: AlphaCandidate, *, tag: str = ""
    ) -> Optional[AlphaPoolEntry]:
        if not candidate.sim_result or not candidate.sim_result.alpha_id:
            return None
        r = candidate.sim_result
        entry = AlphaPoolEntry(
            alpha_id=r.alpha_id,
            expr=candidate.expr,
            settings_dict=candidate.settings.to_api_dict(),
            sharpe=r.sharpe or 0.0,
            fitness=r.fitness or 0.0,
            returns=r.returns or 0.0,
            turnover=r.turnover or 0.0,
            tag=tag,
        )
        self.add(entry)
        return entry

    def clean(self, *, keep_top_n: Optional[int] = None) -> int:
        before = len(self._entries)
        if keep_top_n is not None:
            self._entries = self.top_n_by_fitness(keep_top_n)
        self._rebuild_detector()
        self.save()
        return before - len(self._entries)

    def _rebuild_detector(self) -> None:
        self._detector = DuplicateDetector()
        for e in self._entries:
            self._detector.add(e.expr)
