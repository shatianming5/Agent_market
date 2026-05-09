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

    def upsert(self, entry: AlphaPoolEntry) -> str:
        """Insert or update by alpha_id.

        Returns one of: ``"inserted"`` (new), ``"updated"`` (replaced
        existing), ``"unchanged"`` (existing entry already matches).
        Used by the submit-worker so per-alpha outcomes (ACTIVE / REJECTED
        / UNSUBMITTED + reasons) overwrite previously-stored stale state.

        Subtlety: ``self.entries`` returns a list whose elements share
        object identity with ``self._entries``. Callers that pull an
        entry, mutate it in place, then call ``upsert`` end up passing
        the same object that's already stored — naive ``e == entry``
        would always be True after the mutation, so the in-memory change
        would be silently dropped without writing to disk. Guard against
        that by checking ``e is not entry``: when identity matches, we
        always rewrite + save.

        Codex R3-CRIT: every upserted entry is marked authoritative for
        this save so the precedence-merge logic doesn't silently revert
        intentional demotions (e.g. ACTIVE → REJECTED, LOCAL_BLOCKED →
        UNSUBMITTED via reset).
        """
        for i, e in enumerate(self._entries):
            if e.alpha_id == entry.alpha_id:
                if e is not entry and e.to_dict() == entry.to_dict():
                    return "unchanged"
                self._entries[i] = entry
                self._save(authoritative_ids={entry.alpha_id})
                return "updated"
        self._entries.append(entry)
        self._save(authoritative_ids={entry.alpha_id})
        return "inserted"

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

    # Status-precedence ordering for the per-entry merge during _save.
    # An on-disk row with HIGHER precedence is never overwritten by an
    # in-memory row with LOWER precedence — protects ACTIVE / blocked
    # terminal verdicts from being demoted by a stale concurrent writer
    # (Codex review R2-#2).
    _STATUS_PRECEDENCE: dict[str, int] = {
        "ACTIVE": 100,
        "REJECTED": 80,
        "VERIFICATION_FAILED": 80,
        "LOCAL_BLOCKED": 60,
        "SELF_CORR_BLOCKED": 60,
        "UNSUBMITTED": 20,
        "QUEUED": 10,
        "": 0,
    }

    def _save(self, *, merge_missing: bool = True,
              authoritative_ids: Optional[set[str]] = None) -> None:
        """Atomic, cross-process-safe write.

        Args:
          merge_missing: when True (default), entries that exist on disk
            but not in our in-memory state are appended back. This
            protects against the lost-update race where two writers each
            insert distinct alphas. When False (used by deliberate
            shrinks like :func:`cmd_pool_dedup`), the on-disk file is
            replaced verbatim with the in-memory state — entries removed
            from memory STAY removed.
          authoritative_ids: alpha_ids the caller deliberately mutated.
            For these IDs the in-memory version is written verbatim,
            bypassing status-precedence and same-status-richness merge
            rules. Used by ``sync-status --reset-local-blocks`` and any
            other path where the caller has authoritative truth (e.g.
            just observed it from WQ) and must override stale terminal
            states on disk.

        Codex review history:
          * R1-#3 added fcntl + merge-from-disk to fix lost-update on
            concurrent inserts.
          * R2-CRIT discovered the merge made deletes impossible —
            ``pool dedup`` would re-add what it just dropped. Hence the
            ``merge_missing`` switch.
          * R2-#2 added per-entry status-precedence: an on-disk
            ``ACTIVE`` is never overwritten by an in-memory
            ``UNSUBMITTED`` from a stale concurrent reader.
          * R3-CRIT discovered the precedence merge silently reverts
            intentional demotions (sync-status --reset-local-blocks
            mutating LOCAL_BLOCKED → UNSUBMITTED was undone by the disk
            read seeing higher-precedence LOCAL_BLOCKED). Added
            ``authoritative_ids`` so caller can opt out of merge for
            specific entries.
          * Also added same-status richness tiebreak: disk row with
            ``rejection_reasons`` set beats in-memory row without.

        On platforms without fcntl (e.g. Windows), the cross-process lock
        degrades to a no-op and concurrent CLI invocations remain
        single-process-safe only — same fallback as ``quota_monitor``.
        """
        auth = authoritative_ids or set()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import fcntl  # POSIX only
            _have_fcntl = True
        except ImportError:
            _have_fcntl = False

        lock_path = self._path.with_suffix(".json.lock")
        with _SAVE_LOCK:
            lock_fd: int | None = None
            if _have_fcntl:
                try:
                    lock_fd = os.open(str(lock_path),
                                       os.O_RDWR | os.O_CREAT, 0o644)
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                except OSError:
                    if lock_fd is not None:
                        os.close(lock_fd)
                        lock_fd = None
            try:
                if merge_missing and self._path.exists():
                    # Re-read disk inside the critical section. Three cases:
                    #   (a) entry on disk we never saw → APPEND
                    #   (b) entry in `auth` (caller deliberately mutated) →
                    #       use memory verbatim, skip merge entirely
                    #   (c) otherwise → status-precedence merge + same-status
                    #       richness tiebreak (don't demote ACTIVE; don't lose
                    #       rejection_reasons via stale memory snapshot)
                    try:
                        on_disk = json.loads(self._path.read_text(encoding="utf-8"))
                        on_disk_by_id = {
                            d.get("alpha_id"): AlphaPoolEntry.from_dict(d)
                            for d in on_disk if d.get("alpha_id")
                        }
                        in_mem_index = {e.alpha_id: i
                                          for i, e in enumerate(self._entries)}
                        for aid, disk_entry in on_disk_by_id.items():
                            if aid in auth:
                                # Caller authoritatively wrote this — never
                                # overwrite from stale disk state.
                                continue
                            if aid not in in_mem_index:
                                self._entries.append(disk_entry)
                                continue
                            i = in_mem_index[aid]
                            mem_entry = self._entries[i]
                            disk_prec = self._STATUS_PRECEDENCE.get(
                                getattr(disk_entry, "verified_status", ""), 0)
                            mem_prec = self._STATUS_PRECEDENCE.get(
                                getattr(mem_entry, "verified_status", ""), 0)
                            if disk_prec > mem_prec:
                                # Don't demote ACTIVE → stale UNSUBMITTED
                                self._entries[i] = disk_entry
                            elif disk_prec == mem_prec:
                                # Same-status richness tiebreak (Codex R3-#4):
                                # prefer the row with more rejection_reasons or
                                # newer verified_at — protects against losing
                                # WQ-probed failure details when an unrelated
                                # writer's stale memory clobbers them.
                                disk_rj = len(getattr(disk_entry, "rejection_reasons", []) or [])
                                mem_rj  = len(getattr(mem_entry,  "rejection_reasons", []) or [])
                                if disk_rj > mem_rj:
                                    self._entries[i] = disk_entry
                                elif disk_rj == mem_rj:
                                    disk_t = float(getattr(disk_entry, "verified_at", 0) or 0)
                                    mem_t  = float(getattr(mem_entry,  "verified_at", 0) or 0)
                                    if disk_t > mem_t:
                                        self._entries[i] = disk_entry
                    except (ValueError, KeyError):
                        # Corrupt on-disk file → trust in-memory state
                        pass

                tmp = self._path.with_suffix(".json.tmp")
                payload = json.dumps([e.to_dict() for e in self._entries], indent=2)
                fd = os.open(str(tmp),
                              os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
                try:
                    os.write(fd, payload.encode("utf-8"))
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.replace(tmp, self._path)
                # Parent-dir fsync makes the rename durable across crashes
                try:
                    dir_fd = os.open(str(self._path.parent), os.O_RDONLY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                except OSError:
                    pass
            finally:
                if lock_fd is not None:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    finally:
                        os.close(lock_fd)

    def replace_all(self, entries: list[AlphaPoolEntry]) -> None:
        """Atomic shrink: replace pool contents with ``entries`` exactly.

        Used by deliberate dedup / GC operations where the caller has
        decided which entries to keep and the on-disk merge from
        :func:`_save` would re-add the dropped ones. Caller is
        responsible for the right policy — ``replace_all`` does NOT do
        per-entry status-precedence checks, so concurrent writers can
        still race. Take an explicit lock at a higher level if needed.
        """
        self._entries = list(entries)
        self._save(merge_missing=False)
