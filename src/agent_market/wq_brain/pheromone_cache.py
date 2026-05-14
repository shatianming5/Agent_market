"""Bounded shared pheromone cache for the colony.

Each colony run owns one cache file at
``<wq_brain_root>/colony/<colony_tag>/shared_cache.json`` with per-altitude
buckets. PheroViz Sec. 4.7 defines the projection ``Π_{K_v}`` that bounds the
node memory; we mirror that here at the *altitude* level since wq_brain's HCT
nodes are coarser than PheroViz's per-figure nodes.

Capacity per altitude (``K_g``):

    L1 = 64  L2 = 48  L3 = 32  L4 = 16

TTL per altitude (seconds):

    L1 = 7 days  L2 = 5 days  L3 = 3 days  L4 = 1 day

Score (used to break ties at eviction time) follows the PheroViz formulation::

    score(ℓ) = α·ΔU + β·support + γ·recency − δ·conflict

with α=1.0, β=0.2, γ=0.1, δ=0.5 and ``recency = exp((ts−now)/(TTL/2))``.

Entries with ``ts`` older than TTL are skipped at read time, never returned,
and dropped on the next write. ``support`` counts successful safe reuses;
``conflicts`` counts logged disagreements with higher-priority links.

The cache is a JSON file guarded by an exclusive file lock for the duration
of every read-modify-write so two parallel colony workers cannot corrupt it.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .paths import wq_brain_root
from .tried_log import (
    ALTITUDE_L1_REGION_UNIVERSE,
    ALTITUDE_L2_OP_FAMILY,
    ALTITUDE_L3_SLOT_PARAM,
    ALTITUDE_L4_NUMERIC_TWEAK,
)


CACHE_VERSION = 1


# Per-altitude capacity. Higher altitude → larger cache because L1 evidence
# (region/universe swaps) transfers more reliably across panels than L4
# numeric tweaks.
CAPACITY: dict[str, int] = {
    ALTITUDE_L1_REGION_UNIVERSE: 64,
    ALTITUDE_L2_OP_FAMILY: 48,
    ALTITUDE_L3_SLOT_PARAM: 32,
    ALTITUDE_L4_NUMERIC_TWEAK: 16,
}

# Per-altitude TTL in seconds. Mirrors CAPACITY ordering.
TTL_SECONDS: dict[str, int] = {
    ALTITUDE_L1_REGION_UNIVERSE: 7 * 24 * 3600,
    ALTITUDE_L2_OP_FAMILY: 5 * 24 * 3600,
    ALTITUDE_L3_SLOT_PARAM: 3 * 24 * 3600,
    ALTITUDE_L4_NUMERIC_TWEAK: 1 * 24 * 3600,
}


SCORE_ALPHA_DELTA_U = 1.0
SCORE_BETA_SUPPORT = 0.2
SCORE_GAMMA_RECENCY = 0.1
SCORE_DELTA_CONFLICT = 0.5


@dataclass
class PheromoneLink:
    """One entry in the shared cache.

    ``support`` and ``conflicts`` start at 0 and are incremented by callers
    when the cache is re-used or contradicted; storing them here lets the
    score evolve over the colony's lifetime.
    """

    altitude: str
    expr: str
    alpha_id: Optional[str]
    region: str
    universe: str
    sharpe: Optional[float]
    fitness: Optional[float]
    turnover: Optional[float]
    delta_U: Optional[float]
    evidence_type: str
    parent_alpha_id: Optional[str]
    source_panel_tag: str
    ts: float
    support: int = 0
    conflicts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "altitude": self.altitude,
            "expr": self.expr,
            "alpha_id": self.alpha_id,
            "region": self.region,
            "universe": self.universe,
            "sharpe": self.sharpe,
            "fitness": self.fitness,
            "turnover": self.turnover,
            "delta_U": self.delta_U,
            "evidence_type": self.evidence_type,
            "parent_alpha_id": self.parent_alpha_id,
            "source_panel_tag": self.source_panel_tag,
            "ts": self.ts,
            "support": self.support,
            "conflicts": self.conflicts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PheromoneLink":
        return cls(
            altitude=data.get("altitude") or "",
            expr=data.get("expr") or "",
            alpha_id=data.get("alpha_id"),
            region=data.get("region") or "",
            universe=data.get("universe") or "",
            sharpe=data.get("sharpe"),
            fitness=data.get("fitness"),
            turnover=data.get("turnover"),
            delta_U=data.get("delta_U"),
            evidence_type=data.get("evidence_type") or "",
            parent_alpha_id=data.get("parent_alpha_id"),
            source_panel_tag=data.get("source_panel_tag") or "",
            ts=float(data.get("ts") or 0.0),
            support=int(data.get("support") or 0),
            conflicts=int(data.get("conflicts") or 0),
        )


def cache_path(colony_tag: str) -> Path:
    """Return the canonical shared-cache JSON path for a colony tag."""
    return wq_brain_root() / "colony" / colony_tag / "shared_cache.json"


def _now() -> float:
    return time.time()


def _recency(ttl_s: int, ts: float, *, now: float) -> float:
    """Exponential decay with half-life = TTL/2, clamped to [0, 1]."""
    if ttl_s <= 0:
        return 0.0
    age = max(now - ts, 0.0)
    if age >= ttl_s:
        return 0.0
    return math.exp(-age / max(ttl_s / 2.0, 1.0))


def score(link: PheromoneLink, *, now: Optional[float] = None) -> float:
    """Compute the lexicographic score used at eviction time."""
    now_ts = now if now is not None else _now()
    ttl = TTL_SECONDS.get(link.altitude, TTL_SECONDS[ALTITUDE_L4_NUMERIC_TWEAK])
    return (
        SCORE_ALPHA_DELTA_U * float(link.delta_U or 0.0)
        + SCORE_BETA_SUPPORT * float(link.support)
        + SCORE_GAMMA_RECENCY * _recency(ttl, link.ts, now=now_ts)
        - SCORE_DELTA_CONFLICT * float(link.conflicts)
    )


def _empty_buckets() -> dict[str, list[dict[str, Any]]]:
    return {alt: [] for alt in CAPACITY}


@contextmanager
def _locked_open(path: Path):
    """Open ``path`` for read-write with an exclusive flock.

    Creates the file (and parent dirs) when missing so first-write callers
    do not need a separate ``mkdir``/``touch`` step.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``a+`` opens for read/write, creating if missing, without truncating.
    f = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield f
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()


def _load(handle) -> dict[str, Any]:
    handle.seek(0)
    raw = handle.read().strip()
    if not raw:
        return {"version": CACHE_VERSION, "updated_at": 0.0, "by_altitude": _empty_buckets()}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"version": CACHE_VERSION, "updated_at": 0.0, "by_altitude": _empty_buckets()}
    # Backfill missing buckets to keep the schema stable across upgrades.
    buckets = data.get("by_altitude") or {}
    for alt in CAPACITY:
        buckets.setdefault(alt, [])
    data["by_altitude"] = buckets
    return data


def _dump(handle, data: dict[str, Any]) -> None:
    handle.seek(0)
    handle.truncate()
    json.dump(data, handle, indent=2, default=str)
    handle.flush()
    os.fsync(handle.fileno())


def _evict_to_capacity(
    rows: list[dict[str, Any]],
    *,
    altitude: str,
    now: float,
) -> list[dict[str, Any]]:
    """Apply TTL + capacity to a single altitude bucket.

    Steps:
      1. Drop rows whose ``ts`` is older than TTL.
      2. If still over capacity, drop lowest-score rows.
    """
    ttl = TTL_SECONDS.get(altitude, TTL_SECONDS[ALTITUDE_L4_NUMERIC_TWEAK])
    capacity = CAPACITY.get(altitude, 0)
    fresh: list[dict[str, Any]] = []
    for row in rows:
        ts = float(row.get("ts") or 0.0)
        if (now - ts) > ttl:
            continue
        fresh.append(row)
    if capacity <= 0 or len(fresh) <= capacity:
        return fresh
    # Stable sort with score descending, then ts descending, then string id.
    fresh.sort(
        key=lambda r: (
            -score(PheromoneLink.from_dict(r), now=now),
            -float(r.get("ts") or 0.0),
            (r.get("alpha_id") or ""),
        )
    )
    return fresh[:capacity]


def read_cache(colony_tag: str) -> dict[str, list[PheromoneLink]]:
    """Read the shared cache for a colony, applying TTL but no eviction."""
    path = cache_path(colony_tag)
    if not path.exists():
        return {alt: [] for alt in CAPACITY}
    with _locked_open(path) as handle:
        data = _load(handle)
    now = _now()
    out: dict[str, list[PheromoneLink]] = {alt: [] for alt in CAPACITY}
    for alt, rows in data["by_altitude"].items():
        if alt not in out:
            continue
        ttl = TTL_SECONDS.get(alt, TTL_SECONDS[ALTITUDE_L4_NUMERIC_TWEAK])
        for row in rows:
            ts = float(row.get("ts") or 0.0)
            if (now - ts) > ttl:
                continue
            out[alt].append(PheromoneLink.from_dict(row))
    return out


def write_links(
    colony_tag: str, links: Iterable[PheromoneLink]
) -> dict[str, int]:
    """Merge ``links`` into the cache, deduplicating by expr per altitude.

    Returns a ``{altitude: kept_count}`` summary after TTL+capacity eviction.
    When an existing entry shares an expression with a newcomer:

      * support is incremented by 1 (the link "fired" again);
      * the newer row's fitness/sharpe replace the stored ones only when
        either is None on the stored row (so we never overwrite verified
        metrics with worse fresh ones from a different panel).
    """
    path = cache_path(colony_tag)
    with _locked_open(path) as handle:
        data = _load(handle)
        for link in links:
            alt = link.altitude
            if alt not in data["by_altitude"]:
                continue
            bucket = data["by_altitude"][alt]
            replaced = False
            for stored in bucket:
                if (stored.get("expr") or "") == link.expr:
                    stored["support"] = int(stored.get("support") or 0) + 1
                    if stored.get("sharpe") is None:
                        stored["sharpe"] = link.sharpe
                    if stored.get("fitness") is None:
                        stored["fitness"] = link.fitness
                    stored["ts"] = max(
                        float(stored.get("ts") or 0.0), float(link.ts)
                    )
                    replaced = True
                    break
            if not replaced:
                bucket.append(link.to_dict())
        now = _now()
        kept: dict[str, int] = {}
        for alt, bucket in data["by_altitude"].items():
            evicted = _evict_to_capacity(bucket, altitude=alt, now=now)
            data["by_altitude"][alt] = evicted
            kept[alt] = len(evicted)
        data["updated_at"] = now
        data["version"] = CACHE_VERSION
        _dump(handle, data)
    return kept


def record_conflict(colony_tag: str, alpha_id: str) -> bool:
    """Increment ``conflicts`` for the link with the given alpha_id.

    Returns True when the link was found, False otherwise.
    """
    path = cache_path(colony_tag)
    with _locked_open(path) as handle:
        data = _load(handle)
        for bucket in data["by_altitude"].values():
            for stored in bucket:
                if (stored.get("alpha_id") or "") == alpha_id:
                    stored["conflicts"] = int(stored.get("conflicts") or 0) + 1
                    data["updated_at"] = _now()
                    _dump(handle, data)
                    return True
    return False


def top_links(
    colony_tag: str,
    *,
    altitudes: Optional[Iterable[str]] = None,
    per_altitude: int = 8,
) -> list[PheromoneLink]:
    """Return the top-N links per altitude ranked by score, newest first on ties."""
    cache = read_cache(colony_tag)
    if altitudes is None:
        altitudes = CAPACITY.keys()
    now = _now()
    out: list[PheromoneLink] = []
    for alt in altitudes:
        bucket = cache.get(alt) or []
        ranked = sorted(
            bucket,
            key=lambda link: (-score(link, now=now), -link.ts, link.alpha_id or ""),
        )
        out.extend(ranked[:per_altitude])
    return out
