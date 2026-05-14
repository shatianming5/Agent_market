"""Per-panel "best-so-far" tracker for the colony.

PheroViz Proposition 1 (best-so-far monotonicity) says the colony must
maintain ``x_t^\\star = argmax_{τ≤t} U_τ`` so the returned artifact is the
highest-utility one seen, not the most recent. We implement this as a tiny
JSON file ``<wq_brain_root>/colony/<colony_tag>/best_so_far/<panel_tag>.json``
that's updated after every simulate call. The file is the only place where
``utility_score`` is recomputed at decision time; downstream tooling
(``colony status`` CLI, telemetry, prompt_builder) just reads it.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from .paths import wq_brain_root
from .tried_log import utility_score


@dataclass(frozen=True)
class BestSoFar:
    panel_tag: str
    alpha_id: Optional[str]
    expr: Optional[str]
    sharpe: Optional[float]
    fitness: Optional[float]
    turnover: Optional[float]
    delta_U: Optional[float]
    utility: float
    ts: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def best_so_far_path(colony_tag: str, panel_tag: str) -> Path:
    """Path of the panel's best-so-far JSON inside a colony run dir."""
    return (
        wq_brain_root()
        / "colony"
        / colony_tag
        / "best_so_far"
        / f"{panel_tag}.json"
    )


def read_best_so_far(colony_tag: str, panel_tag: str) -> Optional[BestSoFar]:
    path = best_so_far_path(colony_tag, panel_tag)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return BestSoFar(**data)
    except TypeError:
        return None


def write_best_so_far(record: BestSoFar) -> Path:
    """Persist a best-so-far record. Caller must ensure monotonicity."""
    path = best_so_far_path("", record.panel_tag)  # placeholder, overwritten below
    raise NotImplementedError(
        "Use update_best_so_far instead — write_best_so_far is reserved "
        "for future direct-rewrite scenarios."
    )


def update_best_so_far(
    colony_tag: str,
    panel_tag: str,
    *,
    alpha_id: Optional[str],
    expr: Optional[str],
    sharpe: Optional[float],
    fitness: Optional[float],
    turnover: Optional[float],
    delta_U: Optional[float],
    ts: Optional[float] = None,
) -> tuple[BestSoFar, bool]:
    """Compute candidate utility, compare against stored best, write on win.

    Returns ``(record_after_call, was_updated)``. ``record_after_call`` is
    the (possibly unchanged) best record after the call; ``was_updated``
    is True when the candidate beat the prior best and the file was
    rewritten. Missing sharpe/fitness count as utility=0.0 so unverified
    candidates never claim the crown.
    """
    if ts is None:
        ts = time.time()
    candidate_utility = utility_score(
        sharpe=sharpe, fitness=fitness, turnover=turnover
    )
    candidate = BestSoFar(
        panel_tag=panel_tag,
        alpha_id=alpha_id,
        expr=expr,
        sharpe=sharpe,
        fitness=fitness,
        turnover=turnover,
        delta_U=delta_U,
        utility=candidate_utility,
        ts=ts,
    )
    current = read_best_so_far(colony_tag, panel_tag)
    if current is not None and current.utility >= candidate_utility:
        return current, False
    path = best_so_far_path(colony_tag, panel_tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(candidate.to_dict(), indent=2, default=str),
                   encoding="utf-8")
    tmp.replace(path)
    return candidate, True


def best_so_far_directory(colony_tag: str) -> Path:
    return wq_brain_root() / "colony" / colony_tag / "best_so_far"


def list_panel_bests(colony_tag: str) -> list[BestSoFar]:
    """Read every best-so-far file under a colony's directory."""
    base = best_so_far_directory(colony_tag)
    if not base.exists():
        return []
    out: list[BestSoFar] = []
    for f in sorted(base.glob("*.json")):
        record = read_best_so_far(colony_tag, f.stem)
        if record is not None:
            out.append(record)
    return out
