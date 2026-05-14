"""Hard pre-simulate gates used by the colony controller.

These complement the prompt-level advisories already in ``prompt_builder``.
Where ``prompt_builder`` writes natural-language hints into the agent's
system prompt, the helpers here return rejection reasons that ``simulate``
honours by refusing to call WorldQuant BRAIN.

The two gates currently exposed:

  * :func:`slot_cooldown_block` — implements PheroViz Sec. 4.7's
    oscillation guard. When the same *slot* (operator skeleton + field
    set) was just tried with low utility delta, refuse a re-try unless the
    predicted ΔU exceeds ``delta_flip``.
  * :func:`lex_resolve_conflicts` — applies the lexicographic precedence
    key ``Π(ℓ) = (hard, h, exec, support, ΔU, t, -id)`` to collapse a list
    of cache links so only one winner per slot remains.

Slot keying reuses ``prompt_builder._slot_key`` so the hard gate and the
prompt-level cool-down hint agree on what "same slot" means.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from .pheromone_cache import PheromoneLink, score
from .prompt_builder import _slot_key
from .tried_log import (
    ALTITUDE_L1_REGION_UNIVERSE,
    ALTITUDE_L2_OP_FAMILY,
    ALTITUDE_L3_SLOT_PARAM,
    ALTITUDE_L4_NUMERIC_TWEAK,
    utility_score,
)


# Larger numeric value = higher altitude precedence in the lex key.
_ALTITUDE_RANK: dict[str, int] = {
    ALTITUDE_L1_REGION_UNIVERSE: 4,
    ALTITUDE_L2_OP_FAMILY: 3,
    ALTITUDE_L3_SLOT_PARAM: 2,
    ALTITUDE_L4_NUMERIC_TWEAK: 1,
}


# Evidence types that count as ``hard`` constraints in the precedence key —
# these reflect cross-cutting rules the colony must respect rather than soft
# style suggestions.
_HARD_EVIDENCE_TYPES: frozenset[str] = frozenset(
    {"region_swap", "neutralization_swap", "manual"}
)


# Evidence types whose source went through the live WorldQuant simulator
# (i.e. ``exec`` is verified). Used to break ties — verified evidence
# dominates purely judge-preferred edits.
_EXECUTED_EVIDENCE_TYPES: frozenset[str] = frozenset(
    {"seed", "mutation", "crossover", "op_swap", "param_shift", "decay_shift",
     "numeric_tweak", "manual", "colony_shared"}
)


@dataclass(frozen=True)
class CooldownBlock:
    """Returned by :func:`slot_cooldown_block` when a slot is on cool-down."""

    reason: str
    last_seen_alpha_id: Optional[str]
    last_delta_U: Optional[float]
    last_ts: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "last_seen_alpha_id": self.last_seen_alpha_id,
            "last_delta_U": self.last_delta_U,
            "last_ts": self.last_ts,
        }


def slot_cooldown_block(
    expr: str,
    tried_rows: Sequence[dict[str, Any]],
    *,
    window: int = 8,
    delta_flip: float = 0.05,
    now: Optional[float] = None,
) -> Optional[CooldownBlock]:
    """Return a block reason when ``expr``'s slot is on cool-down.

    Cool-down fires when the same slot was tried in the last ``window``
    rows AND the most recent attempt's ΔU is below ``delta_flip``. Callers
    pass the predicted ΔU as a soft override via the row metadata; if the
    most recent row's ΔU exceeds ``delta_flip``, the cool-down is lifted
    (the slot is "earning its keep").
    """
    if now is None:
        now = time.time()
    target = _slot_key(expr)
    if target is None:
        return None
    ordered = sorted(tried_rows, key=lambda r: float(r.get("ts") or 0.0))
    recent = ordered[-window:]
    last_match: Optional[dict[str, Any]] = None
    for row in reversed(recent):
        row_expr = row.get("expr") or ""
        key = _slot_key(row_expr)
        if key == target:
            last_match = row
            break
    if last_match is None:
        return None
    du = last_match.get("delta_U")
    if du is None:
        # Backfill ΔU from utility metrics when missing.
        prev_row = next(
            (r for r in reversed(recent)
             if r is not last_match and r.get("sharpe") is not None),
            None,
        )
        if prev_row is not None and last_match.get("sharpe") is not None:
            du = (
                utility_score(
                    sharpe=last_match.get("sharpe"),
                    fitness=last_match.get("fitness"),
                    turnover=last_match.get("turnover"),
                )
                - utility_score(
                    sharpe=prev_row.get("sharpe"),
                    fitness=prev_row.get("fitness"),
                    turnover=prev_row.get("turnover"),
                )
            )
    if du is not None and float(du) > delta_flip:
        # The previous attempt at this slot earned utility — let the new
        # one in.
        return None
    return CooldownBlock(
        reason=(
            f"slot recently retried with ΔU={du!r} ≤ δ_flip={delta_flip}; "
            f"escalate altitude or wait for the cool-down window to pass"
        ),
        last_seen_alpha_id=last_match.get("alpha_id"),
        last_delta_U=float(du) if du is not None else None,
        last_ts=float(last_match.get("ts") or 0.0),
    )


def lex_precedence_key(link: PheromoneLink, *, now: Optional[float] = None):
    """Return the lexicographic precedence tuple ``Π(ℓ)``.

    Order convention: sort *descending*, so a higher tuple wins.

      ``(hard, altitude_rank, exec, support, ΔU, ts, alpha_id_desc)``

    where ``hard``/``exec`` are 0/1 ints. ``alpha_id_desc`` (the negated
    alpha id string) breaks ties deterministically.
    """
    hard = 1 if link.evidence_type in _HARD_EVIDENCE_TYPES else 0
    exec_ = 1 if link.evidence_type in _EXECUTED_EVIDENCE_TYPES else 0
    altitude_rank = _ALTITUDE_RANK.get(link.altitude, 0)
    return (
        hard,
        altitude_rank,
        exec_,
        float(link.support),
        float(link.delta_U or 0.0),
        float(link.ts),
        # ``-id`` per PheroViz — alpha_id strings are ordered descending
        # via tuple of negated codepoints to keep deterministic.
        tuple(-ord(c) for c in (link.alpha_id or "")),
    )


def lex_resolve_conflicts(
    links: Iterable[PheromoneLink],
) -> list[PheromoneLink]:
    """Collapse links sharing a slot to a single lex-winning representative.

    "Slot" is determined by :func:`prompt_builder._slot_key` so the gate
    and prompt-level cooldown agree.
    """
    by_slot: dict[Any, PheromoneLink] = {}
    for link in links:
        slot = _slot_key(link.expr) or ("__unkeyed__", link.expr)
        incumbent = by_slot.get(slot)
        if incumbent is None:
            by_slot[slot] = link
            continue
        if lex_precedence_key(link) > lex_precedence_key(incumbent):
            by_slot[slot] = link
    # Stable order: by precedence key descending, then expression for
    # readable debug output.
    return sorted(
        by_slot.values(),
        key=lambda l: (lex_precedence_key(l),),
        reverse=True,
    )
