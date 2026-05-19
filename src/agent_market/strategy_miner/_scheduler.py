"""Adaptive family budget allocation via Thompson Sampling."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

logger = logging.getLogger(__name__)


@dataclass
class FamilyStats:
    """Track performance of a candidate family."""
    family: str
    trials: int = 0
    successes: int = 0  # passed constraints
    total_reward: float = 0.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(self.trials, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "trials": self.trials,
            "successes": self.successes,
            "total_reward": self.total_reward,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FamilyStats:
        return cls(
            family=str(d.get("family", "")),
            trials=int(d.get("trials", 0)),
            successes=int(d.get("successes", 0)),
            total_reward=float(d.get("total_reward", 0.0)),
        )


class BanditScheduler:
    """Thompson Sampling scheduler for candidate family allocation.

    Replaces fixed slot rotation with adaptive budget allocation.
    Families that produce better candidates get more slots.
    """

    def __init__(
        self,
        families: List[str],
        *,
        exploration_bonus: float = 1.0,
        min_exploration_slots: int = 1,
    ):
        self.families = list(families)
        self.exploration_bonus = exploration_bonus
        self.min_exploration_slots = min_exploration_slots
        self.stats: Dict[str, FamilyStats] = {
            f: FamilyStats(family=f) for f in families
        }

    def select_families(
        self,
        n: int,
        *,
        family_weights: Mapping[str, float] | None = None,
    ) -> List[str]:
        """Select n families using Thompson Sampling.

        Each family is sampled from Beta(successes + 1, failures + 1).
        At least min_exploration_slots are reserved for random exploration.
        """
        if n <= 0:
            return []
        weights = {
            str(f): max(0.0, float(w))
            for f, w in dict(family_weights or {}).items()
        }

        def _weighted_pool(pool: list[str]) -> list[str]:
            positive = [family for family in pool if weights.get(family, 1.0) > 0.0]
            return positive or list(pool)

        def _pick_weighted_random(pool: list[str]) -> str:
            candidates = _weighted_pool(pool)
            total = sum(weights.get(family, 1.0) for family in candidates)
            if total <= 0.0:
                return random.choice(candidates)
            target = random.random() * total
            acc = 0.0
            for family in candidates:
                acc += weights.get(family, 1.0)
                if acc >= target:
                    return family
            return candidates[-1]

        if n <= self.min_exploration_slots:
            selected: list[str] = []
            remaining = list(self.families)
            while remaining and len(selected) < n:
                choice = _pick_weighted_random(remaining)
                selected.append(choice)
                remaining = [family for family in remaining if family != choice]
            return selected

        selected: List[str] = []
        remaining_pool = list(self.families)

        # Reserve exploration slots
        explore_count = min(self.min_exploration_slots, n)
        explore_families: list[str] = []
        while remaining_pool and len(explore_families) < explore_count:
            choice = _pick_weighted_random(remaining_pool)
            explore_families.append(choice)
            remaining_pool = [family for family in remaining_pool if family != choice]
        selected.extend(explore_families)

        # Fill remaining slots via Thompson Sampling
        remaining = n - len(selected)
        for _ in range(remaining):
            if not remaining_pool:
                remaining_pool = _weighted_pool(list(self.families))
            scores = {}
            for f in _weighted_pool(remaining_pool):
                s = self.stats.get(f)
                if s is None or s.trials == 0:
                    # Untried families get high exploration score
                    scores[f] = random.betavariate(1 + self.exploration_bonus, 1)
                else:
                    alpha = s.successes + 1
                    beta_param = (s.trials - s.successes) + 1
                    scores[f] = random.betavariate(alpha, beta_param)
                scores[f] *= max(1e-6, weights.get(f, 1.0))

            best = max(scores, key=scores.get)  # type: ignore[arg-type]
            selected.append(best)
            remaining_pool = [family for family in remaining_pool if family != best]

        return selected

    def update(self, family: str, reward: float, passed: bool) -> None:
        """Update stats after candidate evaluation."""
        if family not in self.stats:
            self.stats[family] = FamilyStats(family=family)
        s = self.stats[family]
        s.trials += 1
        if passed:
            s.successes += 1
        s.total_reward += max(0.0, reward)
        logger.debug(
            "Bandit update: %s trials=%d successes=%d avg_reward=%.4f",
            family, s.trials, s.successes, s.avg_reward,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "families": self.families,
            "exploration_bonus": self.exploration_bonus,
            "min_exploration_slots": self.min_exploration_slots,
            "stats": {k: v.to_dict() for k, v in self.stats.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BanditScheduler:
        sched = cls(
            families=d.get("families", []),
            exploration_bonus=float(d.get("exploration_bonus", 1.0)),
            min_exploration_slots=int(d.get("min_exploration_slots", 1)),
        )
        for k, v in d.get("stats", {}).items():
            sched.stats[k] = FamilyStats.from_dict(v)
        return sched


def resolve_family_weights(
    schedule: list[dict[str, Any]] | None,
    *,
    iteration: int,
) -> dict[str, float]:
    """Resolve the active per-family weights for the current iteration.

    Later matching rules override earlier ones. Each rule supports:
    - iteration_gte: inclusive lower bound (default 0)
    - iteration_lte: inclusive upper bound (optional)
    - weights: {family: multiplier}
    """
    resolved: dict[str, float] = {}
    for raw_rule in list(schedule or []):
        if not isinstance(raw_rule, dict):
            continue
        try:
            iteration_gte = int(raw_rule.get("iteration_gte", 0) or 0)
        except Exception:
            iteration_gte = 0
        raw_lte = raw_rule.get("iteration_lte", None)
        try:
            iteration_lte = None if raw_lte in (None, "") else int(raw_lte)
        except Exception:
            iteration_lte = None
        if iteration < iteration_gte:
            continue
        if iteration_lte is not None and iteration > iteration_lte:
            continue
        weights = raw_rule.get("weights") or {}
        if not isinstance(weights, dict):
            continue
        for family, weight in weights.items():
            try:
                resolved[str(family)] = max(0.0, float(weight))
            except Exception:
                continue
    return resolved
