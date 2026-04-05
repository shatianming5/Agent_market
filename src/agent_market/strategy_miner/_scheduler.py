"""Adaptive family budget allocation via Thompson Sampling."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

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

    def select_families(self, n: int) -> List[str]:
        """Select n families using Thompson Sampling.

        Each family is sampled from Beta(successes + 1, failures + 1).
        At least min_exploration_slots are reserved for random exploration.
        """
        if n <= 0:
            return []
        if n <= self.min_exploration_slots:
            return random.sample(self.families, min(n, len(self.families)))

        selected: List[str] = []

        # Reserve exploration slots
        explore_count = min(self.min_exploration_slots, n)
        explore_families = random.sample(self.families, min(explore_count, len(self.families)))
        selected.extend(explore_families)

        # Fill remaining slots via Thompson Sampling
        remaining = n - len(selected)
        for _ in range(remaining):
            scores = {}
            for f in self.families:
                s = self.stats.get(f)
                if s is None or s.trials == 0:
                    # Untried families get high exploration score
                    scores[f] = random.betavariate(1 + self.exploration_bonus, 1)
                else:
                    alpha = s.successes + 1
                    beta_param = (s.trials - s.successes) + 1
                    scores[f] = random.betavariate(alpha, beta_param)

            best = max(scores, key=scores.get)  # type: ignore[arg-type]
            selected.append(best)

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
