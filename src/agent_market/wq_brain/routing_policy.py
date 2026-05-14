"""Tiny learned routing policy μ_θ + offline trainer.

Wraps a multinomial-logistic-style scorer over four actions {stay, deeper,
bubble_up, jump_root} and an offline trainer that consumes telemetry +
tried_log rows. The intent is *not* to win a benchmark; it's to give the
colony controller a learnable knob to defer to when its rule-based heuristic
is uncertain. The KL-regularised PG variant lives in the future — this MVP
keeps the surface area small enough that the routing.decide() rule remains
the trusted fallback.

Feature vector (6 dims, all bounded to keep the optimiser stable):

  * ``last_delta_U`` clipped to [-2, 2]
  * ``mean_recent_delta_U`` clipped to [-2, 2]
  * ``stall_count`` clipped to [0, 5]
  * ``osc_count`` clipped to [0, 5]
  * ``cross_panel_conflict`` clipped to [0, 1]
  * ``altitude_rank`` ∈ {0, 1, 2, 3} (L4→L1 ordinal)
  * (bias is added implicitly inside the scorer)

Weights file shape::

    {
      "version": 1,
      "feature_names": [...],
      "actions": ["stay", "deeper", "bubble_up", "jump_root"],
      "weights": {action: {feature: value, ..., "_bias": value}},
      "trained_at": <ts>,
      "training_samples": <n>,
      "training_epochs": <n>,
      "training_lr": <lr>
    }
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .routing import (
    ACTION_BUBBLE_UP,
    ACTION_DEEPER,
    ACTION_JUMP_ROOT,
    ACTION_STAY,
    RoutingDecision,
    RoutingState,
    RoutingThresholds,
    decide,
)
from .tried_log import (
    ALTITUDE_L1_REGION_UNIVERSE,
    ALTITUDE_L2_OP_FAMILY,
    ALTITUDE_L3_SLOT_PARAM,
    ALTITUDE_L4_NUMERIC_TWEAK,
)


POLICY_VERSION = 1

ACTIONS: tuple[str, ...] = (
    ACTION_STAY,
    ACTION_DEEPER,
    ACTION_BUBBLE_UP,
    ACTION_JUMP_ROOT,
)

FEATURE_NAMES: tuple[str, ...] = (
    "last_delta_U",
    "mean_recent_delta_U",
    "stall_count",
    "osc_count",
    "cross_panel_conflict",
    "altitude_rank",
)

_ALTITUDE_RANK: dict[str, int] = {
    ALTITUDE_L4_NUMERIC_TWEAK: 0,
    ALTITUDE_L3_SLOT_PARAM: 1,
    ALTITUDE_L2_OP_FAMILY: 2,
    ALTITUDE_L1_REGION_UNIVERSE: 3,
}


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def featurise(state: RoutingState) -> dict[str, float]:
    """Map a :class:`RoutingState` to a feature dict for the scorer."""
    return {
        "last_delta_U": _clip(float(state.last_delta_U), -2.0, 2.0),
        "mean_recent_delta_U": _clip(float(state.mean_recent_delta_U), -2.0, 2.0),
        "stall_count": _clip(float(state.stall_count), 0.0, 5.0),
        "osc_count": _clip(float(state.osc_count), 0.0, 5.0),
        "cross_panel_conflict": _clip(float(state.cross_panel_conflict), 0.0, 1.0),
        "altitude_rank": float(_ALTITUDE_RANK.get(state.current_altitude, 1)),
    }


@dataclass
class LearnedPolicy:
    weights: dict[str, dict[str, float]] = field(default_factory=dict)
    trained_at: float = 0.0
    training_samples: int = 0
    training_epochs: int = 0
    training_lr: float = 0.0

    @classmethod
    def empty(cls) -> "LearnedPolicy":
        weights = {
            act: {f: 0.0 for f in FEATURE_NAMES} | {"_bias": 0.0}
            for act in ACTIONS
        }
        return cls(weights=weights)

    def _score(self, action: str, feats: dict[str, float]) -> float:
        w = self.weights.get(action) or {}
        s = float(w.get("_bias", 0.0))
        for f, v in feats.items():
            s += float(w.get(f, 0.0)) * v
        return s

    def predict(
        self, state: RoutingState
    ) -> tuple[str, float, dict[str, float]]:
        """Return (best_action, margin, all_scores).

        ``margin`` = best score − second-best score. Callers gate fallback
        on this — when margin is small the policy is uncertain.
        """
        feats = featurise(state)
        scores = {a: self._score(a, feats) for a in ACTIONS}
        ranked = sorted(scores.values(), reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) >= 2 else 0.0
        best = max(scores, key=scores.get)
        return best, margin, scores

    # ── persistence ────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "version": POLICY_VERSION,
                "feature_names": list(FEATURE_NAMES),
                "actions": list(ACTIONS),
                "weights": self.weights,
                "trained_at": self.trained_at,
                "training_samples": self.training_samples,
                "training_epochs": self.training_epochs,
                "training_lr": self.training_lr,
            }, indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Optional["LearnedPolicy"]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        weights = data.get("weights") or {}
        if not weights:
            return None
        return cls(
            weights=weights,
            trained_at=float(data.get("trained_at") or 0.0),
            training_samples=int(data.get("training_samples") or 0),
            training_epochs=int(data.get("training_epochs") or 0),
            training_lr=float(data.get("training_lr") or 0.0),
        )

    # ── training ───────────────────────────────────────────────────────
    def train(
        self,
        samples: Sequence[tuple[RoutingState, str, float]],
        *,
        epochs: int = 20,
        lr: float = 0.1,
    ) -> "LearnedPolicy":
        """Train via reward-weighted softmax cross-entropy.

        For each sample ``(state, action_taken, reward)`` we treat
        ``action_taken`` as the target label and weight its log-likelihood
        by ``reward`` (clipped to [-1, 1]). Negative-reward samples get a
        gradient *away* from the taken action by negating the weight.

        Tiny dataset, tiny model — no regularisation, no early stopping.
        """
        if not samples:
            return self
        if not self.weights:
            empty = self.empty()
            self.weights = empty.weights
        clipped_samples = [
            (state, action, _clip(float(reward), -1.0, 1.0))
            for state, action, reward in samples
            if action in ACTIONS
        ]
        if not clipped_samples:
            return self
        for _ in range(epochs):
            for state, action, weight in clipped_samples:
                feats = featurise(state)
                scores = {a: self._score(a, feats) for a in ACTIONS}
                max_s = max(scores.values())
                exp = {a: math.exp(scores[a] - max_s) for a in ACTIONS}
                total = sum(exp.values()) or 1.0
                probs = {a: exp[a] / total for a in ACTIONS}
                # Gradient of weighted NLL w.r.t. score(a):
                #   ∂L/∂s_a = weight * (probs[a] - 1[a==action])
                # SGD step uses the *negated* gradient times lr.
                for a in ACTIONS:
                    target = 1.0 if a == action else 0.0
                    grad = weight * (probs[a] - target)
                    w = self.weights[a]
                    w["_bias"] = float(w.get("_bias", 0.0)) - lr * grad
                    for f, v in feats.items():
                        w[f] = float(w.get(f, 0.0)) - lr * grad * v
        self.training_samples = len(clipped_samples)
        self.training_epochs = int(epochs)
        self.training_lr = float(lr)
        self.trained_at = time.time()
        return self


def policy_path(colony_tag: str) -> Path:
    """Where ``<colony_tag>``'s learned routing weights live on disk."""
    from .paths import wq_brain_root
    return wq_brain_root() / "colony" / colony_tag / "routing_policy.json"


def hybrid_decide(
    state: RoutingState,
    *,
    policy: Optional[LearnedPolicy] = None,
    thresholds: RoutingThresholds = RoutingThresholds(),
    margin_min: float = 0.10,
) -> RoutingDecision:
    """Decide using the learned policy when it's confident, else rules.

    Falls back to :func:`routing.decide` whenever the policy is missing,
    untrained (training_samples == 0), or its top-2 score margin is below
    ``margin_min``. Cross-panel conflict and global diagnosis always defer
    to the rule path so safety-critical escalations are never overridden.
    """
    # Hard safety rules trump the learned policy.
    if (
        state.cross_panel_conflict > thresholds.tau_root
        or state.diagnosis_scope == "global"
    ):
        return decide(state, thresholds)
    if policy is None or policy.training_samples == 0:
        return decide(state, thresholds)
    action, margin, scores = policy.predict(state)
    if margin < margin_min:
        return decide(state, thresholds)
    target = _altitude_for_action(action, state.current_altitude)
    rationale = (
        f"Learned policy μ_θ chose **{action}** with margin {margin:.3f} "
        f"(scores={ {a: round(s, 3) for a, s in scores.items()} })."
    )
    inputs = {
        "policy_action": action,
        "policy_margin": round(margin, 4),
        "policy_scores": {a: round(s, 4) for a, s in scores.items()},
        "current_altitude": state.current_altitude,
    }
    return RoutingDecision(
        action=action,
        target_altitude=target,
        rationale=rationale,
        inputs=inputs,
    )


def _altitude_for_action(action: str, current: str) -> str:
    """Mirror routing._next_altitude_after_*. Kept local to avoid imports cycles."""
    from .routing import _next_altitude_after_bubble, _next_altitude_after_deeper
    if action == ACTION_BUBBLE_UP:
        return _next_altitude_after_bubble(current)
    if action == ACTION_DEEPER:
        return _next_altitude_after_deeper(current)
    if action == ACTION_JUMP_ROOT:
        return ALTITUDE_L1_REGION_UNIVERSE
    return current


def samples_from_history(
    rows: Iterable[dict],
    *,
    diagnosis_scope: str = "local",
    cross_panel_conflict: float = 0.0,
) -> list[tuple[RoutingState, str, float]]:
    """Reconstruct (state, action_taken, reward) tuples from tried_log rows.

    The history must contain the new pheromone metadata (``altitude`` and
    ``delta_U``); legacy rows are ignored. The "action taken" is inferred
    from the altitude transition between consecutive rows; the reward is
    the *next* row's ``delta_U`` (i.e. the gain produced by the action).
    """
    from .routing import _next_altitude_after_bubble, _next_altitude_after_deeper

    samples: list[tuple[RoutingState, str, float]] = []
    enriched = [r for r in rows if r.get("altitude")]
    enriched.sort(key=lambda r: float(r.get("ts") or 0.0))
    for prev, curr in zip(enriched, enriched[1:]):
        prev_alt = prev.get("altitude") or ""
        curr_alt = curr.get("altitude") or ""
        if not prev_alt or not curr_alt:
            continue
        if curr_alt == prev_alt:
            action = ACTION_STAY
        elif _next_altitude_after_bubble(prev_alt) == curr_alt:
            action = ACTION_BUBBLE_UP
        elif _next_altitude_after_deeper(prev_alt) == curr_alt:
            action = ACTION_DEEPER
        else:
            action = ACTION_JUMP_ROOT
        reward = float(curr.get("delta_U") or 0.0)
        # Re-build a RoutingState as if we were about to act at `prev`.
        state = RoutingState(
            last_delta_U=float(prev.get("delta_U") or 0.0),
            mean_recent_delta_U=float(prev.get("delta_U") or 0.0),
            stall_count=0,
            osc_count=0,
            cross_panel_conflict=cross_panel_conflict,
            diagnosis_scope=diagnosis_scope,
            current_altitude=prev_alt,
        )
        samples.append((state, action, reward))
    return samples
