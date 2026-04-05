"""Tests for bandit scheduler."""
from agent_market.strategy_miner._scheduler import BanditScheduler, FamilyStats


def test_bandit_selects_n_families():
    sched = BanditScheduler(["a", "b", "c"])
    selected = sched.select_families(3)
    assert len(selected) == 3


def test_bandit_favors_successful():
    sched = BanditScheduler(["good", "bad"], exploration_bonus=0.1, min_exploration_slots=0)
    for _ in range(20):
        sched.update("good", 1.0, True)
        sched.update("bad", 0.0, False)
    # After 20 successes vs 20 failures, "good" should dominate
    counts = {"good": 0, "bad": 0}
    for _ in range(100):
        for f in sched.select_families(1):
            counts[f] += 1
    assert counts["good"] > counts["bad"]


def test_bandit_serialization():
    sched = BanditScheduler(["a", "b"])
    sched.update("a", 1.0, True)
    d = sched.to_dict()
    restored = BanditScheduler.from_dict(d)
    assert restored.stats["a"].trials == 1
    assert restored.stats["a"].successes == 1


def test_untried_families_get_explored():
    sched = BanditScheduler(["tried", "untried"], min_exploration_slots=1)
    sched.update("tried", 0.5, True)
    # With exploration slot, untried should appear sometimes
    seen_untried = False
    for _ in range(20):
        if "untried" in sched.select_families(2):
            seen_untried = True
            break
    assert seen_untried
