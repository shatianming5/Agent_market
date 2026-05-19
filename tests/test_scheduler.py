"""Tests for bandit scheduler."""
from agent_market.strategy_miner._scheduler import BanditScheduler


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


def test_candidate_family_serialization():
    """candidate_family survives checkpoint roundtrip."""
    from agent_market.strategy_miner.dtypes import StrategyCandidate
    from pathlib import Path

    c = StrategyCandidate("Test", "code", Path("/tmp/x.py"))
    c.candidate_family = "rule/mean-reversion"
    d = c.to_dict()
    assert d["candidate_family"] == "rule/mean-reversion"
    restored = StrategyCandidate.from_dict(d)
    assert restored.candidate_family == "rule/mean-reversion"


def test_bandit_state_in_miner_state():
    """bandit_state persists in MinerState checkpoint."""
    from agent_market.strategy_miner.dtypes import MinerState
    from agent_market.strategy_miner._scheduler import BanditScheduler

    state = MinerState()
    sched = BanditScheduler(["rule/mr", "ml/lgbm"])
    sched.update("rule/mr", 1.0, True)
    state.bandit_state = sched.to_dict()

    d = state.to_dict()
    assert "bandit_state" in d
    assert d["bandit_state"]["stats"]["rule/mr"]["successes"] == 1

    restored = MinerState.from_dict(d)
    assert restored.bandit_state["stats"]["rule/mr"]["successes"] == 1

    # Can restore scheduler from state
    restored_sched = BanditScheduler.from_dict(restored.bandit_state)
    assert restored_sched.stats["rule/mr"].successes == 1
