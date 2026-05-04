"""dtypes round-trip + behaviour tests."""
from __future__ import annotations

from agent_market.wq_brain.dtypes import (
    AlphaCandidate,
    AlphaPoolEntry,
    AlphaSettings,
    SimulationResult,
)


def test_alpha_settings_to_api_dict_uses_camel_case():
    s = AlphaSettings(region="USA", universe="TOP3000", decay=6)
    d = s.to_api_dict()
    assert d["instrumentType"] == "EQUITY"
    assert d["region"] == "USA"
    assert d["universe"] == "TOP3000"
    assert d["decay"] == 6
    assert d["unitHandling"] == "VERIFY"
    assert d["nanHandling"] == "ON"
    assert d["language"] == "FASTEXPR"
    assert d["visualization"] is False


def test_simulation_passes_quality_default_thresholds():
    r = SimulationResult(sharpe=1.30, fitness=1.05)
    assert r.passes_quality() is True


def test_simulation_passes_quality_below_sharpe():
    r = SimulationResult(sharpe=1.20, fitness=1.10)
    assert r.passes_quality() is False


def test_simulation_passes_quality_below_fitness():
    r = SimulationResult(sharpe=1.50, fitness=0.95)
    assert r.passes_quality() is False


def test_simulation_passes_quality_none_values():
    assert SimulationResult().passes_quality() is False
    assert SimulationResult(sharpe=2.0).passes_quality() is False


def test_simulation_failed_check_names():
    r = SimulationResult(checks=[
        {"name": "LOW_SHARPE", "result": "FAIL"},
        {"name": "HIGH_TURNOVER", "result": "PASS"},
        {"name": "CORRELATION", "result": "FAIL"},
    ])
    assert set(r.failed_check_names()) == {"LOW_SHARPE", "CORRELATION"}


def test_alpha_candidate_to_dict_has_source_and_settings():
    s = AlphaSettings()
    c = AlphaCandidate(expr="rank(close)", settings=s, source="scan")
    d = c.to_dict()
    assert d["expr"] == "rank(close)"
    assert d["source"] == "scan"
    assert d["settings"]["region"] == "USA"
    assert "sim_result" not in d


def test_alpha_pool_entry_round_trip():
    e = AlphaPoolEntry(
        alpha_id="A1", expr="rank(close)", settings_dict={"region": "USA"},
        sharpe=1.5, fitness=1.2, returns=0.15, turnover=0.20, tag="t1", source="scan",
    )
    d = e.to_dict()
    e2 = AlphaPoolEntry.from_dict(d)
    assert e2.alpha_id == "A1"
    assert e2.sharpe == 1.5
    assert e2.tag == "t1"
    assert e2.source == "scan"


def test_alpha_pool_entry_from_dict_ignores_unknown_fields():
    d = {
        "alpha_id": "A1", "expr": "rank(close)", "settings_dict": {},
        "sharpe": 1.0, "fitness": 1.0, "returns": 0.1, "turnover": 0.1,
        "extra_field_that_does_not_exist": "ignored",
    }
    e = AlphaPoolEntry.from_dict(d)
    assert e.alpha_id == "A1"
