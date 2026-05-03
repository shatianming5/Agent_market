"""Tests for wq_brain.multiregion — mocked, no real WQ calls."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, call, patch

import pytest

from agent_market.wq_brain.multiregion import (
    MultiRegionConfig,
    MultiRegionRunner,
    RegionSpec,
    parse_regions,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_session():
    sess = MagicMock(spec=[
        "_logged_in", "_login_lock", "_global_sem", "_max_concurrent",
        "login", "batch_simulate", "submit_simulation", "poll_simulation",
    ])
    sess._logged_in = True
    sess._login_lock = threading.Lock()
    sess._global_sem = threading.Semaphore(100)  # large — tests don't want blocking
    sess._max_concurrent = 100
    return sess


# ── RegionSpec ───────────────────────────────────────────────────────────────


class TestRegionSpec:
    def test_from_str_region_only(self):
        spec = RegionSpec.from_str("CHN")
        assert spec.region == "CHN"
        assert spec.universe == "TOP2000"
        assert spec.neutralization == "INDUSTRY"

    def test_from_str_region_universe(self):
        spec = RegionSpec.from_str("USA:TOP3000")
        assert spec.region == "USA"
        assert spec.universe == "TOP3000"
        assert spec.tag_suffix == "USA_TOP3000"

    def test_from_str_full(self):
        spec = RegionSpec.from_str("IND:TOP500:MARKET")
        assert spec.neutralization == "MARKET"

    def test_tag_suffix_auto(self):
        spec = RegionSpec(region="EUR", universe="TOP1200")
        assert spec.tag_suffix == "EUR_TOP1200"

    def test_invalid_neutralization_raises(self):
        with pytest.raises(ValueError, match="Invalid neutralization"):
            RegionSpec(region="USA", universe="TOP3000", neutralization="INVALID")

    def test_unknown_region_falls_back_to_defaults(self):
        spec = RegionSpec.from_str("XYZ")
        assert spec.universe == "TOP3000"  # global default

    def test_parse_regions(self):
        specs = parse_regions("USA:TOP3000,CHN:TOP2000,IND:TOP500")
        assert len(specs) == 3
        assert [s.region for s in specs] == ["USA", "CHN", "IND"]


# ── MultiRegionConfig ────────────────────────────────────────────────────────


class TestMultiRegionConfig:
    def test_make_region_config_inherits_shared_params(self):
        config = MultiRegionConfig(
            base_tag="test",
            regions=[RegionSpec.from_str("USA:TOP3000")],
            max_iterations=7,
            batch_size=3,
            global_max_concurrent=5,
            auto_submit=True,
        )
        cfg = config.make_region_config(config.regions[0])
        assert cfg.region == "USA"
        assert cfg.universe == "TOP3000"
        assert cfg.tag == "test_USA_TOP3000"
        assert cfg.max_iterations == 7
        assert cfg.batch_size == 3
        assert cfg.max_concurrent == 5
        assert cfg.auto_submit is True

    def test_each_region_gets_unique_tag(self):
        regions = parse_regions("USA:TOP3000,CHN:TOP2000")
        config = MultiRegionConfig(base_tag="run", regions=regions)
        tags = {config.make_region_config(s).tag for s in regions}
        assert len(tags) == 2
        assert "run_USA_TOP3000" in tags
        assert "run_CHN_TOP2000" in tags


# ── MultiRegionRunner ────────────────────────────────────────────────────────


class TestMultiRegionRunner:
    def test_three_regions_all_run(self, mock_session):
        regions = parse_regions("USA:TOP3000,CHN:TOP2000,IND:TOP500")
        config = MultiRegionConfig(base_tag="smoke", regions=regions, max_iterations=1)

        with patch("agent_market.wq_brain.multiregion.WQBrainRunner") as MockRunner:
            MockRunner.return_value = MagicMock()
            runner = MultiRegionRunner(config, session=mock_session)
            result = runner.run()

        assert MockRunner.call_count == 3
        assert len(result["errors"]) == 0
        assert len(result["results"]) == 3

    def test_shared_session_injected_into_each_runner(self, mock_session):
        regions = parse_regions("USA:TOP3000,CHN:TOP2000")
        config = MultiRegionConfig(base_tag="s", regions=regions, max_iterations=1)

        sessions_seen = []

        def capture_session(cfg, session=None):
            sessions_seen.append(session)
            return MagicMock()

        with patch("agent_market.wq_brain.multiregion.WQBrainRunner", side_effect=capture_session):
            runner = MultiRegionRunner(config, session=mock_session)
            runner.run()

        assert all(s is mock_session for s in sessions_seen)

    def test_region_error_isolated_others_continue(self, mock_session):
        regions = parse_regions("USA:TOP3000,CHN:TOP2000")
        config = MultiRegionConfig(base_tag="s", regions=regions, max_iterations=1)

        call_count = [0]

        def _maybe_fail(cfg, session=None):
            inst = MagicMock()
            call_count[0] += 1
            if call_count[0] == 1:
                inst.run.side_effect = RuntimeError("region blown up")
            return inst

        with patch("agent_market.wq_brain.multiregion.WQBrainRunner", side_effect=_maybe_fail):
            runner = MultiRegionRunner(config, session=mock_session)
            result = runner.run()

        assert call_count[0] == 2
        assert len(result["errors"]) == 1
        assert len(result["results"]) == 1

    def test_global_sem_updated_before_threads(self, mock_session):
        regions = parse_regions("USA:TOP3000")
        config = MultiRegionConfig(
            base_tag="s", regions=regions, max_iterations=1, global_max_concurrent=7
        )

        with patch("agent_market.wq_brain.multiregion.WQBrainRunner") as MockRunner:
            MockRunner.return_value = MagicMock()
            runner = MultiRegionRunner(config, session=mock_session)
            runner.run()

        assert mock_session._max_concurrent == 7

    def test_login_called_once_for_shared_session(self):
        regions = parse_regions("USA:TOP3000,CHN:TOP2000")
        config = MultiRegionConfig(base_tag="s", regions=regions, max_iterations=1)

        fresh_session = MagicMock()
        fresh_session._logged_in = False
        fresh_session._login_lock = threading.Lock()
        fresh_session._global_sem = threading.Semaphore(100)
        fresh_session._max_concurrent = 100

        with patch("agent_market.wq_brain.multiregion.session_from_env", return_value=fresh_session):
            with patch("agent_market.wq_brain.multiregion.WQBrainRunner") as MockRunner:
                MockRunner.return_value = MagicMock()
                runner = MultiRegionRunner(config)  # no session injected — should build one
                runner.run()

        fresh_session.login.assert_called_once()
