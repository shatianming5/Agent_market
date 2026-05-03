"""Tests for wq_brain.client — mocked HTTP, no real WQ account needed."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent_market.wq_brain.client import WQSession
from agent_market.wq_brain.dtypes import AlphaCandidate, AlphaSettings, SimulationResult


@pytest.fixture
def sess():
    s = WQSession("test@example.com", "password123")
    return s


def _mock_response(status_code: int = 200, json_data: dict = None, headers: dict = None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = headers or {}
    resp.text = json.dumps(json_data or {})
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestLogin:
    def test_login_success(self, sess):
        with patch("requests.Session.post", return_value=_mock_response(200)):
            sess.login()
        assert sess._logged_in

    def test_login_wrong_credentials(self, sess):
        with patch("requests.Session.post", return_value=_mock_response(401)):
            with pytest.raises(PermissionError, match="invalid credentials"):
                sess.login()
        assert not sess._logged_in


class TestSubmitSimulation:
    def test_submit_returns_sim_url(self, sess):
        sess._logged_in = True
        sim_url = "https://api.worldquantbrain.com/simulations/abc123"
        mock_resp = _mock_response(
            201,
            json_data={"id": "abc123"},
            headers={"Location": sim_url},
        )
        settings = AlphaSettings()
        with patch("requests.Session.request", return_value=mock_resp):
            url = sess.submit_simulation("rank(close)", settings)
        assert url == sim_url

    def test_submit_no_location_raises(self, sess):
        sess._logged_in = True
        mock_resp = _mock_response(201, json_data={}, headers={})
        settings = AlphaSettings()
        with patch("requests.Session.request", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="No simulation URL"):
                sess.submit_simulation("rank(close)", settings)


class TestPollSimulation:
    def test_poll_returns_when_complete(self, sess):
        sess._logged_in = True
        complete_data = {
            "status": "COMPLETE",
            "id": "abc123",
            "is": {"sharpe": 1.8, "fitness": 1.5, "returns": 0.12, "turnover": 0.4},
        }
        with patch("requests.Session.request", return_value=_mock_response(200, complete_data)):
            result = sess.poll_simulation("https://api.worldquantbrain.com/simulations/abc123")
        assert result["status"] == "COMPLETE"

    def test_poll_timeout(self, sess):
        sess._logged_in = True
        running_data = {"status": "RUNNING"}
        with patch("requests.Session.request", return_value=_mock_response(200, running_data)):
            with pytest.raises(TimeoutError):
                sess.poll_simulation("https://sim_url", timeout=0.1)


class TestParseSimResult:
    def test_parse_complete(self, sess):
        raw = {
            "status": "COMPLETE",
            "id": "abc123",
            "is": {
                "sharpe": 1.9,
                "fitness": 1.6,
                "returns": 0.15,
                "turnover": 0.45,
                "drawdown": 0.08,
            },
        }
        result = sess.parse_sim_result(raw)
        assert result.sharpe == pytest.approx(1.9)
        assert result.fitness == pytest.approx(1.6)
        assert result.alpha_id == "abc123"
        assert result.passes_quality

    def test_parse_low_sharpe_fails_quality(self, sess):
        raw = {"status": "COMPLETE", "id": "x", "is": {"sharpe": 0.8, "fitness": 0.5}}
        result = sess.parse_sim_result(raw)
        assert not result.passes_quality

    def test_parse_error_status(self, sess):
        raw = {"status": "ERROR", "error": "expression syntax error"}
        result = sess.parse_sim_result(raw)
        assert result.status == "ERROR"
        assert result.error == "expression syntax error"
        assert not result.passes_quality


class TestThreadSafety:
    def test_login_called_once_under_concurrent_ensure_login(self, sess):
        import threading

        login_calls = []

        def _mock_login():
            login_calls.append(1)
            sess._logged_in = True

        sess.login = _mock_login

        threads = [threading.Thread(target=sess._ensure_login) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(login_calls) == 1

    def test_global_sem_limits_batch_concurrency(self, sess):
        import threading
        import time as _time

        from agent_market.wq_brain.dtypes import AlphaCandidate, AlphaSettings, SimulationResult

        max_c = 2
        sess._logged_in = True
        sess._global_sem = threading.Semaphore(max_c)
        sess._max_concurrent = max_c

        active = [0]
        peak = [0]
        lock = threading.Lock()

        def mock_sim(expr, settings, **kw):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            _time.sleep(0.02)
            with lock:
                active[0] -= 1
            return SimulationResult(status="COMPLETE", alpha_id="x")

        candidates = [
            AlphaCandidate(expr=f"rank(close_{i})", settings=AlphaSettings()) for i in range(6)
        ]

        with patch.object(sess, "simulate_and_parse", side_effect=mock_sim):
            with patch("agent_market.wq_brain.client.time.sleep"):  # skip 1s rate-limit sleep
                sess.batch_simulate(candidates)

        assert peak[0] <= max_c


class TestRetry:
    def test_retries_on_401_and_relogins(self, sess):
        sess._logged_in = True
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_response(401)
            return _mock_response(200, {"ok": True})

        with patch("requests.Session.request", side_effect=side_effect):
            with patch.object(sess, "login") as mock_login:
                resp = sess._request_with_retry("GET", "https://x")
        mock_login.assert_called()
        assert resp.status_code == 200
