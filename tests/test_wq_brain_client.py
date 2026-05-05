"""WQSession HTTP client tests with mocked requests."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from agent_market.wq_brain.client import WQSession
from agent_market.wq_brain.dtypes import AlphaSettings


def _make_resp(*, status: int = 200, json_data=None, headers=None, text: str = "") -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.text = text
    r.json.return_value = json_data or {}
    if status >= 400:
        r.raise_for_status.side_effect = Exception(f"HTTP {status}")
    else:
        r.raise_for_status.return_value = None
    return r


def test_login_success():
    sess = WQSession("e@x.com", "pw", api_base="https://api.example.com")
    with patch("requests.Session.post", return_value=_make_resp(status=200)):
        sess.login()
        assert sess._logged_in is True


def test_login_invalid_credentials_raises_permission_error():
    sess = WQSession("e@x.com", "pw")
    with patch("requests.Session.post", return_value=_make_resp(status=401)):
        with pytest.raises(PermissionError):
            sess.login()


def test_ensure_login_only_logs_in_once_concurrently():
    sess = WQSession("e@x.com", "pw")
    call_count = [0]

    def fake_login(*a, **kw):
        call_count[0] += 1
        return _make_resp(status=200)

    with patch("requests.Session.post", side_effect=fake_login):
        threads = [threading.Thread(target=sess._ensure_login) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert call_count[0] == 1, f"login called {call_count[0]}x, expected 1"


def test_submit_simulation_returns_location_url():
    sess = WQSession("e@x.com", "pw", api_base="https://api.example.com")
    sess._logged_in = True
    settings = AlphaSettings()
    with patch.object(sess, "_request_with_retry") as mock_req:
        mock_req.return_value = _make_resp(
            status=201,
            headers={"Location": "/simulations/SIM123"},
        )
        url = sess.submit_simulation("rank(close)", settings)
        assert url == "https://api.example.com/simulations/SIM123"


def test_parse_sim_result_with_nested_alpha_dict():
    sess = WQSession("e@x.com", "pw")
    raw = {
        "status": "COMPLETE",
        "alpha": {"id": "A1", "is": {"sharpe": 1.5, "fitness": 1.2,
                                     "returns": 0.15, "turnover": 0.18,
                                     "drawdown": -0.05}},
    }
    r = sess.parse_sim_result(raw)
    assert r.alpha_id == "A1"
    assert r.sharpe == 1.5
    assert r.fitness == 1.2
    assert r.status == "COMPLETE"


def test_parse_sim_result_with_string_alpha_id():
    sess = WQSession("e@x.com", "pw")
    raw = {"status": "COMPLETE", "alpha": "A1", "is": {"sharpe": 1.0}}
    r = sess.parse_sim_result(raw)
    assert r.alpha_id == "A1"
    assert r.status == "COMPLETE"


def test_parse_sim_result_error_status():
    sess = WQSession("e@x.com", "pw")
    raw = {"status": "ERROR", "error": "compile failure"}
    r = sess.parse_sim_result(raw)
    assert r.status == "ERROR"
    assert r.error == "compile failure"


def test_get_alpha_correlations_handles_dict_response():
    sess = WQSession("e@x.com", "pw")
    sess._logged_in = True
    with patch.object(sess, "_request_with_retry") as mock_req:
        mock_req.return_value = _make_resp(
            status=200,
            json_data={"alphas": [{"id": "B1", "correlation": 0.4}]},
        )
        out = sess.get_alpha_correlations("A1")
        assert len(out) == 1
        assert out[0]["correlation"] == 0.4


def test_session_global_semaphore_limits_concurrency():
    sess = WQSession("e@x.com", "pw", max_concurrent=2)
    assert sess._global_sem._value == 2
    sess._global_sem.acquire()
    sess._global_sem.acquire()
    assert sess._global_sem._value == 0
    sess._global_sem.release()
    sess._global_sem.release()


def test_submit_alpha_uses_post_submit_endpoint():
    sess = WQSession("e@x.com", "pw", api_base="https://api.example.com")
    sess._logged_in = True
    captured = {}

    def fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        return _make_resp(status=201, json_data={})

    with patch("requests.Session.request", side_effect=fake_request):
        result = sess.submit_alpha("ABC123")

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/alphas/ABC123/submit")
    assert result.get("submitted") is True
    assert result.get("alpha_id") == "ABC123"


def test_submit_alpha_raises_on_4xx_with_body():
    sess = WQSession("e@x.com", "pw")
    sess._logged_in = True

    def fake_request(method, url, **kwargs):
        return _make_resp(status=400, json_data={
            "message": "Self-correlation 0.79 above cutoff 0.7"
        })

    with patch("requests.Session.request", side_effect=fake_request):
        with pytest.raises(RuntimeError, match="Self-correlation"):
            sess.submit_alpha("ABC123")


def test_get_alpha_status_returns_subset():
    sess = WQSession("e@x.com", "pw")
    sess._logged_in = True
    with patch.object(sess, "_request_with_retry") as mock_req:
        mock_req.return_value = _make_resp(status=200, json_data={
            "id": "X1",
            "status": "UNSUBMITTED",
            "stage": "IS",
            "dateSubmitted": None,
            "grade": "GOOD",
        })
        s = sess.get_alpha_status("X1")
    assert s["status"] == "UNSUBMITTED"
    assert s["stage"] == "IS"
    assert s["grade"] == "GOOD"


def test_request_with_retry_relogs_on_401():
    sess = WQSession("e@x.com", "pw")
    sess._logged_in = True

    call_log: list[str] = []

    def fake_request(method, url, **kwargs):
        call_log.append("req")
        if len(call_log) == 1:
            return _make_resp(status=401)
        return _make_resp(status=200, json_data={"ok": True})

    def fake_post(*a, **kw):
        call_log.append("login")
        return _make_resp(status=200)

    with patch("requests.Session.request", side_effect=fake_request), \
         patch("requests.Session.post", side_effect=fake_post):
        resp = sess._request_with_retry("GET", "https://api.example.com/foo")
        assert resp.status_code == 200
        assert "login" in call_log
        assert call_log.count("req") == 2
