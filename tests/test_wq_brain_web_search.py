"""web_search tests with mocked urllib responses."""
from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import patch

import pytest

from agent_market.wq_brain.web_search import (
    fetch_url,
    search_bing,
    search_brave,
    search_github,
    search_wikipedia,
    web_search,
)


def _fake_response(payload: dict | str, *, content_type: str = "application/json") -> object:
    class _R:
        def __init__(self, body: bytes, ctype: str) -> None:
            self._body = body
            self.headers = {"Content-Type": ctype}

        def read(self, *a, **kw):
            return self._body

        def geturl(self):
            return "https://example.com"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    if isinstance(payload, dict):
        body = json.dumps(payload).encode("utf-8")
    else:
        body = payload.encode("utf-8")
    return _R(body, content_type)


def test_search_wikipedia_parses_results():
    fake = _fake_response({
        "query": {
            "search": [
                {"title": "Momentum factor",
                 "snippet": 'a <span class="searchmatch">momentum</span> <span class="searchmatch">factor</span> bar'},
                {"title": "Carhart four-factor model", "snippet": "details"},
            ]
        }
    })
    with patch("agent_market.wq_brain.web_search.urllib.request.urlopen", return_value=fake):
        out = search_wikipedia("momentum", max_results=2)
    assert len(out) == 2
    assert out[0]["source"] == "wikipedia"
    assert out[0]["title"] == "Momentum factor"
    assert "Momentum_factor" in out[0]["url"]
    # searchmatch span tags stripped
    assert "<span" not in out[0]["snippet"]
    assert "</span>" not in out[0]["snippet"]
    assert "momentum" in out[0]["snippet"]


def test_search_github_parses_results():
    fake = _fake_response({
        "items": [
            {"full_name": "owner/repo1", "html_url": "https://github.com/owner/repo1",
             "description": "WorldQuant alpha library"},
        ]
    })
    with patch("agent_market.wq_brain.web_search.urllib.request.urlopen", return_value=fake):
        out = search_github("worldquant", max_results=5)
    assert len(out) == 1
    assert out[0]["source"] == "github"
    assert out[0]["title"] == "owner/repo1"
    assert out[0]["url"].startswith("https://github.com")


def test_search_brave_skipped_without_key(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    out = search_brave("anything")
    assert out == []


def test_search_brave_uses_key(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "test-key")
    fake = _fake_response({"web": {"results": [
        {"title": "T", "url": "https://t.com", "description": "snippet"},
    ]}})
    captured = {}

    def _capture_open(req, timeout):
        captured["headers"] = dict(req.header_items())
        return fake

    with patch("agent_market.wq_brain.web_search.urllib.request.urlopen", side_effect=_capture_open):
        out = search_brave("query", max_results=1)

    assert any("Subscription-Token" in h or "subscription-token" in h.lower() for h in captured["headers"])
    assert len(out) == 1
    assert out[0]["source"] == "brave"


def test_web_search_auto_fallback_uses_first_non_empty():
    # Brave returns empty (no key); Wikipedia returns hits → wikipedia is used
    with patch("agent_market.wq_brain.web_search.search_brave", return_value=[]), \
         patch("agent_market.wq_brain.web_search.search_wikipedia",
               return_value=[{"source": "wikipedia", "title": "X", "url": "u", "snippet": ""}]), \
         patch("agent_market.wq_brain.web_search.search_bing", return_value=[]), \
         patch("agent_market.wq_brain.web_search.search_github", return_value=[]):
        out = web_search("q", max_results=5)
    assert out["backends_used"] == ["wikipedia"]
    assert out["count"] == 1


def test_web_search_auto_falls_back_to_bing_when_others_empty():
    with patch("agent_market.wq_brain.web_search.search_brave", return_value=[]), \
         patch("agent_market.wq_brain.web_search.search_wikipedia", return_value=[]), \
         patch("agent_market.wq_brain.web_search.search_bing",
               return_value=[{"source": "bing", "title": "B", "url": "bu", "snippet": ""}]), \
         patch("agent_market.wq_brain.web_search.search_github", return_value=[]):
        out = web_search("q", max_results=5)
    assert out["backends_used"] == ["bing"]
    assert out["count"] == 1


def test_search_bing_parses_html_results():
    fake_html = """
    <li class="b_algo">
      <h2><a href="https://example.com/a">Result A title</a></h2>
      <p class="b_lineclamp">snippet text A</p>
    </li>
    <li class="b_algo">
      <h2><a href="https://example.com/b">Result B</a></h2>
      <p>snippet B</p>
    </li>
    """
    fake = _fake_response(fake_html, content_type="text/html; charset=utf-8")
    with patch("agent_market.wq_brain.web_search.urllib.request.urlopen", return_value=fake):
        out = search_bing("query", max_results=5)
    assert len(out) == 2
    assert out[0]["source"] == "bing"
    assert out[0]["title"] == "Result A title"
    assert out[0]["url"] == "https://example.com/a"
    assert "snippet text A" in out[0]["snippet"]
    assert out[1]["url"] == "https://example.com/b"


def test_web_search_explicit_sources_merges():
    with patch("agent_market.wq_brain.web_search.search_wikipedia",
               return_value=[{"source": "wikipedia", "title": "W", "url": "wu", "snippet": ""}]), \
         patch("agent_market.wq_brain.web_search.search_github",
               return_value=[{"source": "github", "title": "G", "url": "gu", "snippet": ""}]):
        out = web_search("q", max_results=5, sources=("wikipedia", "github"))
    assert "wikipedia" in out["backends_used"]
    assert "github" in out["backends_used"]
    assert out["count"] == 2


def test_fetch_url_strips_html_and_caps():
    html = "<html><head><script>x()</script></head><body><p>Hello</p><p>World</p></body></html>"
    fake = _fake_response(html, content_type="text/html; charset=utf-8")
    with patch("agent_market.wq_brain.web_search.urllib.request.urlopen", return_value=fake):
        out = fetch_url("https://example.com", max_chars=100)
    assert out["ok"] is True
    assert "Hello" in out["text"]
    assert "World" in out["text"]
    assert "<p>" not in out["text"]
    assert "x()" not in out["text"]  # script body skipped
    assert out["truncated"] is False


def test_fetch_url_handles_error():
    with patch("agent_market.wq_brain.web_search.urllib.request.urlopen", side_effect=Exception("boom")):
        out = fetch_url("https://nonexistent.invalid")
    assert out["ok"] is False
    assert "boom" in out["error"]
