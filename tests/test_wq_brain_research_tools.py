"""Tests for WQ BRAIN literature-search helpers."""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from agent_market.wq_brain.research_tools import (
    build_arxiv_search_query,
    search_arxiv,
    search_papers,
    search_ssrn_web,
)


class _FakeResponse:
    def __init__(self, payload: str) -> None:
        self.payload = payload.encode("utf-8")

    def read(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *args):  # noqa: ANN002
        return False


def test_arxiv_query_expands_quant_finance_aliases():
    query, categories = build_arxiv_search_query(
        "cross-sectional alpha factor 2024",
        categories="q-fin.*,stat.ML,cs.CE",
    )
    assert "cat:q-fin.TR" in query
    assert "cat:q-fin.ST" in query
    assert "cat:stat.ML" in query
    assert "cat:cs.CE" in query
    assert 'all:"cross-sectional"' in query
    assert "all:alpha" in query
    assert "all:factor" in query
    assert "all:2024" not in query
    assert "q-fin.TR" in categories


def test_arxiv_long_query_keeps_core_terms_but_allows_recall():
    query, _categories = build_arxiv_search_query(
        "order flow imbalance alpha market microstructure",
        categories="q-fin.TR",
    )
    assert "all:order AND all:flow AND all:imbalance" in query
    assert " OR all:alpha OR all:market OR all:microstructure" in query


def test_search_arxiv_uses_category_filter_and_parses_atom():
    atom = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2401.01234v1</id>
        <updated>2024-01-03T00:00:00Z</updated>
        <published>2024-01-02T00:00:00Z</published>
        <title>Order flow imbalance alpha</title>
        <summary>Market microstructure signal for short horizon returns.</summary>
        <author><name>Jane Quant</name></author>
        <arxiv:primary_category term="q-fin.TR" />
        <category term="q-fin.TR" />
        <category term="stat.ML" />
        <link href="http://arxiv.org/abs/2401.01234v1" rel="alternate" />
        <link title="pdf" href="http://arxiv.org/pdf/2401.01234v1" type="application/pdf" />
      </entry>
    </feed>
    """
    captured = {}

    def _fake_open(req, timeout):  # noqa: ANN001
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        return _FakeResponse(atom)

    with patch("agent_market.wq_brain.research_tools.urllib.request.urlopen", side_effect=_fake_open):
        out = search_arxiv("order flow imbalance", max_results=2, categories="q-fin.*,stat.ML")

    params = parse_qs(urlparse(captured["url"]).query)
    decoded_query = params["search_query"][0]
    assert "cat:q-fin.TR" in decoded_query
    assert "all:order" in decoded_query
    assert "all:flow" in decoded_query
    assert "all:imbalance" in decoded_query
    assert params["sortBy"] == ["relevance"]
    assert out["ok"] is True
    assert out["count"] == 1
    paper = out["papers"][0]
    assert paper["source"] == "arxiv"
    assert paper["arxiv_id"] == "2401.01234v1"
    assert paper["primary_category"] == "q-fin.TR"
    assert paper["pdf_url"].endswith("2401.01234v1")


def test_search_papers_merges_requested_sources():
    with patch(
        "agent_market.wq_brain.research_tools.search_arxiv",
        return_value={"ok": True, "source": "arxiv", "count": 1, "papers": [{"source": "arxiv", "title": "A"}]},
    ), patch(
        "agent_market.wq_brain.research_tools.search_ssrn_web",
        return_value={"ok": True, "source": "ssrn_web", "count": 1, "papers": [{"source": "ssrn_web", "title": "S"}]},
    ):
        out = search_papers("alpha", sources="arxiv,ssrn", max_results=1)

    assert out["ok"] is True
    assert out["count"] == 2
    assert {p["source"] for p in out["papers"]} == {"arxiv", "ssrn_web"}
    assert [s["source"] for s in out["sources"]] == ["arxiv", "ssrn_web"]


def test_search_ssrn_web_is_transparent_site_search():
    with patch(
        "agent_market.wq_brain.research_tools.search_bing",
        return_value=[{"title": "SSRN Alpha", "url": "https://papers.ssrn.com/x", "snippet": "working paper"}],
    ) as mock_search:
        out = search_ssrn_web("factor timing", max_results=1)

    query = mock_search.call_args[0][0]
    assert "site:ssrn.com" in query
    assert out["access_note"].startswith("SSRN is queried through site search")
    assert out["papers"][0]["source"] == "ssrn_web"
