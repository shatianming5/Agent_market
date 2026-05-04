"""skill_search tests against the vendored worldquant-skill content."""
from __future__ import annotations

from agent_market.wq_brain.skill_search import (
    list_skill_files,
    search_skill,
    _split_markdown,
    _tokenize,
)


def test_tokenize_extracts_ascii_and_cjk():
    toks = _tokenize("rank ts_corr neutralization 中性化")
    assert "rank" in toks
    assert "ts_corr" in toks
    assert "neutralization" in toks
    assert "中性化" in toks


def test_split_markdown_groups_by_heading():
    text = """# Title
intro paragraph.

## Section A
Para 1 of A.

Para 2 of A.

## Section B
Para 1 of B.
"""
    chunks = _split_markdown(text)
    assert len(chunks) >= 3
    sections = {s for s, _ in chunks}
    assert "Section A" in sections
    assert "Section B" in sections


def test_list_skill_files_returns_vendored_content():
    out = list_skill_files()
    assert out["ok"] is True
    file_names = {f["name"] for f in out["files"]}
    assert "knowledge_base_search.md" in file_names
    assert "alpha_research_recorder.md" in file_names
    assert "factor_backtest.md" in file_names
    # All files non-empty
    assert all(f["size_bytes"] > 0 for f in out["files"])


def test_search_skill_finds_neutralization():
    out = search_skill("neutralization choices", top_k=3)
    assert out["ok"] is True
    assert out["total_chunks"] > 50  # plenty of chunks across 4 markdown files
    assert len(out["results"]) >= 1
    top = out["results"][0]
    assert top["file"] == "factor_backtest.md"
    assert "neutralization" in top["text"].lower() or "中性化" in top["text"]


def test_search_skill_finds_chinese_query():
    out = search_skill("降低 turnover", top_k=3)
    assert out["ok"] is True
    assert any("turnover" in r["text"].lower() for r in out["results"])


def test_search_skill_handles_empty_query():
    out = search_skill("", top_k=3)
    assert out["ok"] is False
    assert "empty" in out["error"].lower()


def test_search_skill_no_match_returns_empty_results():
    out = search_skill("xyzqwt99rare777", top_k=5)
    # Either ok=True with empty results, or ok=False — both acceptable
    if out["ok"]:
        assert out["results"] == []
