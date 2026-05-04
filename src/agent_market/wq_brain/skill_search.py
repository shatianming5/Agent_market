"""Search the vendored worldquant-skill knowledge base.

The vendor_skill/ directory ships 4 SKILL markdown files (knowledge_base_search,
alpha_research_recorder, factor_backtest, _README) + 3 templates. We split
each into ~paragraph-sized chunks and rank them via TF-IDF + Jaccard hybrid
against the query.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]*")


def _tokenize(text: str) -> list[str]:
    # Lowercase ASCII; keep CJK; drop short tokens
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2]


@dataclass
class SkillChunk:
    file: str
    section: str
    text: str
    tokens: frozenset


def _split_markdown(text: str) -> list[tuple[str, str]]:
    """Split into (section_heading, chunk_text). Chunks are 2-6 line groups
    under the latest H2/H3 heading, trimmed."""
    chunks: list[tuple[str, str]] = []
    current_section = "(intro)"
    buffer: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith(("## ", "### ", "#### ")):
            if buffer:
                joined = "\n".join(buffer).strip()
                if joined:
                    chunks.append((current_section, joined))
                buffer = []
            current_section = line.lstrip("# ").strip()
            continue
        if line == "" and buffer:
            joined = "\n".join(buffer).strip()
            if joined:
                chunks.append((current_section, joined))
            buffer = []
            continue
        buffer.append(line)
    if buffer:
        joined = "\n".join(buffer).strip()
        if joined:
            chunks.append((current_section, joined))
    return chunks


def _vendor_dir() -> Path:
    return Path(__file__).resolve().parent / "vendor_skill"


def _load_chunks() -> list[SkillChunk]:
    out: list[SkillChunk] = []
    vendor = _vendor_dir()
    if not vendor.exists():
        return out
    for p in sorted(vendor.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        for section, chunk in _split_markdown(text):
            tokens = frozenset(_tokenize(chunk))
            if len(tokens) < 2:
                continue
            out.append(SkillChunk(file=p.name, section=section, text=chunk, tokens=tokens))
    return out


def _build_idf(chunks: list[SkillChunk]) -> dict[str, float]:
    n = max(len(chunks), 1)
    df: Counter = Counter()
    for c in chunks:
        for t in c.tokens:
            df[t] += 1
    return {t: math.log((n + 1) / (cnt + 1)) + 1.0 for t, cnt in df.items()}


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _tfidf_score(q_tokens: list[str], chunk: SkillChunk, idf: dict[str, float]) -> float:
    qt = Counter(q_tokens)
    et = Counter(_tokenize(chunk.text))
    return sum(
        math.sqrt(qt[t]) * math.sqrt(et[t]) * (idf.get(t, 1.0) ** 2)
        for t in qt if t in et
    )


_CHUNKS_CACHE: list[SkillChunk] | None = None
_IDF_CACHE: dict[str, float] | None = None


def _ensure_loaded() -> tuple[list[SkillChunk], dict[str, float]]:
    global _CHUNKS_CACHE, _IDF_CACHE
    if _CHUNKS_CACHE is None:
        _CHUNKS_CACHE = _load_chunks()
        _IDF_CACHE = _build_idf(_CHUNKS_CACHE)
    return _CHUNKS_CACHE, _IDF_CACHE or {}


def search_skill(query: str, *, top_k: int = 5) -> dict[str, Any]:
    chunks, idf = _ensure_loaded()
    if not chunks:
        return {
            "ok": False,
            "error": "vendor_skill/ is empty — no SKILL.md content available",
            "query": query,
            "results": [],
        }
    q_tokens = _tokenize(query)
    if not q_tokens:
        return {"ok": False, "error": "empty query after tokenization", "query": query, "results": []}
    q_set = frozenset(q_tokens)
    scored = [
        (c, 0.4 * _jaccard(q_set, c.tokens) + 0.6 * _tfidf_score(q_tokens, c, idf))
        for c in chunks
    ]
    scored.sort(key=lambda x: -x[1])
    top = [(c, s) for c, s in scored[:top_k] if s > 0]
    return {
        "ok": True,
        "query": query,
        "vendor_dir": str(_vendor_dir()),
        "total_chunks": len(chunks),
        "results": [
            {
                "score": round(s, 3),
                "file": c.file,
                "section": c.section,
                "text": c.text[:1500],
            }
            for c, s in top
        ],
    }


def list_skill_files() -> dict[str, Any]:
    vendor = _vendor_dir()
    if not vendor.exists():
        return {"ok": False, "vendor_dir": str(vendor), "files": []}
    files = []
    for p in sorted(vendor.iterdir()):
        if p.is_file():
            files.append({
                "name": p.name,
                "size_bytes": p.stat().st_size,
            })
    return {"ok": True, "vendor_dir": str(vendor), "files": files}
