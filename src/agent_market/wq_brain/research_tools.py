"""Literature-search helpers for WQ BRAIN agent research.

The agent needs paper search that is explicitly anchored to quantitative
finance.  A naked arXiv ``all:<query>`` search is too broad for alpha mining
and routinely returns unrelated physics / generic ML papers.  This module
keeps the search surface small and JSON-friendly while adding domain filters
and a multi-source fallback for sources where no stable official API exists.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html import unescape
from typing import Any, Iterable

from .web_search import USER_AGENT, search_bing


ARXIV_Q_FIN_CATEGORIES: tuple[str, ...] = (
    "q-fin.CP",  # Computational Finance
    "q-fin.EC",  # Economics
    "q-fin.GN",  # General Finance
    "q-fin.MF",  # Mathematical Finance
    "q-fin.PM",  # Portfolio Management
    "q-fin.PR",  # Pricing of Securities
    "q-fin.RM",  # Risk Management
    "q-fin.ST",  # Statistical Finance
    "q-fin.TR",  # Trading and Market Microstructure
)

DEFAULT_ARXIV_CATEGORIES: tuple[str, ...] = (
    *ARXIV_Q_FIN_CATEGORIES,
    "stat.ML",
    "cs.CE",
)

ARXIV_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "q-fin": ARXIV_Q_FIN_CATEGORIES,
    "q-fin.*": ARXIV_Q_FIN_CATEGORIES,
    "quant": ARXIV_Q_FIN_CATEGORIES,
    "finance": ARXIV_Q_FIN_CATEGORIES,
    "stat": ("stat.ML",),
    "stat.*": ("stat.ML",),
    "ml": ("stat.ML", "cs.LG"),
    "cs": ("cs.CE", "cs.LG"),
    "cs.*": ("cs.CE", "cs.LG"),
}

ARXIV_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "for", "from", "in", "of", "on", "or", "the", "to", "with",
})

SEMANTIC_SCHOLAR_FIELDS = (
    "title,abstract,year,publicationDate,url,venue,citationCount,"
    "externalIds,authors,tldr,fieldsOfStudy,publicationTypes,openAccessPdf"
)

OPENALEX_SELECT = (
    "id,doi,display_name,publication_year,publication_date,cited_by_count,"
    "primary_location,open_access,authorships,concepts,abstract_inverted_index"
)


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _split_csv(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_parts = value.split(",")
    else:
        raw_parts = []
        for item in value:
            raw_parts.extend(str(item).split(","))
    return [part.strip() for part in raw_parts if part.strip()]


def normalize_arxiv_categories(categories: str | Iterable[str] | None = None) -> list[str]:
    """Expand aliases like ``q-fin.*`` into concrete arXiv categories."""
    parts = _split_csv(categories) or list(DEFAULT_ARXIV_CATEGORIES)
    out: list[str] = []
    for part in parts:
        alias = part.lower()
        expanded = ARXIV_CATEGORY_ALIASES.get(alias)
        if expanded:
            out.extend(expanded)
            continue
        if not re.fullmatch(r"[a-z-]+(?:\.[A-Za-z0-9-]+)?", part):
            raise ValueError(f"invalid arXiv category: {part!r}")
        out.append(part)
    return _dedupe(out)


def build_arxiv_search_query(
    query: str,
    *,
    categories: str | Iterable[str] | None = None,
    raw_query: bool = False,
) -> tuple[str, list[str]]:
    """Return the arXiv ``search_query`` string plus expanded categories."""
    clean = " ".join((query or "").split())
    if not clean:
        raise ValueError("empty arXiv query")

    cats = normalize_arxiv_categories(categories)
    if raw_query:
        keyword_clause = clean
    else:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]*|\d{2,}", clean)
        terms = [
            token
            for token in tokens
            if token.lower() not in ARXIV_STOPWORDS
            and not re.fullmatch(r"(?:19|20)\d{2}", token)
        ]
        if not terms:
            terms = [token for token in tokens if token.lower() not in ARXIV_STOPWORDS] or [clean]

        clauses = []
        for term in terms:
            safe = term.replace('"', " ")
            if "-" in safe:
                clauses.append(f'all:"{safe}"')
            else:
                clauses.append(f"all:{safe}")
        if len(clauses) <= 4:
            keyword_clause = "(" + " AND ".join(clauses) + ")"
        else:
            required = " AND ".join(clauses[:3])
            optional = " OR ".join(clauses[3:])
            keyword_clause = f"(({required}) OR {optional})"

    if not cats:
        return keyword_clause, []
    category_clause = "(" + " OR ".join(f"cat:{cat}" for cat in cats) + ")"
    return f"{category_clause} AND {keyword_clause}", cats


def _http_get_json(url: str, *, headers: dict[str, str] | None = None, timeout: float = 20.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _clean_text(text: str | None, *, max_chars: int = 1600) -> str:
    return " ".join((text or "").split())[:max_chars]


def _arxiv_id(raw_id: str) -> str:
    if "/abs/" in raw_id:
        return raw_id.rsplit("/abs/", 1)[1]
    return raw_id.rstrip("/").rsplit("/", 1)[-1]


def _tag_text(block: str, tag: str) -> str:
    match = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", block, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    text = re.sub(r"<[^>]+>", " ", match.group(1))
    return unescape(" ".join(text.split()))


def _attrs(raw: str) -> dict[str, str]:
    return {
        key.lower(): unescape(value)
        for key, value in re.findall(r"([A-Za-z_:][-A-Za-z0-9_:]*)=[\"']([^\"']*)[\"']", raw)
    }


def _parse_arxiv_atom_fallback(xml_data: bytes) -> list[dict[str, Any]]:
    """Parse arXiv Atom with regex when Python's expat module is unavailable."""
    text = xml_data.decode("utf-8", errors="replace")
    entries: list[dict[str, Any]] = []
    for block in re.findall(r"<entry\b[^>]*>(.*?)</entry>", text, flags=re.IGNORECASE | re.DOTALL):
        raw_id = _tag_text(block, "id")
        categories_found = [
            attrs.get("term", "")
            for raw_attrs in re.findall(r"<category\b([^>]*)/?>", block, flags=re.IGNORECASE)
            for attrs in [_attrs(raw_attrs)]
            if attrs.get("term")
        ]
        primary_match = re.search(
            r"<arxiv:primary_category\b([^>]*)/?>",
            block,
            flags=re.IGNORECASE,
        )
        primary = _attrs(primary_match.group(1)).get("term", "") if primary_match else ""
        authors = [
            _clean_text(_tag_text(author_block, "name"), max_chars=120)
            for author_block in re.findall(r"<author\b[^>]*>(.*?)</author>", block, flags=re.IGNORECASE | re.DOTALL)
        ]
        pdf_url = ""
        for raw_attrs in re.findall(r"<link\b([^>]*)/?>", block, flags=re.IGNORECASE):
            attrs = _attrs(raw_attrs)
            if attrs.get("title") == "pdf" or attrs.get("type") == "application/pdf":
                pdf_url = attrs.get("href", "")
                break
        entries.append({
            "source": "arxiv",
            "title": _clean_text(_tag_text(block, "title"), max_chars=400),
            "id": raw_id,
            "arxiv_id": _arxiv_id(raw_id),
            "url": raw_id,
            "pdf_url": pdf_url,
            "abstract": _clean_text(_tag_text(block, "summary")),
            "published": _tag_text(block, "published"),
            "updated": _tag_text(block, "updated"),
            "authors": [a for a in authors if a][:8],
            "primary_category": primary,
            "categories": categories_found,
        })
    return entries


def _parse_arxiv_atom(xml_data: bytes) -> list[dict[str, Any]]:
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    try:
        root = ET.fromstring(xml_data)
    except Exception:
        return _parse_arxiv_atom_fallback(xml_data)

    papers: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        raw_id = entry.findtext("atom:id", default="", namespaces=ns) or ""
        categories_found = [
            c.attrib.get("term", "")
            for c in entry.findall("atom:category", ns)
            if c.attrib.get("term")
        ]
        primary = entry.find("arxiv:primary_category", ns)
        authors = [
            _clean_text(author.findtext("atom:name", default="", namespaces=ns), max_chars=120)
            for author in entry.findall("atom:author", ns)
        ]
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break
        papers.append({
            "source": "arxiv",
            "title": _clean_text(entry.findtext("atom:title", default="", namespaces=ns), max_chars=400),
            "id": raw_id.strip(),
            "arxiv_id": _arxiv_id(raw_id),
            "url": raw_id.strip(),
            "pdf_url": pdf_url,
            "abstract": _clean_text(entry.findtext("atom:summary", default="", namespaces=ns)),
            "published": (entry.findtext("atom:published", default="", namespaces=ns) or "").strip(),
            "updated": (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip(),
            "authors": [a for a in authors if a][:8],
            "primary_category": primary.attrib.get("term", "") if primary is not None else "",
            "categories": categories_found,
        })
    return papers


def search_arxiv(
    query: str,
    *,
    max_results: int = 5,
    categories: str | Iterable[str] | None = None,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    raw_query: bool = False,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Search arXiv with quant-finance category anchors."""
    max_results = max(1, min(int(max_results), 50))
    if sort_by not in {"relevance", "submittedDate", "lastUpdatedDate"}:
        raise ValueError("sort_by must be relevance, submittedDate, or lastUpdatedDate")
    if sort_order not in {"ascending", "descending"}:
        raise ValueError("sort_order must be ascending or descending")

    search_query, expanded_categories = build_arxiv_search_query(
        query, categories=categories, raw_query=raw_query
    )
    params = {
        "search_query": search_query,
        "start": "0",
        "max_results": str(max_results),
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        xml_data = resp.read()

    papers = _parse_arxiv_atom(xml_data)

    return {
        "ok": True,
        "source": "arxiv",
        "query": query,
        "search_query": search_query,
        "category_filter": expanded_categories,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "url": url,
        "count": len(papers),
        "papers": papers,
    }


def search_semantic_scholar(
    query: str,
    *,
    max_results: int = 5,
    year: str | None = None,
    fields_of_study: str | Iterable[str] | None = None,
    min_citation_count: int | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Search Semantic Scholar Academic Graph paper relevance endpoint."""
    max_results = max(1, min(int(max_results), 20))
    params: dict[str, str] = {
        "query": query,
        "limit": str(max_results),
        "fields": SEMANTIC_SCHOLAR_FIELDS,
    }
    if year:
        params["year"] = year
    fos = _split_csv(fields_of_study)
    if fos:
        params["fieldsOfStudy"] = ",".join(fos)
    if min_citation_count is not None:
        params["minCitationCount"] = str(max(0, int(min_citation_count)))

    headers: dict[str, str] = {}
    if key := os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""):
        headers["x-api-key"] = key
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params)
    data = _http_get_json(url, headers=headers, timeout=timeout)
    papers: list[dict[str, Any]] = []
    for item in data.get("data", [])[:max_results]:
        tldr = item.get("tldr") or {}
        external_ids = item.get("externalIds") or {}
        papers.append({
            "source": "semantic_scholar",
            "title": _clean_text(item.get("title"), max_chars=400),
            "id": item.get("paperId", ""),
            "url": item.get("url", ""),
            "abstract": _clean_text(item.get("abstract")),
            "year": item.get("year"),
            "published": item.get("publicationDate") or "",
            "venue": item.get("venue") or "",
            "citation_count": item.get("citationCount"),
            "fields_of_study": item.get("fieldsOfStudy") or [],
            "publication_types": item.get("publicationTypes") or [],
            "external_ids": external_ids,
            "doi": external_ids.get("DOI", ""),
            "arxiv_id": external_ids.get("ArXiv", ""),
            "tldr": _clean_text(tldr.get("text"), max_chars=600),
            "open_access_pdf": (item.get("openAccessPdf") or {}).get("url", ""),
            "authors": [
                _clean_text(author.get("name"), max_chars=120)
                for author in (item.get("authors") or [])[:8]
            ],
        })
    return {
        "ok": True,
        "source": "semantic_scholar",
        "query": query,
        "url": url,
        "count": len(papers),
        "total": data.get("total"),
        "papers": papers,
    }


def _openalex_abstract(inv: dict[str, list[int]] | None) -> str:
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for idx in idxs:
            positions.append((idx, word))
    return _clean_text(" ".join(word for _, word in sorted(positions)), max_chars=1600)


def search_openalex(
    query: str,
    *,
    max_results: int = 5,
    from_year: int | None = None,
    to_year: int | None = None,
    sort_by: str = "relevance",
    domain_filter: bool = True,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Search OpenAlex works and normalize the metadata shape."""
    max_results = max(1, min(int(max_results), 25))
    params: dict[str, str] = {
        "search": query,
        "per-page": str(max_results),
        "select": OPENALEX_SELECT,
    }
    filters: list[str] = []
    if domain_filter:
        filters.append("concepts.id:C162324750|C144133560|C41008148")
    if from_year is not None:
        filters.append(f"from_publication_date:{int(from_year):04d}-01-01")
    if to_year is not None:
        filters.append(f"to_publication_date:{int(to_year):04d}-12-31")
    if filters:
        params["filter"] = ",".join(filters)
    if sort_by == "citations":
        params["sort"] = "cited_by_count:desc"
    if mailto := os.environ.get("OPENALEX_MAILTO", ""):
        params["mailto"] = mailto

    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    data = _http_get_json(url, timeout=timeout)
    papers: list[dict[str, Any]] = []
    for item in data.get("results", [])[:max_results]:
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {}
        authors = []
        for authorship in (item.get("authorships") or [])[:8]:
            author = authorship.get("author") or {}
            name = author.get("display_name")
            if name:
                authors.append(_clean_text(name, max_chars=120))
        papers.append({
            "source": "openalex",
            "title": _clean_text(item.get("display_name"), max_chars=400),
            "id": item.get("id", ""),
            "url": primary_location.get("landing_page_url") or item.get("id", ""),
            "doi": item.get("doi", ""),
            "abstract": _openalex_abstract(item.get("abstract_inverted_index")),
            "year": item.get("publication_year"),
            "published": item.get("publication_date") or "",
            "citation_count": item.get("cited_by_count"),
            "venue": source.get("display_name", ""),
            "open_access": item.get("open_access") or {},
            "authors": authors,
            "concepts": [
                {
                    "display_name": c.get("display_name", ""),
                    "score": c.get("score"),
                }
                for c in (item.get("concepts") or [])[:6]
            ],
        })
    meta = data.get("meta") or {}
    return {
        "ok": True,
        "source": "openalex",
        "query": query,
        "url": url,
        "count": len(papers),
        "total": meta.get("count"),
        "papers": papers,
    }


def search_ssrn_web(query: str, *, max_results: int = 5) -> dict[str, Any]:
    """Search SSRN through a general search engine.

    SSRN does not provide a stable unauthenticated public paper-search API for
    this workflow.  We therefore keep this as a transparent web-search fallback
    and mark it as such in the JSON response.
    """
    max_results = max(1, min(int(max_results), 10))
    ssrn_query = f"site:ssrn.com {query} quantitative finance alpha factor"
    hits = search_bing(ssrn_query, max_results=max_results)
    papers = [
        {
            "source": "ssrn_web",
            "title": hit.get("title", ""),
            "url": hit.get("url", ""),
            "abstract": hit.get("snippet", ""),
            "id": "",
            "published": "",
        }
        for hit in hits[:max_results]
    ]
    return {
        "ok": True,
        "source": "ssrn_web",
        "query": query,
        "search_query": ssrn_query,
        "access_note": "SSRN is queried through site search; no stable public SSRN API is assumed.",
        "count": len(papers),
        "papers": papers,
    }


def search_papers(
    query: str,
    *,
    max_results: int = 5,
    sources: str | Iterable[str] | None = None,
    arxiv_categories: str | Iterable[str] | None = None,
    arxiv_sort_by: str = "relevance",
    year: str | None = None,
    fields_of_study: str | Iterable[str] | None = None,
    min_citation_count: int | None = None,
) -> dict[str, Any]:
    """Run a small multi-source literature search and return normalized papers."""
    requested = _split_csv(sources) or ["arxiv", "semantic_scholar", "openalex", "ssrn"]
    papers: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for source in requested:
        key = source.lower().replace("-", "_")
        try:
            if key == "arxiv":
                report = search_arxiv(
                    query,
                    max_results=max_results,
                    categories=arxiv_categories,
                    sort_by=arxiv_sort_by,
                )
            elif key in {"semantic", "semantic_scholar", "s2"}:
                report = search_semantic_scholar(
                    query,
                    max_results=max_results,
                    year=year,
                    fields_of_study=fields_of_study,
                    min_citation_count=min_citation_count,
                )
            elif key == "openalex":
                from_year = None
                to_year = None
                if year and re.fullmatch(r"\d{4}", year):
                    from_year = int(year)
                    to_year = int(year)
                elif year and re.fullmatch(r"\d{4}-", year):
                    from_year = int(year[:4])
                report = search_openalex(
                    query,
                    max_results=max_results,
                    from_year=from_year,
                    to_year=to_year,
                )
            elif key in {"ssrn", "ssrn_web"}:
                report = search_ssrn_web(query, max_results=max_results)
            else:
                errors.append({"source": source, "error": "unknown source"})
                continue
        except Exception as exc:
            errors.append({"source": source, "error": str(exc)})
            continue

        source_reports.append({
            "source": report.get("source", source),
            "count": report.get("count", 0),
            "total": report.get("total"),
            "url": report.get("url", ""),
            "search_query": report.get("search_query", ""),
            "access_note": report.get("access_note", ""),
        })
        papers.extend(report.get("papers", []))

    return {
        "ok": bool(papers) or not errors,
        "query": query,
        "sources_requested": requested,
        "sources": source_reports,
        "errors": errors,
        "count": len(papers),
        "papers": papers,
    }
