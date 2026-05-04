"""General web search with graceful fallback chain.

Priority:
  1. Brave Search API (if BRAVE_API_KEY env set) — best quality
  2. Wikipedia REST API (always available, narrow but reliable)
  3. GitHub repository search (always available, code/repo only)

Each backend returns a list of {title, url, snippet, source} dicts.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any

USER_AGENT = "wq_brain-agent/1.0 (+https://github.com)"


def _http_get_json(url: str, *, headers: dict[str, str] | None = None, timeout: float = 15.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def search_brave(query: str, *, max_results: int = 5) -> list[dict[str, Any]]:
    key = os.environ.get("BRAVE_API_KEY", "")
    if not key:
        return []
    url = (
        "https://api.search.brave.com/res/v1/web/search?"
        f"q={urllib.parse.quote(query)}&count={max_results}"
    )
    data = _http_get_json(url, headers={"X-Subscription-Token": key, "Accept": "application/json"})
    out: list[dict[str, Any]] = []
    for item in (data.get("web") or {}).get("results", [])[:max_results]:
        out.append({
            "source": "brave",
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": (item.get("description") or "")[:600],
        })
    return out


def search_wikipedia(query: str, *, max_results: int = 5) -> list[dict[str, Any]]:
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&format=json"
        f"&list=search&srsearch={urllib.parse.quote(query)}&srlimit={max_results}"
    )
    try:
        data = _http_get_json(url)
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for item in (data.get("query") or {}).get("search", [])[:max_results]:
        title = item.get("title", "")
        page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        snippet = (item.get("snippet") or "").replace('<span class="searchmatch">', "").replace("</span>", "")
        out.append({
            "source": "wikipedia",
            "title": title,
            "url": page_url,
            "snippet": snippet[:600],
        })
    return out


def search_github(query: str, *, max_results: int = 5) -> list[dict[str, Any]]:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    url = (
        "https://api.github.com/search/repositories?"
        f"q={urllib.parse.quote(query)}&per_page={max_results}"
    )
    try:
        data = _http_get_json(url, headers=headers)
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for item in data.get("items", [])[:max_results]:
        out.append({
            "source": "github",
            "title": item.get("full_name", ""),
            "url": item.get("html_url", ""),
            "snippet": (item.get("description") or "")[:600],
        })
    return out


def web_search(
    query: str,
    *,
    max_results: int = 5,
    sources: tuple[str, ...] = ("auto",),
) -> dict[str, Any]:
    """Run a web search across enabled backends.

    sources=("auto",) tries brave→wikipedia→github in order, returning the
    first non-empty result set. Use ("wikipedia", "github") to query both
    and merge results.
    """
    chosen: list[str]
    if sources == ("auto",):
        chosen = ["brave", "wikipedia", "github"]
        merge = False
    else:
        chosen = list(sources)
        merge = True

    backends = {"brave": search_brave, "wikipedia": search_wikipedia, "github": search_github}
    used: list[str] = []
    results: list[dict[str, Any]] = []

    for name in chosen:
        fn = backends.get(name)
        if not fn:
            continue
        try:
            res = fn(query, max_results=max_results)
        except Exception as exc:
            res = []
            results.append({"source": name, "title": "", "url": "", "snippet": f"backend error: {exc}"})
        if res:
            used.append(name)
            results.extend(res)
            if not merge:
                break

    return {
        "query": query,
        "backends_used": used,
        "count": len(results),
        "results": results,
    }


def fetch_url(url: str, *, timeout: float = 20.0, max_chars: int = 6000) -> dict[str, Any]:
    """Fetch a URL, return text (best-effort plain-text extract)."""
    from html.parser import HTMLParser

    class _Stripper(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self._buf: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "noscript"):
                self._skip += 1

        def handle_endtag(self, tag):
            if tag in ("script", "style", "noscript") and self._skip > 0:
                self._skip -= 1

        def handle_data(self, data):
            if self._skip == 0:
                self._buf.append(data)

        def text(self) -> str:
            return " ".join(s.strip() for s in self._buf if s.strip())

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "").lower()
            raw = resp.read(2_000_000)  # cap raw bytes at 2MB
            charset = "utf-8"
            for part in ctype.split(";"):
                if "charset=" in part:
                    charset = part.split("charset=", 1)[1].strip()
            body = raw.decode(charset, errors="replace")
            final_url = resp.geturl()
    except Exception as exc:
        return {"ok": False, "url": url, "error": str(exc)}

    if "html" in ctype:
        parser = _Stripper()
        try:
            parser.feed(body)
        except Exception:
            pass
        text = parser.text()
    else:
        text = body

    # Collapse whitespace
    text = " ".join(text.split())
    return {
        "ok": True,
        "url": final_url,
        "content_type": ctype,
        "char_count": len(text),
        "text": text[:max_chars],
        "truncated": len(text) > max_chars,
    }
