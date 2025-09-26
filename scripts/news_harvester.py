#!/usr/bin/env python
"""Harvest news articles via RSS, News Sitemaps, and optional APIs.

Usage:
    python scripts/news_harvester.py --config conf/feeds.yaml

Outputs a parquet file under data/raw/news/ with extracted metadata and text.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse
from urllib import robotparser

import aiohttp
import feedparser
import pandas as pd
import tldextract
import yaml
from aiolimiter import AsyncLimiter
from langdetect import detect as detect_lang
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from xml.etree import ElementTree

import trafilatura

DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
USER_AGENT = os.getenv("NEWS_HARVESTER_USER_AGENT", DEFAULT_UA)
ROBOTS_CACHE: Dict[str, robotparser.RobotFileParser] = {}
TIMEOUT = aiohttp.ClientTimeout(total=20)
FETCH_LIMITER = AsyncLimiter(5, 1)  # 5 req / sec by default


class NewsRecord(BaseModel):
    source: str
    url: str
    title: Optional[str]
    published_at: Optional[dt.datetime]
    author: Optional[str]
    lang: Optional[str]
    summary: Optional[str]
    text: Optional[str]
    raw_json: Optional[str] = None
    collected_at: dt.datetime


@dataclass
class Config:
    rss: List[str]
    news_sitemaps: List[str]
    use_newsapi: bool = False
    newsapi_key: Optional[str] = None
    use_gdelt: bool = False
    gdelt_query: Optional[str] = None


def load_config(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(
        rss=data.get("rss", []),
        news_sitemaps=data.get("news_sitemaps", []),
        use_newsapi=bool(data.get("use_newsapi", False)),
        newsapi_key=data.get("newsapi_key"),
        use_gdelt=bool(data.get("use_gdelt", False)),
        gdelt_query=data.get("gdelt_query"),
    )


def get_output_path(base_dir: Path) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"news_{timestamp}.parquet"


def normalise_url(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    query = parsed.query
    scheme = parsed.scheme or "https"
    return f"{scheme}://{netloc}{path}" + (f"?{query}" if query else "")


def load_robots(domain: str) -> robotparser.RobotFileParser:
    if domain in ROBOTS_CACHE:
        return ROBOTS_CACHE[domain]
    robots_url = f"https://{domain}/robots.txt"
    parser = robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
    except Exception:
        # treat as allow if robots not accessible
        parser.parse("")
    ROBOTS_CACHE[domain] = parser
    return parser


def is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    domain = parsed.netloc
    if not domain:
        return False
    robots = load_robots(domain)
    try:
        return robots.can_fetch(USER_AGENT, url)
    except Exception:
        return False


async def fetch(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    async with FETCH_LIMITER:
        try:
            async with session.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as resp:
                if resp.status != 200:
                    return None
                return await resp.text()
        except Exception:
            return None


def parse_rss(xml: str, source_url: str) -> List[Dict[str, Any]]:
    feed = feedparser.parse(xml)
    entries = []
    for entry in feed.entries:
        link = entry.get("link")
        if not link:
            continue
        entries.append(
            {
                "url": link,
                "title": entry.get("title"),
                "summary": entry.get("summary"),
                "published": entry.get("published") or entry.get("updated"),
                "author": entry.get("author"),
                "raw": entry,
                "source_feed": source_url,
            }
        )
    return entries


def parse_news_sitemap(xml: str, source_url: str) -> List[Dict[str, Any]]:
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        return []
    ns = {
        "news": "http://www.google.com/schemas/sitemap-news/0.9"
    }
    entries: List[Dict[str, Any]] = []
    for url in root.findall("{*}url"):
        loc_el = url.find("{*}loc")
        if loc_el is None:
            continue
        link = loc_el.text
        if not link:
            continue
        publication_date = None
        title = None
        news_el = url.find("news:news", ns)
        if news_el is not None:
            pub_date_el = news_el.find("news:publication_date", ns)
            if pub_date_el is not None:
                publication_date = pub_date_el.text
            title_el = news_el.find("news:title", ns)
            if title_el is not None:
                title = title_el.text
        entries.append(
            {
                "url": link.strip(),
                "title": title,
                "summary": None,
                "published": publication_date,
                "author": None,
                "raw": {},
                "source_feed": source_url,
            }
        )
    return entries


async def gather_feed_entries(cfg: Config) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
        for rss_url in cfg.rss:
            xml = await fetch(session, rss_url)
            if xml:
                collected.extend(parse_rss(xml, rss_url))
        for sitemap_url in cfg.news_sitemaps:
            xml = await fetch(session, sitemap_url)
            if xml:
                collected.extend(parse_news_sitemap(xml, sitemap_url))
    return collected


async def fetch_article(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    if not is_allowed(url):
        return None
    html = await fetch(session, url)
    return html


def extract_text(url: str, html: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not html:
        downloaded = trafilatura.fetch_url(url)
    else:
        downloaded = html
    if not downloaded:
        return None, None
    result = trafilatura.extract(downloaded, url=url, include_comments=False, output_format="json")
    if not result:
        return None, None
    data = json.loads(result)
    return data.get("text"), data.get("summary")


def to_datetime(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        parsed = feedparser._parse_date(value)
        if parsed is None:
            return None
        return dt.datetime(*parsed[:6], tzinfo=dt.timezone.utc)
    except Exception:
        try:
            return dt.datetime.fromisoformat(value)
        except Exception:
            return None


async def process_entries(entries: List[Dict[str, Any]]) -> List[NewsRecord]:
    unique_urls: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        url = entry.get("url")
        if not url:
            continue
        norm = normalise_url(url)
        if norm not in unique_urls:
            unique_urls[norm] = entry
    records: List[NewsRecord] = []
    async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
        tasks = []
        for url, entry in unique_urls.items():
            tasks.append(fetch_article(session, url))
        html_results = await asyncio.gather(*tasks)
    for (url, entry), html in zip(unique_urls.items(), html_results):
        text, summary = extract_text(url, html)
        if not text:
            continue
        try:
            lang = detect_lang(text[:4000]) if text else None
        except Exception:
            lang = None
        domain = tldextract.extract(url)
        record = NewsRecord(
            source="{}.{}".format(domain.domain, domain.suffix) if domain.suffix else domain.domain,
            url=url,
            title=entry.get("title"),
            published_at=to_datetime(entry.get("published")),
            author=entry.get("author"),
            lang=lang,
            summary=summary or entry.get("summary"),
            text=text,
            raw_json=json.dumps(entry.get("raw", {})),
            collected_at=dt.datetime.now(dt.timezone.utc),
        )
        records.append(record)
    return records


async def run(config_path: Path, output_dir: Path) -> Path:
    cfg = load_config(config_path)
    entries = await gather_feed_entries(cfg)
    if not entries:
        raise SystemExit("No entries gathered from provided feeds/sitemaps")
    records = await process_entries(entries)
    if not records:
        raise SystemExit("No articles could be extracted from entries")
    df = pd.DataFrame([r.model_dump() for r in records])
    # ensure timezone aware to ns
    df["published_at"] = pd.to_datetime(df["published_at"])
    df["collected_at"] = pd.to_datetime(df["collected_at"])
    out_path = get_output_path(output_dir)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} articles to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest news via RSS and sitemaps")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw/news"))
    args = parser.parse_args()
    asyncio.run(run(args.config, args.output))


if __name__ == "__main__":
    main()