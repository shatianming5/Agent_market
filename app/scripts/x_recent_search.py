#!/usr/bin/env python
"""Fetch tweets via X API recent search endpoint.

Usage:
    python scripts/x_recent_search.py --query "keyword" --max-results 50

Requires environment variable X_BEARER_TOKEN. If not provided, the script
exits gracefully without fetching.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import yaml

RECENT_SEARCH_URL = "https://api.x.com/2/tweets/search/recent"
DEFAULT_FIELDS = {
    "tweet.fields": "author_id,created_at,lang,public_metrics,referenced_tweets" ,
    "expansions": "author_id", 
    "user.fields": "username,verified,created_at"
}


def load_keywords(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    topics = data.get("topics", {})
    keywords: List[str] = []
    for topic in topics.values():
        keywords.extend(topic.get("include", []))
    return keywords


def build_query(cli_query: Optional[str], keywords: List[str]) -> str:
    if cli_query:
        return cli_query
    if keywords:
        parts = [f"\"{kw}\"" if " " in kw else kw for kw in keywords[:10]]
        return " OR ".join(parts)
    return "news lang:en -is:retweet"


def fetch_recent(query: str, max_results: int, bearer: str, next_token: Optional[str] = None) -> Dict[str, Any]:
    params = {
        "query": query,
        "max_results": max_results,
    }
    params.update(DEFAULT_FIELDS)
    if next_token:
        params["next_token"] = next_token
    headers = {"Authorization": f"Bearer {bearer}"}
    with httpx.Client(timeout=30) as client:
        resp = client.get(RECENT_SEARCH_URL, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


def save_results(data: Dict[str, Any], output_dir: Path) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"x_recent_{timestamp}.jsonl"
    includes = data.get("includes", {})
    users = {user["id"]: user for user in includes.get("users", [])}
    with out_path.open("w", encoding="utf-8") as fh:
        for tweet in data.get("data", []):
            author = users.get(tweet.get("author_id"), {})
            record = {
                "tweet_id": tweet.get("id"),
                "text": tweet.get("text"),
                "created_at": tweet.get("created_at"),
                "lang": tweet.get("lang"),
                "author_id": tweet.get("author_id"),
                "author_username": author.get("username"),
                "author_verified": author.get("verified"),
                "public_metrics": tweet.get("public_metrics"),
                "raw": tweet,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(data.get('data', []))} tweets to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch tweets via recent search")
    parser.add_argument("--query", type=str, default=None, help="X API search query")
    parser.add_argument("--max-results", type=int, default=50, choices=range(10, 101), metavar="[10-100]")
    parser.add_argument("--keywords", type=Path, default=Path("conf/keywords.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/x"))
    args = parser.parse_args()

    bearer = os.getenv("X_BEARER_TOKEN")
    if not bearer:
        print("X_BEARER_TOKEN not set; skipping recent search fetch.")
        return

    keywords = load_keywords(args.keywords)
    query = build_query(args.query, keywords)

    try:
        data = fetch_recent(query, args.max_results, bearer)
    except httpx.HTTPStatusError as exc:
        print(f"X API request failed: {exc.response.status_code} {exc.response.text}")
        return
    except Exception as exc:
        print(f"Unexpected error while calling X API: {exc}")
        return

    save_results(data, args.output)


if __name__ == "__main__":
    main()