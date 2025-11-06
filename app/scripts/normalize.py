#!/usr/bin/env python
"""Normalize harvested news and X data into a unified schema."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from simhash import Simhash

NEWS_RAW_DIR = Path("data/raw/news")
X_RAW_DIR = Path("data/raw/x")
OUTPUT_DIR = Path("data/clean/news")

SCHEMA_COLUMNS = [
    "source",
    "url",
    "title",
    "published_at",
    "author",
    "lang",
    "text",
    "entities",
    "tags",
    "raw_json",
    "collected_at",
]


def load_news_records() -> List[Dict]:
    records: List[Dict] = []
    if not NEWS_RAW_DIR.exists():
        return records
    for path in sorted(NEWS_RAW_DIR.glob("news_*.parquet")):
        if path.stat().st_size == 0:
            continue
        df = pd.read_parquet(path)
        for row in df.to_dict("records"):
            records.append(
                {
                    "source": row.get("source"),
                    "url": row.get("url"),
                    "title": row.get("title"),
                    "published_at": row.get("published_at"),
                    "author": row.get("author"),
                    "lang": row.get("lang"),
                    "text": row.get("text"),
                    "entities": [],
                    "tags": ["news"],
                    "raw_json": row.get("raw_json"),
                    "collected_at": row.get("collected_at"),
                }
            )
    return records


def load_x_recent_records() -> List[Dict]:
    records: List[Dict] = []
    if not X_RAW_DIR.exists():
        return records
    for path in sorted(X_RAW_DIR.glob("x_recent_*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                records.append(
                    {
                        "source": "x",
                        "url": f"https://x.com/i/web/status/{data.get('tweet_id')}",
                        "title": None,
                        "published_at": data.get("created_at"),
                        "author": data.get("author_username"),
                        "lang": data.get("lang"),
                        "text": data.get("text"),
                        "entities": [],
                        "tags": ["x"],
                        "raw_json": json.dumps(data, ensure_ascii=False),
                        "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    }
                )
    for path in sorted(X_RAW_DIR.glob("x_stream.ndjson")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                data = payload.get("data", {})
                if not data:
                    continue
                includes = payload.get("includes", {})
                users = {u.get("id"): u for u in includes.get("users", [])}
                user = users.get(data.get("author_id"), {})
                records.append(
                    {
                        "source": "x",
                        "url": f"https://x.com/i/web/status/{data.get('id')}",
                        "title": None,
                        "published_at": data.get("created_at"),
                        "author": user.get("username"),
                        "lang": data.get("lang"),
                        "text": data.get("text"),
                        "entities": [],
                        "tags": ["x"],
                        "raw_json": json.dumps(payload, ensure_ascii=False),
                        "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    }
                )
    return records


def parse_datetime(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None or value == "" or pd.isna(value):
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


def is_duplicate(record: Dict, url_seen: Dict[str, Dict], simhashes: List[Tuple[Simhash, Dict]]) -> bool:
    url = record.get("url")
    if url and url in url_seen:
        return True
    text = record.get("text")
    if not text:
        return False
    sh = Simhash(text)
    for existing_hash, _ in simhashes:
        if sh.distance(existing_hash) <= 3:
            return True
    simhashes.append((sh, record))
    if url:
        url_seen[url] = record
    return False


def normalize(output_dir: Path) -> Path:
    records = load_news_records()
    records.extend(load_x_recent_records())
    if not records:
        raise SystemExit("No raw records found to normalize")

    url_seen: Dict[str, Dict] = {}
    simhashes: List[Tuple[Simhash, Dict]] = []
    normalized: List[Dict] = []
    for rec in records:
        if is_duplicate(rec, url_seen, simhashes):
            continue
        rec["published_at"] = parse_datetime(rec.get("published_at"))
        rec["collected_at"] = parse_datetime(rec.get("collected_at")) or pd.Timestamp.now(tz="UTC")
        normalized.append(rec)

    df = pd.DataFrame(normalized, columns=SCHEMA_COLUMNS)
    df["entities"] = df["entities"].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    df["tags"] = df["tags"].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "all.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} normalized records to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize news and X raw data")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    normalize(args.output)


if __name__ == "__main__":
    main()