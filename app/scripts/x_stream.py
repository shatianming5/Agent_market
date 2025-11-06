#!/usr/bin/env python
"""Consume X filtered stream with rule management.

Usage:
    python scripts/x_stream.py --rules conf/x_rules.yaml --max-messages 10

Requires X_STREAM_TOKEN or X_BEARER_TOKEN. Without credentials the script
exits successfully and prints a notice (no mock data generated).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import httpx
import yaml

RULES_ENDPOINT = "https://api.x.com/2/tweets/search/stream/rules"
STREAM_ENDPOINT = "https://api.x.com/2/tweets/search/stream"
DEFAULT_FIELDS = {
    "tweet.fields": "author_id,created_at,lang,public_metrics,referenced_tweets",
    "expansions": "author_id",
    "user.fields": "username,verified"
}


def load_rules(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("rules", [])


def auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def sync_rules(token: str, desired: List[Dict[str, str]]) -> None:
    if not desired:
        return
    headers = auth_headers(token)
    with httpx.Client(timeout=30) as client:
        current = client.get(RULES_ENDPOINT, headers=headers)
        current.raise_for_status()
        current_data = current.json().get("data", [])
        current_values = {rule.get("value"): rule.get("id") for rule in current_data if rule.get("value")}

        to_delete = [rid for value, rid in current_values.items() if value not in {rule["value"] for rule in desired}]
        if to_delete:
            client.post(RULES_ENDPOINT, headers=headers, json={"delete": {"ids": to_delete}})

        new_values = {rule["value"] for rule in desired}
        missing = [rule for rule in desired if rule["value"] not in current_values]
        if missing:
            payload = {"add": missing}
            resp = client.post(RULES_ENDPOINT, headers=headers, json=payload)
            resp.raise_for_status()


def run_stream(token: str, max_messages: int, rules: List[Dict[str, str]], output: Path) -> None:
    headers = auth_headers(token)
    params = DEFAULT_FIELDS.copy()
    output.mkdir(parents=True, exist_ok=True)
    out_path = output / "x_stream.ndjson"
    sync_rules(token, rules)

    received = 0
    with httpx.stream("GET", STREAM_ENDPOINT, headers=headers, params=params, timeout=None) as resp:
        resp.raise_for_status()
        with out_path.open("a", encoding="utf-8") as fh:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fh.write(json.dumps(data, ensure_ascii=False) + "\n")
                received += 1
                if received >= max_messages:
                    break
    print(f"Collected {received} messages to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consume X filtered stream")
    parser.add_argument("--rules", type=Path, default=Path("conf/x_rules.yaml"))
    parser.add_argument("--max-messages", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("data/raw/x"))
    args = parser.parse_args()

    token = os.getenv("X_STREAM_TOKEN") or os.getenv("X_BEARER_TOKEN")
    if not token:
        print("X_STREAM_TOKEN/X_BEARER_TOKEN not set; skipping filtered stream connection.")
        return

    rules = load_rules(args.rules)
    try:
        run_stream(token, args.max_messages, rules, args.output)
    except httpx.HTTPStatusError as exc:
        print(f"Stream request failed: {exc.response.status_code} {exc.response.text}")
    except KeyboardInterrupt:
        print("Stream interrupted by user")
    except Exception as exc:
        print(f"Unexpected error while running stream: {exc}")


if __name__ == "__main__":
    main()