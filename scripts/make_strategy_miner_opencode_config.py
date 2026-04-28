#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a strategy-miner runtime config that forces OpenCode.")
    parser.add_argument("--base", required=True, help="Base miner config JSON.")
    parser.add_argument("--output", required=True, help="Output runtime config JSON.")
    parser.add_argument("--model", default="gpt-5.2", help="OpenCode model name.")
    parser.add_argument("--agent-url", default=None, help="Optional OpenCode server URL.")
    parser.add_argument("--max-parallel-candidates", type=int, default=None)
    args = parser.parse_args()

    base_path = _resolve(args.base)
    output_path = _resolve(args.output)
    payload = json.loads(base_path.read_text(encoding="utf-8"))

    budget = payload.setdefault("budget", {})
    budget["provider"] = "opencode"
    budget["model"] = str(args.model)
    if args.agent_url is None:
        budget.pop("base_url", None)
    else:
        budget["base_url"] = str(args.agent_url)
    if args.max_parallel_candidates is not None:
        budget["max_parallel_candidates"] = int(args.max_parallel_candidates)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
