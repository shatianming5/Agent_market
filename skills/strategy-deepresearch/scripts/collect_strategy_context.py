#!/usr/bin/env python3
"""Collect a compact local evidence bundle for Agent_market strategy-loop audits."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as exc:
        return {"_error": f"invalid json: {exc}", "_path": str(path)}


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def latest_run(runs_root: Path) -> Path:
    candidates = [p for p in runs_root.iterdir() if p.is_dir()] if runs_root.exists() else []
    if not candidates:
        raise SystemExit(f"no strategy-loop runs found under {runs_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def candidate_name(row: Mapping[str, Any]) -> str | None:
    candidate = row.get("candidate")
    if isinstance(candidate, Mapping):
        return str(candidate.get("name") or "") or None
    return None


def best_row(rows: list[Any]) -> Mapping[str, Any]:
    valid = [r for r in rows if isinstance(r, Mapping)]
    if not valid:
        return {}
    return max(valid, key=lambda r: as_float(r.get("score")) if as_float(r.get("score")) is not None else float("-inf"))


def compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "iteration": row.get("iteration"),
        "name": candidate_name(row),
        "score": row.get("score"),
        "constraints_ok": row.get("constraints_ok"),
        "score_components": row.get("score_components") if isinstance(row.get("score_components"), Mapping) else {},
        "research_metrics": row.get("research_metrics") if isinstance(row.get("research_metrics"), Mapping) else {},
        "freqtrade_metrics": row.get("freqtrade_metrics") if isinstance(row.get("freqtrade_metrics"), Mapping) else {},
        "parameter_signature": row.get("parameter_signature"),
        "violations": row.get("violations") or [],
        "diagnostics": row.get("diagnostics"),
        "promotion": row.get("promotion") if isinstance(row.get("promotion"), Mapping) else {},
        "candidate_path": row.get("candidate_path"),
    }


def iteration_files(run_root: Path, iteration: int, repo: Path) -> dict[str, Any]:
    idir = run_root / f"iter_{iteration:02d}"
    names = [
        "context/prepare.json",
        "candidate.json",
        "signal_export.json",
        "backtest.json",
        "evaluation.json",
        "analysis.md",
        "error.json",
        "freqtrade_backtest.log",
    ]
    files: dict[str, Any] = {}
    for name in names:
        path = idir / name
        if path.exists():
            files[name] = {
                "path": rel(path, repo),
                "bytes": path.stat().st_size,
                "mtime": path.stat().st_mtime,
            }
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default="")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--recent", type=int, default=8)
    args = parser.parse_args()

    repo = args.repo.resolve()
    runs_root = repo / "artifacts" / "factor_strategy_loop"
    run_root = runs_root / args.run_id if args.run_id else latest_run(runs_root)
    run_id = run_root.name

    checkpoint = load_json(run_root / "checkpoint.json", {})
    leaderboard = load_json(run_root / "leaderboard.json", {"rows": []})
    rows = leaderboard.get("rows") if isinstance(leaderboard, Mapping) else []
    rows = rows if isinstance(rows, list) else []
    best = best_row(rows)
    latest = rows[-1] if rows and isinstance(rows[-1], Mapping) else {}

    state = checkpoint.get("state") if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("state"), Mapping) else {}
    config = checkpoint.get("config") if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("config"), Mapping) else {}
    recent_rows = [compact_row(r) for r in rows[-max(1, int(args.recent)) :] if isinstance(r, Mapping)]
    best_iter = int(best.get("iteration") or 0) if best else 0
    latest_iter = int(latest.get("iteration") or 0) if latest else 0

    payload = {
        "version": "strategy-deepresearch-context-v1",
        "created_at": time.time(),
        "repo": str(repo),
        "run_id": run_id,
        "run_root": rel(run_root, repo),
        "checkpoint": {
            "path": rel(run_root / "checkpoint.json", repo),
            "config": config,
            "state": state,
        },
        "leaderboard": {
            "path": rel(run_root / "leaderboard.json", repo),
            "rows": len(rows),
            "best": compact_row(best) if best else {},
            "latest": compact_row(latest) if latest else {},
            "recent": recent_rows,
        },
        "artifacts": {
            "best_iteration": iteration_files(run_root, best_iter, repo) if best_iter else {},
            "latest_iteration": iteration_files(run_root, latest_iter, repo) if latest_iter else {},
            "run_best_dir": rel(run_root / "best", repo) if (run_root / "best").exists() else None,
            "final_promotion": rel(run_root / "final_promotion.json", repo) if (run_root / "final_promotion.json").exists() else None,
        },
        "repo_files": {
            "strategy_loop": "src/agent_market/factor_lab/strategy_loop.py",
            "rank_portfolio": "src/agent_market/factor_lab/rank_portfolio.py",
            "freqtrade_loader": "user_data/strategies/ELRankPortfolioLeverageStrategy.py",
            "freqtrade_cli": "scripts/freqtrade_cli.py",
            "monitor": "scripts/strategy_loop_monitor.py",
        },
    }

    out = args.out
    if out is None:
        out = repo / "artifacts" / "strategy_deepresearch" / run_id / "context.json"
    out = out if out.is_absolute() else repo / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
