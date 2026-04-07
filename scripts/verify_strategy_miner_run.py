#!/usr/bin/env python3
"""Verify a Strategy Miner run (no-template + static validation + leaderboard).

This is a lightweight gate intended for CI/local automation.

Checks:
- checkpoint best_score is finite and not -inf (backward compatible with legacy best_reward)
- leaderboard has eligible items (constraints_ok)
- top-N entries are not template-sourced and pass sandbox static validation

Also prints top-N key metrics + risk gate results.
"""

from __future__ import annotations

import ast
import argparse
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_num(v) -> str:  # noqa: ANN001
    if v is None:
        return "?"
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return str(v)
        if abs(x) >= 1000:
            return f"{x:.0f}"
        if abs(x) >= 100:
            return f"{x:.2f}"
        return f"{x:.4f}" if abs(x) < 10 else f"{x:.3f}"
    except Exception:
        return str(v)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a strategy miner run")
    parser.add_argument("--run-id", required=True, help="Run id (hex)")
    parser.add_argument("--top", type=int, default=3, help="Validate top-N leaderboard entries")
    args = parser.parse_args()

    from agent_market.strategy_miner.artifacts import leaderboard_path
    from agent_market.strategy_miner.dtypes import MinerState
    from agent_market.strategy_miner.runner import miner_run_dir
    from agent_market.strategy_miner.sandbox import validate_strategy_code

    forbidden_model_imports = {
        # Keep the model wrapper away from obvious dangerous surfaces.
        "os",
        "subprocess",
        "socket",
        "requests",
        "importlib",
        "ctypes",
        "multiprocessing",
        "signal",
        "pty",
        "webbrowser",
        "http",
        "urllib",
        "ftplib",
        "smtplib",
        "telnetlib",
        "xmlrpc",
        "shutil",
        "talib",
    }

    run_id = str(args.run_id).strip().lower()
    miner_dir = miner_run_dir(run_id)
    cp = miner_dir / "checkpoint.json"
    if not cp.exists():
        print(f"ERROR: checkpoint not found: {cp}")
        return 2

    state = MinerState.from_dict(_load_json(cp))
    if not math.isfinite(float(state.best_score)) or float(state.best_score) == float("-inf"):
        print(f"ERROR: best_score invalid: {state.best_score}")
        return 3

    lb_path = leaderboard_path(miner_dir)
    if not lb_path.exists():
        print(f"ERROR: leaderboard not found: {lb_path}")
        return 4

    lb = _load_json(lb_path)
    items = lb.get("items") or []
    if not isinstance(items, list) or not items:
        print("ERROR: leaderboard has no eligible items")
        return 5

    top_n = max(1, int(args.top))
    failures: list[str] = []

    for row in items[:top_n]:
        name = str(row.get("name") or "").strip()
        it = int(row.get("iteration") or 0)
        candidate_type = str(row.get("candidate_type") or "").strip().lower()
        if not name:
            failures.append(f"leaderboard row missing name: {row}")
            continue

        cand_dir = miner_dir / "candidates" / f"iter_{it:04d}"
        code_path = cand_dir / f"{name}.py"
        if not code_path.exists():
            failures.append(f"missing candidate code: {code_path}")
            continue

        code = code_path.read_text(encoding="utf-8", errors="replace")
        if name == "TemplateRsiStrategy" or "TemplateRsiStrategy" in code:
            failures.append(f"template strategy detected in top{top_n}: {name}")

        if candidate_type in {"ml", "dl", "rl"}:
            # Model candidates are deterministic wrappers that legitimately need
            # file IO and ML deps; the strict sandbox validator is intended for
            # LLM-authored rule strategies.
            try:
                tree = ast.parse(code)
            except SyntaxError as exc:
                failures.append(f"syntax error for {name}: {exc}")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_mod = str(alias.name).split(".", 1)[0]
                        if root_mod in forbidden_model_imports:
                            failures.append(f"forbidden import for {name}: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        root_mod = str(node.module).split(".", 1)[0]
                        if root_mod in forbidden_model_imports:
                            failures.append(f"forbidden import for {name}: {node.module}")
        else:
            ok, msg = validate_strategy_code(code)
            if not ok:
                failures.append(f"static validation failed for {name}: {msg}")

    if failures:
        print("VERIFY FAIL")
        for f in failures:
            print("-", f)
        return 6

    best = (lb.get("best") or {}).get("name")
    cfg = lb.get("config") or {}
    print("VERIFY PASS")
    print(f"run_id: {run_id}")
    print(f"best_score: {state.best_score}")
    print(f"best: {best}")
    print(f"leaderboard: {lb_path}")
    print(
        "risk_gates:",
        f"min_trades={cfg.get('min_trades')} max_abs_drawdown={cfg.get('max_abs_drawdown')} min_winrate={cfg.get('min_winrate')}",
    )

    print(f"\nTop{top_n}:")
    for i, row in enumerate(items[:top_n], 1):
        violations = row.get("constraint_violations") or []
        print(
            f"{i}. {row.get('name')} "
            f"reward={_fmt_num(row.get('reward'))} "
            f"profit_pct={_fmt_num(row.get('profit_total_pct'))}% "
            f"trades={row.get('trades')} "
            f"winrate={_fmt_num(row.get('winrate'))} "
            f"max_dd={_fmt_num(row.get('max_drawdown_abs'))} "
            f"constraints_ok={row.get('constraints_ok')} "
            f"violations={violations}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
