#!/usr/bin/env python3
"""Print live Strategy Loop progress with research and Freqtrade metrics."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as exc:
        return {"_error": f"invalid json: {exc}"}


def metric(metrics: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = metrics.get(name)
        if value is not None:
            return value
    return None


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def candidate_name(row: Mapping[str, Any]) -> str:
    candidate = row.get("candidate")
    if isinstance(candidate, Mapping):
        return str(candidate.get("name") or "-")
    return "-"


def metrics_line(label: str, metrics: Mapping[str, Any]) -> str:
    return (
        f"{label}: "
        f"profit={fmt(metric(metrics, 'profit_pct', 'total_return_pct'))}% "
        f"dd={fmt(metric(metrics, 'max_drawdown_pct'))}% "
        f"p/dd={fmt(metric(metrics, 'profit_over_max_drawdown'))} "
        f"trades={fmt(metric(metrics, 'trades'), 0)} "
        f"liq={fmt(metric(metrics, 'simulated_liquidations'), 0)} "
        f"turnover={fmt(metric(metrics, 'avg_turnover'))}"
    )


def row_block(label: str, row: Mapping[str, Any]) -> list[str]:
    if not row:
        return [f"{label}: -"]
    research = row.get("research_metrics") if isinstance(row.get("research_metrics"), Mapping) else row.get("metrics")
    if not isinstance(research, Mapping):
        research = {}
    freqtrade = row.get("freqtrade_metrics") if isinstance(row.get("freqtrade_metrics"), Mapping) else {}
    components = row.get("score_components") if isinstance(row.get("score_components"), Mapping) else {}
    return [
        (
            f"{label}: iter={fmt(row.get('iteration'), 0)} "
            f"name={candidate_name(row)} "
            f"score={fmt(row.get('score'))} "
            f"ok={fmt(row.get('constraints_ok'))} "
            f"diag={row.get('diagnostics') or '-'}"
        ),
        (
            "  components: "
            f"research={fmt(components.get('research_score'))} "
            f"freqtrade={fmt(components.get('freqtrade_score'))} "
            f"composite={fmt(components.get('composite_score'))}"
        ),
        "  " + metrics_line("research", research),
        "  " + metrics_line("freqtrade", freqtrade),
    ]


def render(run_id: str, root: Path) -> tuple[str, bool]:
    run_root = root / "artifacts" / "factor_strategy_loop" / run_id
    checkpoint = load_json(run_root / "checkpoint.json", {})
    leaderboard = load_json(run_root / "leaderboard.json", {"rows": []})

    rows = leaderboard.get("rows") if isinstance(leaderboard, Mapping) else []
    rows = rows if isinstance(rows, list) else []
    state = checkpoint.get("state") if isinstance(checkpoint, Mapping) else {}
    config = checkpoint.get("config") if isinstance(checkpoint, Mapping) else {}
    state = state if isinstance(state, Mapping) else {}
    config = config if isinstance(config, Mapping) else {}

    max_iterations = int(config.get("max_iterations") or 0)
    iteration = int(state.get("iteration") or 0)
    phase = state.get("phase") or "-"
    best_row = max((r for r in rows if isinstance(r, Mapping)), key=lambda r: float(r.get("score") or float("-inf")), default={})
    latest = rows[-1] if rows and isinstance(rows[-1], Mapping) else {}
    completed = bool(max_iterations and (len(rows) >= max_iterations or iteration > max_iterations))

    lines = [
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] run={run_id}",
        (
            f"state: iter={iteration}/{max_iterations or '-'} phase={phase} rows={len(rows)} "
            f"status={state.get('status') or '-'} score_mode={config.get('score_mode') or '-'} "
            f"promote_policy={config.get('promote_policy') or '-'} completed={fmt(completed)}"
        ),
    ]
    lines.extend(row_block("latest", latest))
    lines.extend(row_block("best", best_row))
    return "\n".join(lines), completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--interval", type=float, default=120.0)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--stop-when-complete", action="store_true")
    args = parser.parse_args()

    while True:
        text, completed = render(args.run_id, args.root.resolve())
        print(text, flush=True)
        print("", flush=True)
        if args.once or (args.stop_when_complete and completed):
            break
        time.sleep(max(1.0, float(args.interval)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
