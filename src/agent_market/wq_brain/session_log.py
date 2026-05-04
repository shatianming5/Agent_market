"""Structured per-run logging for WQ Brain (no external YAML dependency)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _yaml_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, str) and any(c in v for c in ":#{}[]|>&*!,\"'"):
        return "'" + v.replace("'", "''") + "'"
    return str(v)


def _yaml_dump(data: dict | list, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(data, dict):
        if not data:
            return "{}"
        lines = []
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                inner = _yaml_dump(v, indent + 1)
                lines.append(f"{pad}{k}:\n{inner}")
            else:
                lines.append(f"{pad}{k}: {_yaml_scalar(v)}")
        return "\n".join(lines)
    if isinstance(data, list):
        if not data:
            return "[]"
        lines = []
        for item in data:
            if isinstance(item, dict) and item:
                # Inline first key with "- ", subsequent keys indented to align
                item_lines = []
                for i, (k, v) in enumerate(item.items()):
                    prefix = f"{pad}- " if i == 0 else f"{pad}  "
                    if isinstance(v, (dict, list)):
                        inner = _yaml_dump(v, indent + 1)
                        item_lines.append(f"{prefix}{k}:\n{inner}")
                    else:
                        item_lines.append(f"{prefix}{k}: {_yaml_scalar(v)}")
                lines.append("\n".join(item_lines))
            elif isinstance(item, list):
                inner = _yaml_dump(item, indent + 1)
                lines.append(f"{pad}-\n{inner}")
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return _yaml_scalar(data)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def write_session_metadata(run_dir: Path, config: dict) -> None:
    meta = {
        "run_id": config.get("run_id", ""),
        "tag": config.get("tag", ""),
        "started_at": _now_utc(),
        "region": config.get("region", "USA"),
        "universe": config.get("universe", "TOP3000"),
        "max_iterations": config.get("max_iterations", 0),
        "batch_size": config.get("batch_size", 5),
        "decay": config.get("decay", 0),
        "model": config.get("model", ""),
        "hermes_provider": config.get("hermes_provider", ""),
    }
    (run_dir / "session_metadata.yml").write_text(
        _yaml_dump(meta), encoding="utf-8"
    )


def append_round(iter_dir: Path, record: dict) -> None:
    data = {
        "iteration": record.get("iteration"),
        "generated": record.get("generated", 0),
        "simulated": record.get("simulated", 0),
        "passed": record.get("passed", 0),
        "submitted": record.get("submitted", 0),
        "top_sharpe": record.get("top_sharpe"),
        "top_fitness": record.get("top_fitness"),
        "top_expr": record.get("top_expr"),
        "failure_summary": record.get("failure_summary", {}),
        "pool_size": record.get("pool_size", 0),
        "used_mutation": record.get("used_mutation", False),
        "kb_hits": record.get("kb_hits", 0),
    }
    (iter_dir / "round.yml").write_text(_yaml_dump(data), encoding="utf-8")


def write_final_summary(run_dir: Path, state: dict) -> None:
    cfg = state.get("config", {})
    history = state.get("score_history", [])

    rows = []
    for h in history:
        sh = f"{h['top_sharpe']:.2f}" if h.get("top_sharpe") is not None else "N/A"
        fi = f"{h['top_fitness']:.2f}" if h.get("top_fitness") is not None else "N/A"
        fail_items = list(h.get("failure_summary", {}).items())[:3]
        fail_str = ", ".join(f"{k}:{v}" for k, v in fail_items) or "—"
        rows.append(
            f"| {h.get('iteration','?'):>4} | {h.get('simulated',0):>3} "
            f"| {h.get('passed',0):>4} | {sh:>9} | {fi:>10} | {fail_str} |"
        )

    lines = [
        "# WQ Brain Run Summary\n",
        f"**Run ID:** {cfg.get('run_id', 'N/A')}  \n",
        f"**Tag:** {cfg.get('tag', 'N/A')}  \n",
        f"**Completed:** {_now_utc()}  \n",
        "\n## Stats\n",
        f"- Iterations completed: {state.get('iteration', 0)}\n",
        f"- Total generated: {state.get('total_generated', 0)}\n",
        f"- Total simulated: {state.get('total_simulated', 0)}\n",
        f"- Total passed QC: {state.get('total_passed', 0)}\n",
        f"- Total submitted: {state.get('total_submitted', 0)}\n",
        "\n## Per-Iteration History\n",
        "| Iter | Sim | Pass | Top Sharpe | Top Fitness | Top Failures |\n",
        "|-----:|----:|-----:|-----------:|------------:|:-------------|\n",
        *[r + "\n" for r in rows],
    ]
    (run_dir / "final_summary.md").write_text("".join(lines), encoding="utf-8")
