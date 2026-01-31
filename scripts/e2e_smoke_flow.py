#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Agent Flow end-to-end and verify required artifacts exist.")
    parser.add_argument(
        "--config",
        default="configs/agent_flow_kucoin_cpu_nollm.json",
        help="Agent Flow config JSON.",
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        default=["feature", "expression", "ml", "backtest"],
        help="Steps to run (default: feature expression ml backtest).",
    )
    args = parser.parse_args(argv)

    cfg_path = _resolve(args.config)
    if not cfg_path.exists():
        print(f"[e2e] missing config: {cfg_path}", file=sys.stderr)
        return 2

    try:
        from agent_market.demo_data import ensure_demo_ohlcv_from_flow_config  # type: ignore

        created = ensure_demo_ohlcv_from_flow_config(ROOT, cfg_path)
        for p in created:
            try:
                rel = p.relative_to(ROOT)
            except Exception:
                rel = p
            print(f"[e2e] demo ohlcv generated: {rel}")
    except Exception as exc:
        print(f"[e2e] WARN: demo data bootstrap skipped ({exc})", file=sys.stderr)

    flow_script = _resolve("scripts/agent_flow.py")
    cmd = [sys.executable, str(flow_script), "--config", str(cfg_path)]
    if args.steps:
        cmd += ["--steps"] + [str(s) for s in args.steps]
    print("[e2e] running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)  # noqa: S603,S607

    required_files = [
        "artifacts/run_meta.json",
        "user_data/freqai_features_real.json",
        "user_data/freqai_expressions_selected.json",
        "user_data/llm_feedback/latest_backtest_summary.json",
    ]
    missing: list[str] = []
    for rel in required_files:
        p = _resolve(rel)
        if not p.exists():
            missing.append(rel)

    models_dir = _resolve("artifacts/models")
    training_summaries = list(models_dir.rglob("training_summary.json")) if models_dir.exists() else []
    if not training_summaries:
        missing.append("artifacts/models/**/training_summary.json")

    bt_dir = _resolve("user_data/backtest_results")
    bt_zips = list(bt_dir.glob("backtest-result-*.zip")) if bt_dir.exists() else []
    if not bt_zips:
        missing.append("user_data/backtest_results/backtest-result-*.zip")

    if missing:
        print("[e2e] FAIL: missing required artifacts:", file=sys.stderr)
        for item in missing:
            print(" -", item, file=sys.stderr)
        return 1

    meta_path = _resolve("artifacts/run_meta.json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        print(f"[e2e] FAIL: invalid run_meta.json ({exc})", file=sys.stderr)
        return 1

    run_id = meta.get("run_id")
    cfg_hash = (meta.get("config") or {}).get("sha256")
    py_version = (meta.get("python") or {}).get("version")
    ft_version = (meta.get("freqtrade") or {}).get("version")
    if not (isinstance(run_id, str) and run_id):
        print("[e2e] FAIL: run_meta.json missing run_id", file=sys.stderr)
        return 1
    if not (isinstance(cfg_hash, str) and cfg_hash):
        print("[e2e] FAIL: run_meta.json missing config.sha256", file=sys.stderr)
        return 1
    if not (isinstance(py_version, str) and py_version):
        print("[e2e] FAIL: run_meta.json missing python.version", file=sys.stderr)
        return 1
    if not (isinstance(ft_version, str) and ft_version):
        print("[e2e] FAIL: run_meta.json missing freqtrade.version", file=sys.stderr)
        return 1

    latest_zip = max(bt_zips, key=lambda p: p.stat().st_mtime)
    latest_train = max(training_summaries, key=lambda p: p.stat().st_mtime)
    print("[e2e] OK")
    print("[e2e] run_id:", run_id)
    print("[e2e] latest backtest:", latest_zip.name)
    print("[e2e] latest training summary:", str(latest_train.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
