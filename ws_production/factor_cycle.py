from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (str(SRC), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _run_command(cmd: list[str], *, cwd: Path = ROOT) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "cmd": [str(item) for item in cmd],
        "cwd": str(cwd),
        "returncode": int(proc.returncode),
        "elapsed_sec": round(time.monotonic() - started, 3),
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
    }


def _ensure_ok(step: str, result: dict[str, Any]) -> None:
    if int(result.get("returncode") or 0) == 0:
        return
    raise RuntimeError(f"{step} failed with exit code {result['returncode']}")


def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
    files = sorted(directory.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _choose_top_expression(expressions_path: Path) -> dict[str, Any]:
    payload = _load_json(expressions_path)
    expressions = list(payload.get("expressions") or [])
    if not expressions:
        raise ValueError(f"No expressions found in {expressions_path}")

    def _score(item: dict[str, Any]) -> tuple[float, str]:
        raw = item.get("score")
        try:
            score = float(raw)
        except Exception:
            score = float("-inf")
        return (score, str(item.get("name") or ""))

    top = sorted(expressions, key=_score, reverse=True)[0]
    return {
        "name": str(top.get("name") or ""),
        "expression": str(top.get("expression") or ""),
        "score": top.get("score"),
        "origin": top.get("origin"),
        "exchange": payload.get("exchange"),
        "timeframe": payload.get("timeframe"),
        "pairs": list(payload.get("pairs") or []),
        "feature_file": payload.get("feature_file"),
    }


def _build_factor_eval_dataset(
    *,
    freqai_config: Path,
    feature_file: Path,
    out_path: Path,
    pair: Optional[str] = None,
) -> dict[str, Any]:
    import pandas as pd

    from agent_market.freqai.features import apply_configured_features

    cfg = _load_json(freqai_config)
    feature_cfg = _load_json(feature_file)
    exchange_cfg = cfg.get("exchange") if isinstance(cfg.get("exchange"), dict) else {}
    timeframe = str(cfg.get("timeframe") or "1h")
    exchange = str(exchange_cfg.get("name") or "gate")
    pairs = [str(item) for item in (exchange_cfg.get("pair_whitelist") or []) if str(item).strip()]
    if pair:
        candidates = [str(pair)]
    else:
        candidates = pairs
    if not candidates:
        raise ValueError(f"No pairs configured in {freqai_config}")

    datadir = Path(str(cfg.get("datadir") or "user_data/data"))
    if not datadir.is_absolute():
        datadir = (ROOT / datadir).resolve()

    selected_pair: Optional[str] = None
    data_path: Optional[Path] = None
    for candidate in candidates:
        maybe = datadir / exchange / f"{candidate.replace('/', '_')}-{timeframe}.feather"
        if maybe.exists():
            selected_pair = candidate
            data_path = maybe
            break
    if selected_pair is None or data_path is None:
        raise FileNotFoundError(
            f"No market data found for pairs={candidates} under {(datadir / exchange).resolve()}"
        )

    frame = pd.read_feather(data_path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    frame = apply_configured_features(frame, feature_cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    return {
        "pair": selected_pair,
        "exchange": exchange,
        "timeframe": timeframe,
        "rows": int(len(frame)),
        "features_parquet": str(out_path.resolve()),
    }


def _should_use_llm(llm_enabled: Optional[bool]) -> bool:
    if llm_enabled is not None:
        return bool(llm_enabled)
    env = os.environ.get("WS_FACTOR_LLM_ENABLED")
    if env is not None:
        return env.strip().lower() not in {"", "0", "false", "no"}
    return shutil.which("opencode") is not None


def run_factor_cycle(
    *,
    workspace: str | Path = ROOT / "ws_production",
    model: Optional[str] = None,
    freqai_config: str | Path = ROOT / "user_data" / "config_freqai_gate_1h.json",
    feature_file: str | Path = ROOT / "user_data" / "freqai_features_real.json",
    expressions_file: str | Path = ROOT / "user_data" / "freqai_expressions_selected.json",
    llm_enabled: Optional[bool] = None,
    top_n: int = 30,
    llm_count: int = 30,
    llm_loops: int = 2,
    llm_temperature: float = 0.25,
    llm_max_turns: int = 12,
    min_samples: int = 80,
    permutations: int = 50,
    eval_rows: int = 4096,
) -> dict[str, Any]:
    workspace_path = Path(workspace).resolve()
    results_dir = workspace_path / "results"
    report_dir = results_dir / "factor_validation"
    factor_eval_root = results_dir / "factor_eval"
    summary_path = results_dir / f"factor_cycle_{_timestamp()}.json"
    latest_path = results_dir / "factor_cycle_latest.json"
    freqai_config_path = Path(freqai_config).resolve()
    feature_file_path = Path(feature_file).resolve()
    expressions_file_path = Path(expressions_file).resolve()

    summary: dict[str, Any] = {
        "timestamp": _utcnow(),
        "workspace": str(workspace_path),
        "ok": False,
        "config": {
            "freqai_config": str(freqai_config_path),
            "feature_file": str(feature_file_path),
            "expressions_file": str(expressions_file_path),
            "llm_enabled": _should_use_llm(llm_enabled),
            "model": str(model or os.environ.get("OPENCODE_MODEL") or "custom/gpt-5.2"),
            "top_n": int(top_n),
            "min_samples": int(min_samples),
            "permutations": int(permutations),
            "eval_rows": int(eval_rows),
        },
        "steps": {},
    }

    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        factor_eval_root.mkdir(parents=True, exist_ok=True)

        feature_cmd = [
            sys.executable,
            str((ROOT / "scripts" / "freqai_feature_agent.py").resolve()),
            "--config",
            _repo_rel(freqai_config_path),
            "--output",
            _repo_rel(feature_file_path),
            "--timeframe",
            str(_load_json(freqai_config_path).get("timeframe") or "1h"),
        ]
        feature_result = _run_command(feature_cmd)
        summary["steps"]["feature"] = feature_result
        _ensure_ok("feature", feature_result)

        expression_cmd = [
            sys.executable,
            str((ROOT / "scripts" / "freqai_expression_agent.py").resolve()),
            "--config",
            _repo_rel(freqai_config_path),
            "--feature-file",
            _repo_rel(feature_file_path),
            "--output",
            _repo_rel(expressions_file_path),
            "--timeframe",
            str(_load_json(freqai_config_path).get("timeframe") or "1h"),
            "--mine",
            "--top-n",
            str(int(top_n)),
            "--min-samples",
            str(int(min_samples)),
            "--feedback-top",
            "8",
        ]
        if _should_use_llm(llm_enabled):
            resolved_model = str(model or os.environ.get("OPENCODE_MODEL") or "custom/gpt-5.2")
            expression_cmd.extend(
                [
                    "--llm-enabled",
                    "--llm-provider",
                    "opencode",
                    "--llm-model",
                    resolved_model,
                    "--llm-workspace",
                    ".",
                    "--llm-count",
                    str(int(llm_count)),
                    "--llm-loops",
                    str(int(llm_loops)),
                    "--llm-max-turns",
                    str(int(llm_max_turns)),
                    "--llm-temperature",
                    str(float(llm_temperature)),
                    "--llm-fallback",
                ]
            )
        else:
            expression_cmd.append("--no-llm")

        feedback_path = ROOT / "user_data" / "llm_feedback" / "latest_backtest_summary.json"
        if feedback_path.exists():
            expression_cmd.extend(["--feedback", _repo_rel(feedback_path)])

        expression_result = _run_command(expression_cmd)
        summary["steps"]["expression"] = expression_result
        _ensure_ok("expression", expression_result)

        verify_cmd = [
            sys.executable,
            str((ROOT / "scripts" / "verify_factors.py").resolve()),
            "--config",
            _repo_rel(freqai_config_path),
            "--feature-file",
            _repo_rel(feature_file_path),
            "--expressions-file",
            _repo_rel(expressions_file_path),
            "--report-dir",
            _repo_rel(report_dir),
            "--permutations",
            str(int(permutations)),
            "--top",
            str(max(5, min(int(top_n), 20))),
        ]
        verify_result = _run_command(verify_cmd)
        summary["steps"]["verify_factors"] = verify_result
        _ensure_ok("verify_factors", verify_result)

        validation_report = _latest_file(report_dir, "factor_validation_*.json")
        if validation_report is None:
            raise FileNotFoundError(f"No factor validation report found under {report_dir}")

        top_expression = _choose_top_expression(expressions_file_path)
        if not str(top_expression.get("expression") or "").strip():
            raise ValueError(f"Top expression in {expressions_file_path} is empty")

        factor_eval_dir = factor_eval_root / summary_path.stem.replace("factor_cycle_", "")
        dataset_path = factor_eval_dir / "features.parquet"
        dataset_info = _build_factor_eval_dataset(
            freqai_config=freqai_config_path,
            feature_file=feature_file_path,
            out_path=dataset_path,
            pair=(top_expression.get("pairs") or [None])[0],
        )
        eval_cmd = [
            sys.executable,
            str((ROOT / "scripts" / "factor_eval.py").resolve()),
            "--expression",
            str(top_expression["expression"]),
            "--features-parquet",
            str(dataset_path.resolve()),
            "--out-dir",
            str(factor_eval_dir.resolve()),
            "--rows",
            str(int(eval_rows)),
            "--seed",
            "7",
        ]
        eval_result = _run_command(eval_cmd)
        summary["steps"]["factor_eval"] = eval_result
        _ensure_ok("factor_eval", eval_result)

        summary["validation_report"] = str(validation_report.resolve())
        summary["top_expression"] = top_expression
        summary["factor_eval_dataset"] = dataset_info
        summary["factor_eval_artifacts"] = {
            "dir": str(factor_eval_dir.resolve()),
            "factor_eval_meta": str((factor_eval_dir / "factor_eval_meta.json").resolve()),
            "factor_scores_json": str((factor_eval_dir / "factor_scores.json").resolve()),
            "pareto_csv": str((factor_eval_dir / "pareto.csv").resolve()),
        }
        summary["ok"] = True
    except Exception as exc:
        summary["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc()[-4000:],
        }
    finally:
        _write_json(summary_path, summary)
        _write_json(latest_path, summary)

    return summary


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run one ws_production factor mining + evaluation cycle.")
    parser.add_argument("--workspace", default="ws_production", help="Workspace directory to attach artifacts to.")
    parser.add_argument("--model", default=None, help="OpenCode model passed to expression mining when LLM is enabled.")
    parser.add_argument(
        "--freqai-config",
        default="user_data/config_freqai_gate_1h.json",
        help="FreqAI config aligned with the ws_production exchange/timeframe.",
    )
    parser.add_argument("--feature-file", default="user_data/freqai_features_real.json")
    parser.add_argument("--expressions-file", default="user_data/freqai_expressions_selected.json")
    parser.add_argument("--llm-enabled", action="store_true", help="Force LLM-enabled factor mining.")
    parser.add_argument("--no-llm", action="store_true", help="Force template-only factor mining.")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--llm-count", type=int, default=30)
    parser.add_argument("--llm-loops", type=int, default=2)
    parser.add_argument("--llm-temperature", type=float, default=0.25)
    parser.add_argument("--llm-max-turns", type=int, default=12)
    parser.add_argument("--min-samples", type=int, default=80)
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--eval-rows", type=int, default=4096)
    parser.add_argument("--strict", action="store_true", help="Return non-zero when the cycle fails.")
    args = parser.parse_args(argv)

    llm_enabled: Optional[bool] = None
    if args.llm_enabled:
        llm_enabled = True
    elif args.no_llm:
        llm_enabled = False

    summary = run_factor_cycle(
        workspace=args.workspace,
        model=args.model,
        freqai_config=ROOT / args.freqai_config,
        feature_file=ROOT / args.feature_file,
        expressions_file=ROOT / args.expressions_file,
        llm_enabled=llm_enabled,
        top_n=int(args.top_n),
        llm_count=int(args.llm_count),
        llm_loops=int(args.llm_loops),
        llm_temperature=float(args.llm_temperature),
        llm_max_turns=int(args.llm_max_turns),
        min_samples=int(args.min_samples),
        permutations=int(args.permutations),
        eval_rows=int(args.eval_rows),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("ok") or not args.strict:
        return 0
    return 1


__all__ = ["run_factor_cycle", "main", "_choose_top_expression", "_build_factor_eval_dataset"]
