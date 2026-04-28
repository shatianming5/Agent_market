#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def _parse_iso8601(s: str | None) -> Optional[datetime]:
    if not s:
        return None
    raw = str(s).strip()
    if not raw:
        return None
    try:
        # `2026-04-07T05:43:06Z` or `...+00:00`
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _pct(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    xs = sorted(values)
    idx = int(round((len(xs) - 1) * q))
    return float(xs[max(0, min(len(xs) - 1, idx))])


@dataclass(frozen=True)
class FlowRun:
    run_id: str
    log_file: Optional[str]
    returncode: int


def _run_agent_flow(
    *,
    config_path: Path,
    steps: list[str],
    log_dir: Path,
    pythonpath: Optional[str],
) -> FlowRun:
    flow_script = REPO_ROOT / "scripts" / "agent_flow.py"
    cmd = [sys.executable, str(flow_script), "--config", str(config_path), "--log-dir", str(log_dir)]
    if steps:
        cmd += ["--steps", *steps]

    env = dict(os.environ)
    if pythonpath is None:
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = pythonpath

    print(f"[acceptance] running: {' '.join(cmd)}")
    if pythonpath is not None:
        print(f"[acceptance] env: PYTHONPATH={pythonpath}")

    run_id: Optional[str] = None
    log_file: Optional[str] = None
    log_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if run_id is None:
            m = re.search(r"\[FLOW\]\s+RUN_ID\s+([0-9a-f]{8,})", line)
            if m:
                run_id = m.group(1)
        if log_file is None and "Agent Flow log file:" in line:
            # Example: "Agent Flow log file: /abs/path.log"
            parts = line.split("Agent Flow log file:", 1)
            if len(parts) == 2:
                log_file = parts[1].strip()

    rc = int(proc.wait())

    # Fallback to latest run_meta.json if run_id wasn't captured.
    if run_id is None:
        try:
            meta = _read_json(REPO_ROOT / "artifacts" / "run_meta.json")
            candidate = str(meta.get("run_id") or "").strip()
            if candidate:
                run_id = candidate
        except Exception:
            pass

    if not run_id:
        raise RuntimeError("agent_flow finished but run_id could not be determined")

    return FlowRun(run_id=str(run_id), log_file=log_file, returncode=rc)


def _count_provider_flakes(log_path: Optional[str]) -> dict[str, int]:
    if not log_path:
        return {"http_503": 0, "service_unavailable": 0}
    p = Path(log_path)
    if not p.exists():
        return {"http_503": 0, "service_unavailable": 0}
    text = p.read_text(encoding="utf-8", errors="replace")
    return {
        "http_503": int(len(re.findall(r"\b503\b", text))),
        "service_unavailable": int(len(re.findall(r"Service Unavailable", text, flags=re.IGNORECASE))),
    }


def _verify_strategy_miner_run(run_id: str) -> dict[str, Any]:
    script = REPO_ROOT / "scripts" / "verify_strategy_miner_run.py"
    cmd = [sys.executable, str(script), "--run-id", run_id, "--top", "1"]
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return {"ok": proc.returncode == 0, "returncode": int(proc.returncode), "output_tail": out.strip()[-2000:]}


def _collect_experiment_timings(*, run_id: str, experiment_id: str) -> dict[str, Any]:
    reg_path = REPO_ROOT / "artifacts" / "experiment_registry.jsonl"
    if not reg_path.exists():
        return {"ok": False, "error": "experiment_registry_missing"}

    total_started: Optional[datetime] = None
    total_ended: Optional[datetime] = None
    by_step: dict[str, float] = {}
    full_run_seconds: Optional[float] = None

    try:
        for raw in reg_path.read_text(encoding="utf-8", errors="replace").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if str(row.get("run_id") or "") != run_id:
                continue

            started = _parse_iso8601(row.get("started_at"))
            ended = _parse_iso8601(row.get("ended_at"))
            if started and (total_started is None or started < total_started):
                total_started = started
            if ended and (total_ended is None or ended > total_ended):
                total_ended = ended

            # Top-level experiment entry.
            if str(row.get("experiment_id") or "") == experiment_id and started and ended:
                full_run_seconds = (ended - started).total_seconds()

            step = row.get("step")
            if step and started and ended:
                by_step[str(step)] = (ended - started).total_seconds()
    except Exception as exc:
        return {"ok": False, "error": f"registry_parse_failed: {exc}"}

    total_seconds = (total_ended - total_started).total_seconds() if (total_started and total_ended) else None
    return {
        "ok": True,
        "full_run_seconds": full_run_seconds,
        "total_seconds": total_seconds,
        "by_step_seconds": by_step,
    }


def _collect_run_evidence(*, run_id: str, experiment_id: str, flow_log_file: Optional[str]) -> dict[str, Any]:
    run_dir = REPO_ROOT / "artifacts" / "runs" / run_id
    miner_dir = run_dir / "strategy_miner"
    meta_path = run_dir / "run_meta.json"
    miner_summary = miner_dir / "strategy_miner_summary.json"

    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = _read_json(meta_path)
        except Exception:
            meta = {}

    summary: dict[str, Any] = {}
    if miner_summary.exists():
        try:
            summary = _read_json(miner_summary)
        except Exception:
            summary = {}

    best = summary.get("best_candidate") if isinstance(summary.get("best_candidate"), dict) else None
    bt = dict(best.get("backtest_summary") or {}) if isinstance(best, dict) else {}

    freqtrade = meta.get("freqtrade") if isinstance(meta.get("freqtrade"), dict) else {}
    freqtrade_ok = bool(freqtrade.get("ok")) if isinstance(freqtrade, dict) else None
    freqtrade_err = str(freqtrade.get("stderr") or "")[:500] if isinstance(freqtrade, dict) else ""

    timings = _collect_experiment_timings(run_id=run_id, experiment_id=experiment_id)
    flakes = _count_provider_flakes(flow_log_file)
    verify = _verify_strategy_miner_run(run_id)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "flow_log_file": flow_log_file,
        "freqtrade_ok": freqtrade_ok,
        "freqtrade_stderr_head": freqtrade_err,
        "best_candidate": {
            "name": best.get("name") if isinstance(best, dict) else None,
            "candidate_type": best.get("candidate_type") if isinstance(best, dict) else None,
            "candidate_family": best.get("candidate_family") if isinstance(best, dict) else None,
            "profit_total_pct": bt.get("profit_total_pct"),
            "trades": bt.get("trades"),
            "realistic_sharpe": bt.get("realistic_sharpe", bt.get("sharpe")),
            "profit_factor": bt.get("profit_factor"),
            "winrate": bt.get("winrate"),
            "max_drawdown_pct": bt.get("max_drawdown_pct"),
        },
        "timings": timings,
        "provider_flakes": flakes,
        "verify": verify,
        "artifacts": {
            "strategy_miner_summary": str(miner_summary) if miner_summary.exists() else None,
            "run_meta": str(meta_path) if meta_path.exists() else None,
        },
    }


def _soak_mode(
    *,
    base_config_path: Path,
    runs: int,
    pythonpath: Optional[str],
    out_path: Path,
    steps: list[str],
) -> int:
    ts = _now_ts()
    log_dir = REPO_ROOT / "runtime_logs" / f"strategy_factory_soak_{ts}"
    cfg_dir = REPO_ROOT / "runtime_configs" / f"strategy_factory_soak_{ts}"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _read_json(base_config_path)
    base_experiment_id = str(((base_cfg.get("experiment") or {}).get("experiment_id")) or "strategy-factory").strip()

    results: list[dict[str, Any]] = []
    failures: list[str] = []

    for i in range(int(runs)):
        exp_id = f"{base_experiment_id}-soak-{ts}-{i:02d}"
        cfg = json.loads(json.dumps(base_cfg))
        _deep_update(
            cfg,
            {
                "experiment": {
                    "experiment_id": exp_id,
                    "hypothesis": f"Soak run {i+1}/{runs}: verify end-to-end stability under PYTHONPATH shadowing and provider flakes.",
                }
            },
        )

        cfg_path = cfg_dir / f"{exp_id}.json"
        _write_json(cfg_path, cfg)

        started = time.time()
        run = _run_agent_flow(config_path=cfg_path, steps=steps, log_dir=log_dir, pythonpath=pythonpath)
        elapsed = time.time() - started

        ev = _collect_run_evidence(run_id=run.run_id, experiment_id=exp_id, flow_log_file=run.log_file)
        ev["experiment_id"] = exp_id
        ev["returncode"] = run.returncode
        ev["wall_seconds"] = elapsed
        results.append(ev)

        if run.returncode != 0:
            failures.append(f"{exp_id}: returncode={run.returncode}")
        if ev.get("freqtrade_ok") is False:
            failures.append(f"{exp_id}: freqtrade_ok=false")
        if not (ev.get("best_candidate") or {}).get("name"):
            failures.append(f"{exp_id}: best_candidate=null")
        if not (ev.get("verify") or {}).get("ok"):
            failures.append(f"{exp_id}: verify_failed")

    wall = [float(r.get("wall_seconds") or 0.0) for r in results if float(r.get("wall_seconds") or 0.0) > 0]
    full = [
        float((r.get("timings") or {}).get("full_run_seconds") or 0.0)
        for r in results
        if float((r.get("timings") or {}).get("full_run_seconds") or 0.0) > 0
    ]
    successes = [r for r in results if (r.get("returncode") == 0 and (r.get("best_candidate") or {}).get("name"))]
    ft_ok = sum(1 for r in results if r.get("freqtrade_ok") is True)
    verify_ok = sum(1 for r in results if (r.get("verify") or {}).get("ok") is True)
    flakes_503 = sum(int((r.get("provider_flakes") or {}).get("http_503") or 0) for r in results)

    manifest = {
        "schema_version": "1.0",
        "mode": "soak",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_config": str(base_config_path),
        "pythonpath": pythonpath,
        "runs_requested": int(runs),
        "results": results,
        "summary": {
            "runs_total": len(results),
            "runs_success": len(successes),
            "freqtrade_ok": ft_ok,
            "verify_ok": verify_ok,
            "provider_503_total": flakes_503,
            "wall_seconds": {
                "min": min(wall) if wall else None,
                "p50": _pct(wall, 0.5),
                "p90": _pct(wall, 0.9),
                "max": max(wall) if wall else None,
                "mean": statistics.mean(wall) if wall else None,
            },
            "full_run_seconds": {
                "min": min(full) if full else None,
                "p50": _pct(full, 0.5),
                "p90": _pct(full, 0.9),
                "max": max(full) if full else None,
                "mean": statistics.mean(full) if full else None,
            },
            "failures": failures[:200],
        },
        "paths": {
            "config_dir": str(cfg_dir),
            "log_dir": str(log_dir),
        },
    }
    _write_json(out_path, manifest)

    md_lines = [
        f"# Strategy Factory Soak Summary ({ts})",
        "",
        f"- Base config: `{base_config_path}`",
        f"- PYTHONPATH: `{pythonpath}`",
        f"- Runs: `{len(results)}` success: `{len(successes)}` verify_ok: `{verify_ok}` freqtrade_ok: `{ft_ok}`",
        f"- Total 503 hits in flow logs: `{flakes_503}`",
        "",
        "## Runs",
        "",
    ]
    for r in results:
        best = r.get("best_candidate") or {}
        name = best.get("name") or "null"
        profit = best.get("profit_total_pct")
        trades = best.get("trades")
        md_lines.append(f"- `{r.get('experiment_id')}` run_id=`{r.get('run_id')}` best=`{name}` profit=`{profit}` trades=`{trades}` rc=`{r.get('returncode')}`")
    _write_text(out_path.with_suffix(".md"), "\n".join(md_lines))

    return 0 if not failures else 1


def _matrix_mode(
    *,
    base_config_path: Path,
    pythonpath: Optional[str],
    out_path: Path,
    steps: list[str],
) -> int:
    ts = _now_ts()
    log_dir = REPO_ROOT / "runtime_logs" / f"strategy_factory_matrix_{ts}"
    cfg_dir = REPO_ROOT / "runtime_configs" / f"strategy_factory_matrix_{ts}"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _read_json(base_config_path)
    base_experiment_id = str(((base_cfg.get("experiment") or {}).get("experiment_id")) or "strategy-factory").strip()

    # Curated matrix: representative cases (not a full Cartesian product).
    cases: list[tuple[str, dict[str, Any]]] = [
        ("ml_5m_baseline", {}),
        (
            "ml_5m_universe_small",
            {
                "micro_feature": {"pairs": ["BTC/USDT", "ETH/USDT"]},
                "factor_compile": {"spec": {"meta": {"universe": ["BTC/USDT", "ETH/USDT"]}}},
                "strategy_miner": {
                    "model_training_pairs": ["BTC/USDT", "ETH/USDT"],
                    "min_trades": 80,
                    "min_acceptable_trades": 40,
                    "target_trades": 160,
                    # With a small training universe but a wider backtest whitelist,
                    # keep constraints loose to avoid best_candidate=null.
                    "min_profit_factor": 0.3,
                    "min_winrate": 0.2,
                    "min_pair_profit_pct": -10.0,
                },
            },
        ),
        (
            "ml_15m_timeframe",
            {
                "micro_feature": {"config": "user_data/config_freqai_gate_15m.json", "timeframe": "15m"},
                "factor_compile": {"spec": {"meta": {"timeframe": "15m"}}},
                "strategy_miner": {
                    "freqtrade_config": "user_data/config_freqai_gate_15m.json",
                    "max_strategy_timeframe": "1h",
                    "allowed_informative_timeframes": ["1h"],
                    "min_trades": 80,
                    "min_acceptable_trades": 40,
                    "target_trades": 200,
                },
            },
        ),
        (
            "ml_1h_timeframe",
            {
                "micro_feature": {"config": "user_data/config_freqai_gate_1h.json", "timeframe": "1h"},
                "factor_compile": {"spec": {"meta": {"timeframe": "1h"}}},
                "strategy_miner": {
                    "freqtrade_config": "user_data/config_freqai_gate_1h.json",
                    "max_strategy_timeframe": "4h",
                    "allowed_informative_timeframes": ["4h"],
                    "min_trades": 20,
                    # Acceptance intent: verify the 1h pipeline doesn't flake.
                    # Loosen profit constraints to avoid best_candidate=null with 1 iteration.
                    "min_profit_factor": 0.3,
                    "min_pair_profit_pct": -5.0,
                    "min_winrate": 0.4,
                    "target_trades": 60,
                    "min_acceptable_trades": 10,
                },
            },
        ),
        (
            "rule_5m_mean_reversion",
            {
                "strategy_miner": {
                    "candidate_types": ["rule"],
                    # Mean reversion tends to produce denser trades than breakout under short timeranges.
                    "search_families": ["rule/mean-reversion"],
                    # Acceptance intent: plumbing smoke. Keep gates extremely loose to avoid best_score=-inf
                    # when an LLM-generated strategy produces sparse/zero trades.
                    "min_trades": 0,
                    "min_acceptable_trades": 0,
                    "min_profit_factor": 0.0,
                    "min_winrate": 0.0,
                    "min_pair_profit_pct": -100.0,
                    "max_abs_drawdown": 2000.0,
                    "target_trades": 120,
                    "roi_target_min_pct": 0.05,
                    "roi_target_max_pct": 0.8,
                },
            },
        ),
        (
            "dl_5m_pytorch_mlp",
            {
                "strategy_miner": {
                    "enable_dl": True,
                    "candidate_types": ["dl"],
                    "search_families": ["dl/pytorch_mlp"],
                    # DL quality is volatile under tiny budgets. Keep this as a stability smoke.
                    "min_trades": 0,
                    "min_acceptable_trades": 0,
                    "min_profit_factor": 0.0,
                    "min_winrate": 0.0,
                    "min_pair_profit_pct": -100.0,
                    "max_abs_drawdown": 2000.0,
                    "target_trades": 180,
                },
            },
        ),
        (
            "rl_5m_ppo",
            {
                "strategy_miner": {
                    "enable_rl": True,
                    "candidate_types": ["rl"],
                    "search_families": ["rl/ppo"],
                    "rl_total_timesteps": 2000,
                    # RL is a pure plumbing smoke here: keep gates extremely loose to avoid best_score=-inf.
                    "min_trades": 0,
                    "min_acceptable_trades": 0,
                    "min_profit_factor": 0.0,
                    "min_winrate": 0.0,
                    "min_pair_profit_pct": -100.0,
                    "max_abs_drawdown": 2000.0,
                    "target_trades": 120,
                },
            },
        ),
        (
            "ml_5m_walkforward",
            {
                "strategy_miner": {
                    "walkforward_enabled": True,
                    "walkforward_folds": 2,
                    "walkforward_train_ratio": 0.6,
                    "min_trades": 100,
                    "min_acceptable_trades": 60,
                },
            },
        ),
        (
            "ml_5m_holdout",
            {
                "strategy_miner": {
                    "selection_timerange": "20260301-20260322",
                    "holdout_timerange": "20260323-20260329",
                    # Keep holdout path exercised but avoid frequent demotions under short windows.
                    "holdout_delta_max_pct": 50.0,
                },
            },
        ),
        (
            "ml_5m_hyperopt_enabled",
            {
                "strategy_miner": {
                    "hyperopt_enabled": True,
                    "hyperopt_epochs": 10,
                    "hyperopt_jobs": 1,
                    "hyperopt_min_trades": 30,
                },
            },
        ),
    ]

    results: list[dict[str, Any]] = []
    failures: list[str] = []

    for case_id, overrides in cases:
        exp_id = f"{base_experiment_id}-matrix-{ts}-{case_id}"
        cfg = json.loads(json.dumps(base_cfg))
        _deep_update(
            cfg,
            {
                "experiment": {
                    "experiment_id": exp_id,
                    "hypothesis": f"Matrix case {case_id}: verify end-to-end stability under toggles/timeframe/candidate_type.",
                }
            },
        )
        _deep_update(cfg, overrides)

        cfg_path = cfg_dir / f"{exp_id}.json"
        _write_json(cfg_path, cfg)

        started = time.time()
        run = _run_agent_flow(config_path=cfg_path, steps=steps, log_dir=log_dir, pythonpath=pythonpath)
        elapsed = time.time() - started

        ev = _collect_run_evidence(run_id=run.run_id, experiment_id=exp_id, flow_log_file=run.log_file)
        ev["experiment_id"] = exp_id
        ev["case_id"] = case_id
        ev["overrides"] = overrides
        ev["returncode"] = run.returncode
        ev["wall_seconds"] = elapsed
        results.append(ev)

        if run.returncode != 0:
            failures.append(f"{case_id}: returncode={run.returncode}")
        if ev.get("freqtrade_ok") is False:
            failures.append(f"{case_id}: freqtrade_ok=false")
        if not (ev.get("verify") or {}).get("ok"):
            failures.append(f"{case_id}: verify_failed")

    manifest = {
        "schema_version": "1.0",
        "mode": "matrix",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_config": str(base_config_path),
        "pythonpath": pythonpath,
        "cases": [c for c, _ in cases],
        "results": results,
        "summary": {
            "cases_total": len(results),
            "failures": failures[:200],
        },
        "paths": {
            "config_dir": str(cfg_dir),
            "log_dir": str(log_dir),
        },
    }
    _write_json(out_path, manifest)

    md_lines = [
        f"# Strategy Factory Matrix Summary ({ts})",
        "",
        f"- Base config: `{base_config_path}`",
        f"- PYTHONPATH: `{pythonpath}`",
        "",
        "## Cases",
        "",
    ]
    for r in results:
        best = r.get("best_candidate") or {}
        name = best.get("name") or "null"
        profit = best.get("profit_total_pct")
        trades = best.get("trades")
        md_lines.append(
            f"- `{r.get('case_id')}` run_id=`{r.get('run_id')}` best=`{name}` profit=`{profit}` trades=`{trades}` rc=`{r.get('returncode')}`"
        )
    _write_text(out_path.with_suffix(".md"), "\n".join(md_lines))

    return 0 if not failures else 1


def _resolve_global_strategy_kb_path(cfg: dict[str, Any]) -> Path:
    raw = str((((cfg.get("strategy_miner") or {}).get("global_strategy_knowledge_base_path")) or "")).strip()
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()
    return (REPO_ROOT / "artifacts" / "control_plane" / "strategy_memory" / "knowledge_base.json").resolve()


def _closure_mode(
    *,
    base_config_path: Path,
    pythonpath: Optional[str],
    out_path: Path,
    steps: list[str],
) -> int:
    """Two-run closure: run1 must write a new strategy_card, run2 must retrieve it."""
    ts = _now_ts()
    log_dir = REPO_ROOT / "runtime_logs" / f"strategy_factory_closure_{ts}"
    cfg_dir = REPO_ROOT / "runtime_configs" / f"strategy_factory_closure_{ts}"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _read_json(base_config_path)
    base_experiment_id = str(((base_cfg.get("experiment") or {}).get("experiment_id")) or "strategy-factory").strip()
    global_kb_path = _resolve_global_strategy_kb_path(base_cfg)

    # Run 1: seed a brand-new card.
    exp_seed = f"{base_experiment_id}-closure-{ts}-seed"
    cfg_seed = json.loads(json.dumps(base_cfg))
    _deep_update(
        cfg_seed,
        {
            "experiment": {
                "experiment_id": exp_seed,
                "hypothesis": "Seed a new strategy_card into the global strategy control-plane memory.",
            }
        },
    )
    cfg_seed_path = cfg_dir / f"{exp_seed}.json"
    _write_json(cfg_seed_path, cfg_seed)

    seed_run = _run_agent_flow(config_path=cfg_seed_path, steps=steps, log_dir=log_dir, pythonpath=pythonpath)

    seed_cards: list[dict[str, Any]] = []
    try:
        if global_kb_path.exists():
            kb = _read_json(global_kb_path)
            for card in list(kb.get("strategy_cards") or []):
                if isinstance(card, dict) and str(card.get("run_id") or "") == seed_run.run_id:
                    seed_cards.append(card)
    except Exception:
        seed_cards = []

    # Run 2: must retrieve the seed card in its prompt snapshot.
    exp_follow = f"{base_experiment_id}-closure-{ts}-follow"
    cfg_follow = json.loads(json.dumps(base_cfg))
    _deep_update(
        cfg_follow,
        {
            "experiment": {
                "experiment_id": exp_follow,
                "hypothesis": "Must retrieve the previous run's newly created strategy_card (control-plane reuse proof).",
            }
        },
    )
    cfg_follow_path = cfg_dir / f"{exp_follow}.json"
    _write_json(cfg_follow_path, cfg_follow)
    follow_run = _run_agent_flow(config_path=cfg_follow_path, steps=steps, log_dir=log_dir, pythonpath=pythonpath)

    follow_summary_path = (
        REPO_ROOT / "artifacts" / "runs" / follow_run.run_id / "strategy_miner" / "strategy_miner_summary.json"
    )
    retrieved_seed = False
    retrieved_card_ids: list[str] = []
    try:
        summary = _read_json(follow_summary_path)
        best = summary.get("best_candidate") if isinstance(summary.get("best_candidate"), dict) else {}
        payload = best.get("candidate_payload") if isinstance(best.get("candidate_payload"), dict) else {}
        sr = payload.get("strategy_retrieval") if isinstance(payload.get("strategy_retrieval"), dict) else {}
        for card in list(sr.get("strategy_cards") or []):
            if not isinstance(card, dict):
                continue
            if str(card.get("run_id") or "") == seed_run.run_id:
                retrieved_seed = True
                cid = str(card.get("card_id") or "").strip()
                if cid:
                    retrieved_card_ids.append(cid)
    except Exception:
        retrieved_seed = False

    manifest = {
        "schema_version": "1.0",
        "mode": "closure",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_config": str(base_config_path),
        "pythonpath": pythonpath,
        "paths": {
            "config_dir": str(cfg_dir),
            "log_dir": str(log_dir),
            "global_strategy_kb": str(global_kb_path),
        },
        "seed": {
            "experiment_id": exp_seed,
            "run_id": seed_run.run_id,
            "returncode": seed_run.returncode,
            "flow_log_file": seed_run.log_file,
            "strategy_cards_written": len(seed_cards),
            "card_ids": [str(c.get("card_id") or "") for c in seed_cards if isinstance(c, dict)],
        },
        "follow": {
            "experiment_id": exp_follow,
            "run_id": follow_run.run_id,
            "returncode": follow_run.returncode,
            "flow_log_file": follow_run.log_file,
            "summary_path": str(follow_summary_path),
            "retrieved_seed": bool(retrieved_seed),
            "retrieved_card_ids": retrieved_card_ids,
        },
    }
    _write_json(out_path, manifest)

    md = [
        f"# Strategy Factory Closure Summary ({ts})",
        "",
        f"- Base config: `{base_config_path}`",
        f"- Global KB: `{global_kb_path}`",
        f"- Seed run: `{seed_run.run_id}` cards_written=`{len(seed_cards)}` rc=`{seed_run.returncode}`",
        f"- Follow run: `{follow_run.run_id}` retrieved_seed=`{retrieved_seed}` rc=`{follow_run.returncode}`",
        "",
    ]
    _write_text(out_path.with_suffix(".md"), "\n".join(md))

    ok = (
        seed_run.returncode == 0
        and follow_run.returncode == 0
        and len(seed_cards) > 0
        and bool(retrieved_seed)
    )
    return 0 if ok else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Strategy Factory acceptance runner (soak + matrix)")
    parser.add_argument(
        "--base-config",
        "--config",
        dest="base_config",
        default="configs/agent_flow_strategy_factory_acceptance_ml.json",
        help="Base AgentFlow config JSON used to generate soak/matrix runs.",
    )
    parser.add_argument("--mode", choices=["soak", "matrix", "closure"], default="soak")
    parser.add_argument("--runs", type=int, default=5, help="Soak runs (mode=soak)")
    parser.add_argument("--pythonpath", default="src:.", help="Set PYTHONPATH for the parent flow (stress test). Use empty to unset.")
    parser.add_argument(
        "--steps",
        nargs="*",
        default=["micro_feature", "factor_compile", "factor_eval", "strategy_miner", "report"],
    )
    parser.add_argument("--out", default=None, help="Manifest path (json). Default under runtime_manifests/")
    args = parser.parse_args(argv)

    base_config_path = (REPO_ROOT / str(args.base_config)).resolve()
    if not base_config_path.exists():
        print(f"[acceptance] missing base config: {base_config_path}", file=sys.stderr)
        return 2

    pythonpath = str(args.pythonpath).strip()
    if not pythonpath:
        pythonpath_val = None
    else:
        pythonpath_val = pythonpath

    ts = _now_ts()
    out_path = Path(str(args.out)).resolve() if args.out else (REPO_ROOT / "runtime_manifests" / f"strategy_factory_{args.mode}_{ts}.json")
    steps = [str(s) for s in (args.steps or []) if str(s).strip()]

    if args.mode == "soak":
        return _soak_mode(
            base_config_path=base_config_path,
            runs=int(args.runs),
            pythonpath=pythonpath_val,
            out_path=out_path,
            steps=steps,
        )
    if args.mode == "matrix":
        return _matrix_mode(base_config_path=base_config_path, pythonpath=pythonpath_val, out_path=out_path, steps=steps)
    return _closure_mode(base_config_path=base_config_path, pythonpath=pythonpath_val, out_path=out_path, steps=steps)


if __name__ == "__main__":
    raise SystemExit(main())
