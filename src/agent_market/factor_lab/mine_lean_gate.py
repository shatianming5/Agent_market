"""Post-mining LEAN validation gate for factor candidate pools."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from agent_market import paths as repo_paths

from . import lean_bridge, rank_portfolio
from .timeframes import normalize_timeframe


STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
DEFAULT_REQUIRED_STATUS = "ok"
DEFAULT_MIN_FINAL_EQUITY = 1.0
DEFAULT_MAX_DRAWDOWN_PCT = 25.0
DEFAULT_MIN_TRADES = 80


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"object is not JSON serializable: {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload


def _repo_meta(path: Path | str) -> str:
    return repo_paths.relpath_for_meta(Path(path))


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_name(value: str) -> str:
    out = []
    for ch in str(value or "").strip():
        out.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    name = "".join(out).strip("._")
    return name or "run"


def make_run_id(tag: str) -> str:
    return f"{_safe_name(tag)}_{_utc_now()}"


def _candidate_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        [{"expression": str(row.get("expression") or ""), "origin": str(row.get("origin") or "")} for row in rows],
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _freeze_candidate_state(
    *,
    tag: str,
    candidate_state: Optional[str | Path],
    output_dir: Path,
) -> dict[str, Any]:
    candidates, source = rank_portfolio.load_candidates(tag, candidate_state=candidate_state)
    rows = [asdict(candidate) for candidate in candidates]
    payload: dict[str, Any] = {
        "version": "factor-mine-lean-gate-candidate-state-v1",
        "tag": tag,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(rows),
        "candidate_fingerprint": _candidate_fingerprint(rows),
        "survivors": rows,
    }
    path = output_dir / "candidate_state.json"
    _write_json(path, payload)
    return {"path": path, "payload": payload}


def _required_statuses(raw: str | None) -> set[str]:
    text = str(raw or DEFAULT_REQUIRED_STATUS).strip().lower()
    if text in {"*", "any", "all"}:
        return {"ok", "partial", "drift"}
    statuses = {part.strip().lower() for part in text.split(",") if part.strip()}
    return statuses or {DEFAULT_REQUIRED_STATUS}


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _expected_ending_open_positions(lean_project: Path) -> dict[str, Any]:
    signals_path = lean_project / "data" / "signals.csv"
    if not signals_path.exists():
        return {
            "expected": None,
            "reason": f"signals.csv missing: {_repo_meta(signals_path)}",
        }
    with signals_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {"expected": 0, "latest_time": None, "nonzero_symbols": [], "terminal_time": None}
    times = sorted({str(row.get("time") or "") for row in rows if str(row.get("time") or "")})
    if len(times) < 2:
        return {"expected": 0, "latest_time": None, "nonzero_symbols": [], "terminal_time": times[-1] if times else None}
    terminal_time = times[-1]
    latest_time = times[-2]
    latest_rows = [row for row in rows if str(row.get("time") or "") == latest_time]
    symbols = [
        str(row.get("symbol") or row.get("pair") or "")
        for row in latest_rows
        if abs(_optional_float(row.get("lean_target_weight")) or 0.0) > 1e-12
    ]
    return {
        "expected": len(symbols),
        "latest_time": latest_time,
        "terminal_time": terminal_time,
        "nonzero_symbols": sorted(symbols),
    }


def assess_comparison(
    comparison: Mapping[str, Any],
    *,
    required_status: str = DEFAULT_REQUIRED_STATUS,
    min_final_equity: float = DEFAULT_MIN_FINAL_EQUITY,
    max_drawdown_pct: float = DEFAULT_MAX_DRAWDOWN_PCT,
    min_trades: int = DEFAULT_MIN_TRADES,
    expected_positions: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate LEAN comparison and absolute performance checks."""
    violations: list[str] = []
    checks: dict[str, Any] = {}
    allowed = _required_statuses(required_status)
    comparison_status = str(comparison.get("status") or "").strip().lower()
    if not comparison_status:
        violations.append("LEAN comparison status missing")
    elif comparison_status not in allowed:
        violations.append(f"LEAN comparison status={comparison_status!r} not in required statuses {sorted(allowed)}")

    metrics = comparison.get("metrics") if isinstance(comparison.get("metrics"), Mapping) else {}
    for field in ("final_equity", "max_drawdown", "trades", "orders", "turnover"):
        item = metrics.get(field) if isinstance(metrics.get(field), Mapping) else {}
        status = str(item.get("status") or "").strip().lower()
        checks[f"{field}_comparison"] = dict(item) if isinstance(item, Mapping) else {}
        if not item or status == "missing":
            violations.append(f"LEAN comparison metric missing: {field}")
        elif status == "drift" and "drift" not in allowed:
            violations.append(f"LEAN comparison metric drift: {field}")

    lean = comparison.get("lean") if isinstance(comparison.get("lean"), Mapping) else {}
    research = comparison.get("research") if isinstance(comparison.get("research"), Mapping) else {}
    final_equity = _optional_float(lean.get("final_equity"))
    max_drawdown = _optional_float(lean.get("max_drawdown"))
    trades = _optional_float(lean.get("trades"))
    orders = _optional_float(lean.get("orders"))
    turnover = _optional_float(lean.get("turnover"))
    max_gross = _optional_float(lean.get("max_gross"))
    fee_cost = _optional_float(lean.get("fee_cost"))
    research_max_gross = _optional_float(research.get("max_gross"))

    checks["final_equity"] = {"value": final_equity, "min_exclusive": float(min_final_equity)}
    if final_equity is None:
        violations.append("LEAN final_equity missing")
    elif final_equity <= float(min_final_equity):
        violations.append(f"LEAN final_equity={final_equity:.6g} <= {float(min_final_equity):.6g}")

    max_drawdown_limit = float(max_drawdown_pct) / 100.0
    checks["max_drawdown"] = {"value": max_drawdown, "max": max_drawdown_limit}
    if max_drawdown is None:
        violations.append("LEAN max_drawdown missing")
    elif max_drawdown > max_drawdown_limit:
        violations.append(f"LEAN max_drawdown={max_drawdown:.6g} > {max_drawdown_limit:.6g}")

    min_trades_i = int(min_trades)
    checks["trades"] = {"value": trades, "min": min_trades_i}
    if trades is None:
        violations.append("LEAN trades missing")
    elif trades < min_trades_i:
        violations.append(f"LEAN trades={trades:.6g} < {min_trades_i}")

    checks["orders"] = {"value": orders, "min": 0}
    if orders is None:
        violations.append("LEAN orders missing")
    elif orders < 0:
        violations.append(f"LEAN orders={orders:.6g} < 0")

    checks["turnover"] = {"value": turnover, "min": 0.0}
    if turnover is None:
        violations.append("LEAN turnover missing")
    elif turnover < 0.0:
        violations.append(f"LEAN turnover={turnover:.6g} < 0")

    checks["max_gross"] = {"value": max_gross, "research": research_max_gross, "max_rel_drift": 0.10}
    if max_gross is None:
        violations.append("LEAN max_gross missing")
    elif research_max_gross is not None and max_gross > research_max_gross * 1.10 + 1e-12:
        violations.append(
            f"LEAN max_gross={max_gross:.6g} > research max_gross {research_max_gross:.6g} by more than 10%"
        )

    checks["fee_cost"] = {"value": fee_cost, "min": 0.0}
    if fee_cost is None:
        violations.append("LEAN fee_cost missing")
    elif fee_cost < 0.0:
        violations.append(f"LEAN fee_cost={fee_cost:.6g} < 0")

    if expected_positions is not None:
        expected_open = expected_positions.get("expected")
        actual_open = _optional_float(lean.get("ending_open_positions"))
        checks["ending_open_positions"] = {
            "value": actual_open,
            "expected": expected_open,
            "latest_time": expected_positions.get("latest_time"),
            "terminal_time": expected_positions.get("terminal_time"),
            "nonzero_symbols": expected_positions.get("nonzero_symbols") or [],
        }
        if expected_open is None:
            violations.append(str(expected_positions.get("reason") or "expected ending open positions unavailable"))
        elif actual_open is None:
            violations.append("LEAN ending_open_positions missing")
        elif actual_open > float(expected_open) + 1e-9:
            violations.append(f"LEAN ending_open_positions={actual_open:.6g} > expected {expected_open}")

    return {
        "status": STATUS_FAILED if violations else STATUS_PASSED,
        "comparison_status": comparison_status,
        "required_statuses": sorted(allowed),
        "violations": violations,
        "checks": checks,
    }


def _capture_version(binary: str) -> dict[str, Any]:
    exe = shutil.which(binary)
    if not exe:
        candidate = Path(binary).expanduser()
        if candidate.exists():
            exe = str(candidate.resolve())
    if not exe:
        return {"ok": False, "error": f"LEAN binary not found: {binary}"}
    try:
        proc = subprocess.run(
            [exe, "--version"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "command": [exe, "--version"], "error": str(exc)}
    return {
        "ok": proc.returncode == 0,
        "command": [exe, "--version"],
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip()[-1000:],
        "stderr": proc.stderr.strip()[-1000:],
    }


def _lean_project_dir(*, tag: str, run_id: str, output_dir: Path) -> Path:
    """Place CLI projects inside the configured LEAN workspace when one exists."""
    lean_config = repo_paths.artifacts_root() / "lean" / "lean.json"
    if lean_config.exists():
        return (
            repo_paths.artifacts_root()
            / "lean"
            / "bridge_projects"
            / _safe_name(tag)
            / _safe_name(run_id)
            / "lean_project"
        ).resolve()
    return (output_dir / "lean_project").resolve()


def _result_path_from_backtest_run(backtest_run: Mapping[str, Any], project: Path) -> Optional[Path]:
    raw = backtest_run.get("result_path")
    if raw:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = (project / path).resolve()
        return path
    return lean_bridge.find_latest_lean_result(project)


def run_mine_lean_gate(
    *,
    tag: str,
    n: int = 30,
    candidate_state: Optional[str | Path] = None,
    run_id: Optional[str] = None,
    output: Optional[str | Path] = None,
    rank_tag: Optional[str] = None,
    venue: str = "okx",
    timeframe: str = "1h",
    data_venue: str = "auto",
    pairs: Optional[Sequence[str] | str] = None,
    start: str = "2025-12-01",
    end: str = "2026-04-12",
    lean_bin: str = "lean",
    lean_timeout: Optional[int] = None,
    lean_data_root: Optional[str | Path] = None,
    lean_required_status: str = DEFAULT_REQUIRED_STATUS,
    min_final_equity: float = DEFAULT_MIN_FINAL_EQUITY,
    max_drawdown_pct: float = DEFAULT_MAX_DRAWDOWN_PCT,
    min_trades: int = DEFAULT_MIN_TRADES,
    force: bool = False,
    rank_kwargs: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Run rank-backtest then LEAN validation for a mined factor candidate pool."""
    started = time.time()
    tf = normalize_timeframe(timeframe)
    rid = _safe_name(run_id or make_run_id(tag))
    out_dir = Path(output).expanduser().resolve() if output else (
        repo_paths.artifacts_root() / "factor_lab" / tag / "lean_gate" / rid
    ).resolve()
    summary_path = out_dir / "mine_lean_gate.json"
    if summary_path.exists() and not force:
        return _read_json(summary_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    rank_run_tag = _safe_name(rank_tag or f"{tag}_lean_gate_{rid}")
    project = _lean_project_dir(tag=tag, run_id=rid, output_dir=out_dir)
    comparison_path = out_dir / "comparison.json"
    lean_version = _capture_version(lean_bin)
    if force and project.exists():
        shutil.rmtree(project)

    base: dict[str, Any] = {
        "version": "factor-mine-lean-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "run_id": rid,
        "rank_tag": rank_run_tag,
        "venue": venue,
        "timeframe": tf,
        "data_venue": data_venue,
        "pairs": pairs,
        "start": start,
        "end": end,
        "n": int(n),
        "lean_bin": lean_bin,
        "lean_timeout": lean_timeout,
        "lean_data_root": str(lean_data_root) if lean_data_root is not None else "",
        "lean_required_status": lean_required_status,
        "gate_thresholds": {
            "min_final_equity": float(min_final_equity),
            "max_drawdown_pct": float(max_drawdown_pct),
            "min_trades": int(min_trades),
        },
        "lean_version": lean_version,
        "artifacts": {
            "dir": _repo_meta(out_dir),
            "summary": _repo_meta(summary_path),
            "lean_project": _repo_meta(project),
            "comparison_json": _repo_meta(comparison_path),
        },
    }

    def fail(reason: str, *, extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        payload = {
            **base,
            "status": STATUS_FAILED,
            "reason": reason,
            "violations": [reason],
            "duration_sec": float(time.time() - started),
        }
        if extra:
            payload.update(dict(extra))
        _write_json(summary_path, payload)
        return payload

    try:
        frozen = _freeze_candidate_state(tag=tag, candidate_state=candidate_state, output_dir=out_dir)
    except Exception as exc:
        return fail(f"candidate state freeze failed: {exc}", extra={"traceback": traceback.format_exc()})

    candidate_state_path = Path(frozen["path"])
    base["candidate_state"] = {
        "source": frozen["payload"].get("source"),
        "path": _repo_meta(candidate_state_path),
        "candidate_count": frozen["payload"].get("candidate_count"),
        "candidate_fingerprint": frozen["payload"].get("candidate_fingerprint"),
    }
    base["artifacts"]["candidate_state"] = _repo_meta(candidate_state_path)

    rank_options = dict(rank_kwargs or {})
    rank_options.update(
        {
            "tag": rank_run_tag,
            "candidate_state": candidate_state_path,
            "n": int(n),
            "venue": venue,
            "timeframe": tf,
            "data_venue": data_venue,
            "start": start,
            "end": end,
        }
    )
    if pairs not in (None, ""):
        rank_options["pairs"] = pairs

    try:
        rank_result = rank_portfolio.rank_backtest(**rank_options)
    except Exception as exc:
        return fail(
            f"rank backtest failed: {exc}",
            extra={
                "rank_options": {k: str(v) if isinstance(v, Path) else v for k, v in rank_options.items()},
                "traceback": traceback.format_exc(),
            },
        )

    rank_artifact = repo_paths.artifacts_root() / "rank_portfolio" / rank_run_tag / "backtest.json"
    if not rank_artifact.exists() and rank_result.get("signals"):
        signal_path = Path(str(rank_result["signals"])).expanduser()
        if not signal_path.is_absolute():
            signal_path = repo_paths.resolve_repo_path(signal_path)
        rank_artifact = signal_path.parent.parent / "backtest.json"
    base["rank_result"] = rank_result
    base["artifacts"]["rank_artifact"] = _repo_meta(rank_artifact)
    if not rank_artifact.exists():
        return fail(f"rank artifact missing after rank backtest: {_repo_meta(rank_artifact)}")

    try:
        export_manifest = lean_bridge.export_project(
            rank_artifact=rank_artifact,
            output=project,
            timeframe=tf,
            data_root=lean_data_root,
        )
    except Exception as exc:
        return fail(f"LEAN export failed: {exc}", extra={"traceback": traceback.format_exc()})

    base["export_manifest"] = export_manifest
    base["artifacts"]["lean_manifest"] = _repo_meta(project / "manifest.json")

    try:
        backtest_run = lean_bridge.run_lean_backtest(
            lean_project=project,
            lean_bin=lean_bin,
            timeout=lean_timeout,
        )
    except Exception as exc:
        return fail(
            f"LEAN backtest failed: {exc}",
            extra={
                "export_manifest": export_manifest,
                "traceback": traceback.format_exc(),
                "artifacts": {
                    **base["artifacts"],
                    "lean_backtest_run": _repo_meta(project / "lean_backtest_run.json"),
                },
            },
        )

    if not bool(backtest_run.get("ok", False)):
        return fail(
            f"LEAN backtest failed: returncode={backtest_run.get('returncode')} ok={backtest_run.get('ok')}",
            extra={
                "export_manifest": export_manifest,
                "backtest_run": backtest_run,
                "artifacts": {
                    **base["artifacts"],
                    "lean_backtest_run": _repo_meta(project / "lean_backtest_run.json"),
                },
            },
        )

    result_path = _result_path_from_backtest_run(backtest_run, project)
    if result_path is None or not result_path.exists():
        return fail(
            "LEAN backtest completed but no result JSON was found",
            extra={
                "export_manifest": export_manifest,
                "backtest_run": backtest_run,
                "artifacts": {
                    **base["artifacts"],
                    "lean_backtest_run": _repo_meta(project / "lean_backtest_run.json"),
                },
            },
        )

    base["artifacts"]["lean_backtest_run"] = _repo_meta(project / "lean_backtest_run.json")
    base["artifacts"]["lean_result"] = _repo_meta(result_path)

    try:
        comparison = lean_bridge.compare_results(
            rank_artifact=rank_artifact,
            lean_result=result_path,
            output=comparison_path,
            timeframe=tf,
        )
    except Exception as exc:
        return fail(
            f"LEAN comparison failed: {exc}",
            extra={
                "export_manifest": export_manifest,
                "backtest_run": backtest_run,
                "lean_result": str(result_path),
                "traceback": traceback.format_exc(),
            },
        )

    expected_positions = _expected_ending_open_positions(project)
    assessment = assess_comparison(
        comparison,
        required_status=lean_required_status,
        min_final_equity=min_final_equity,
        max_drawdown_pct=max_drawdown_pct,
        min_trades=min_trades,
        expected_positions=expected_positions,
    )
    payload = {
        **base,
        "status": assessment["status"],
        "reason": "LEAN post-mining gate passed" if assessment["status"] == STATUS_PASSED else "; ".join(assessment["violations"]),
        "comparison_status": assessment["comparison_status"],
        "required_statuses": assessment["required_statuses"],
        "checks": assessment["checks"],
        "violations": assessment["violations"],
        "comparison": comparison,
        "lean_metrics": comparison.get("lean") if isinstance(comparison.get("lean"), Mapping) else {},
        "research_metrics": comparison.get("research") if isinstance(comparison.get("research"), Mapping) else {},
        "expected_ending_open_positions": expected_positions,
        "backtest_run": backtest_run,
        "duration_sec": float(time.time() - started),
    }
    _write_json(summary_path, payload)
    return payload


__all__ = [
    "DEFAULT_MAX_DRAWDOWN_PCT",
    "DEFAULT_MIN_FINAL_EQUITY",
    "DEFAULT_MIN_TRADES",
    "DEFAULT_REQUIRED_STATUS",
    "STATUS_FAILED",
    "STATUS_PASSED",
    "assess_comparison",
    "make_run_id",
    "run_mine_lean_gate",
]
