from __future__ import annotations

import json
import sys
import math
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import FileResponse

from ..errors import error
from ...runtime import ROOT, SRC

router = APIRouter()
from agent_market import paths  # type: ignore  # noqa: E402


def _ensure_src_on_path() -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def _get_backtest_utils():
    _ensure_src_on_path()
    from agent_market.backtest_results import (  # type: ignore
        build_backtest_summary,
        find_latest_backtest_zip,
        read_backtest_trades,
    )

    return build_backtest_summary, read_backtest_trades, find_latest_backtest_zip


@router.get("/results/list")
def results_list(results_dir: str = "user_data/backtest_results", limit: int = 20):
    try:
        rd = paths.safe_resolve(results_dir, allow_absolute=True)
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))
    if not rd.exists():
        return error("RESULTS_DIR_NOT_FOUND", f"Results dir not found: {rd}")
    items = []
    for p in rd.glob("backtest-result-*.zip"):
        items.append({"name": p.name, "mtime": p.stat().st_mtime, "size": p.stat().st_size})
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"dir": str(rd), "items": items[: max(1, int(limit))]}


@router.get("/results/summary")
def results_summary(name: str, results_dir: str = "user_data/backtest_results"):
    build_backtest_summary, read_backtest_trades, _ = _get_backtest_utils()

    try:
        rd = paths.safe_resolve(results_dir, allow_absolute=True)
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))
    zp = rd / name
    if not zp.exists():
        return error("NOT_FOUND", f"Not found: {zp}")
    summary = build_backtest_summary(zp)
    trades = read_backtest_trades(zp)
    if trades is not None:
        summary["trades"] = trades
    return summary


@router.get("/results/latest-training")
def results_latest_training(models_dir: str = "artifacts/models"):
    try:
        base = paths.safe_resolve(models_dir, allow_absolute=True)
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))
    if not base.exists():
        return error("MODELS_DIR_NOT_FOUND", f"Models dir not found: {base}")
    candidates = list(base.rglob("training_summary.json"))
    if not candidates:
        return {"error": f"No training_summary.json under {base}"}
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        return error("PARSE_ERROR", f"Failed to parse {latest}: {exc}")
    payload["summary_path"] = str(latest)
    return payload


@router.get("/results/latest-summary")
def latest_summary(results_dir: str = "user_data/backtest_results"):
    build_backtest_summary, read_backtest_trades, find_latest_backtest_zip = (
        _get_backtest_utils()
    )

    try:
        rd = paths.safe_resolve(results_dir, allow_absolute=True)
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))
    try:
        zip_path = find_latest_backtest_zip(rd)
    except FileNotFoundError:
        zip_path = None
    if not zip_path:
        return error("NO_ARCHIVES", f"No backtest archives found in {rd}")
    summary = build_backtest_summary(zip_path)
    trades = read_backtest_trades(zip_path)
    if trades is not None:
        summary["trades"] = trades
    return summary


@router.post("/results/prepare-feedback")
def prepare_feedback(
    results_dir: str = "user_data/backtest_results",
    out: str = "user_data/llm_feedback/latest_backtest_summary.json",
):
    build_backtest_summary, _, find_latest_backtest_zip = _get_backtest_utils()

    try:
        rd = paths.safe_resolve(results_dir, allow_absolute=True)
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))
    try:
        zip_path = find_latest_backtest_zip(rd)
    except FileNotFoundError:
        zip_path = None
    if not zip_path:
        return error("NO_ARCHIVES", f"No backtest archives found in {rd}")
    summary = build_backtest_summary(zip_path)
    try:
        out_path = paths.safe_resolve(out, allow_absolute=True)
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"feedback_path": str(out_path)}


@router.get("/results/gallery")
def results_gallery(results_dir: str = "user_data/backtest_results", limit: int = 20):
    try:
        rd = paths.safe_resolve(results_dir, allow_absolute=True)
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))
    if not rd.exists():
        return error("RESULTS_DIR_NOT_FOUND", f"Results dir not found: {rd}")
    zips = sorted(
        rd.glob("backtest-result-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    items = []
    build_backtest_summary, _, _ = _get_backtest_utils()
    for zp in zips[: max(1, int(limit))]:
        try:
            summary = build_backtest_summary(zp)
            summary["name"] = zp.name
            summary["mtime"] = zp.stat().st_mtime
            items.append(summary)
        except Exception as _exc:
            import logging; logging.getLogger(__name__).debug("Skipped: %s", _exc)
            continue
    return {"dir": str(rd), "items": items}


@router.get("/results/bundles/list")
def results_bundles_list(limit: int = 20):
    """List recent bundle.zip under artifacts/runs/*/bundle/bundle.zip."""
    try:
        lim = int(limit)
    except Exception:
        lim = 20
    lim = max(1, min(lim, 200))

    runs_dir = paths.runs_root()
    if not runs_dir.exists():
        return {"items": [], "count": 0, "limit": lim}

    items: list[dict] = []
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        run_id = child.name
        bundle = (child / "bundle" / "bundle.zip").resolve()
        if not bundle.exists():
            continue
        try:
            stat = bundle.stat()
        except Exception as _exc:
            import logging; logging.getLogger(__name__).debug("Skipped: %s", _exc)
            continue
        try:
            path_str = str(bundle.relative_to(ROOT.resolve()))
        except Exception:
            path_str = str(bundle)
        items.append(
            {
                "run_id": run_id,
                "path": path_str,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }
        )
    items.sort(key=lambda x: x.get("mtime") or 0, reverse=True)
    return {"items": items[:lim], "count": len(items), "limit": lim}


@router.get("/results/bundles/download/{run_id}")
def results_bundles_download(run_id: str):
    run_id = str(run_id or "").strip().lower()
    bundle = (paths.run_dir(run_id) / "bundle" / "bundle.zip").resolve()
    if not bundle.exists():
        return error("NOT_FOUND", f"bundle not found: {bundle}")
    return FileResponse(path=str(bundle), filename=f"bundle-{run_id}.zip")


@router.get("/results/aggregate")
def results_aggregate(names: str, results_dir: str = "user_data/backtest_results"):
    try:
        rd = paths.safe_resolve(results_dir, allow_absolute=True)
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))
    if not rd.exists():
        return error("RESULTS_DIR_NOT_FOUND", f"Results dir not found: {rd}")
    build_backtest_summary, _, _ = _get_backtest_utils()
    files = [rd / n for n in names.split(",") if n.strip()]
    metrics = []
    for f in files:
        if not f.exists():
            continue
        try:
            s = build_backtest_summary(f)
            metrics.append(
                {
                    "name": f.name,
                    "profit_total_pct": s.get("profit_total_pct"),
                    "max_drawdown_abs": s.get("max_drawdown_abs"),
                    "trades": s.get("trades"),
                }
            )
        except Exception as _exc:
            import logging; logging.getLogger(__name__).debug("Skipped: %s", _exc)
            continue
    if not metrics:
        return error("NO_VALID_INPUTS", "no valid inputs")
    profs = [float(m.get("profit_total_pct") or 0.0) for m in metrics]
    mean = sum(profs) / max(1, len(profs))
    var = sum((p - mean) ** 2 for p in profs) / max(1, len(profs))
    std = math.sqrt(var)
    score = mean / (std + 1e-9)
    return {"items": metrics, "mean_profit": mean, "std_profit": std, "robust_score": score}
