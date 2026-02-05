#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from agent_market import paths  # noqa: E402


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _topic_symbol(topic: str) -> Optional[str]:
    t = str(topic or "")
    if ":" not in t:
        return None
    return t.split(":", 1)[1].strip() or None


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _iter_ndjson_gz(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _date_from_iso(value: Optional[str]) -> str:
    if not value:
        return datetime.now(timezone.utc).strftime("%Y%m%d")
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y%m%d")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y%m%d")


def _write_parquet(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        # Write an empty parquet with a minimal schema.
        rows = [{"ts": None, "symbol": None}]
    import pandas as pd  # noqa: PLC0415

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def _convert_match_to_parquet(src_gz: Path, out_parquet: Path, *, max_rows: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total = 0
    for msg in _iter_ndjson_gz(src_gz):
        data = msg.get("data") if isinstance(msg.get("data"), dict) else {}
        symbol = _topic_symbol(str(msg.get("topic") or "")) or data.get("symbol")
        ts_ms = data.get("time")
        rows.append(
            {
                "ts_ms": int(ts_ms) if ts_ms is not None else None,
                "received_at": msg.get("_received_at"),
                "exchange": msg.get("_exchange"),
                "symbol": symbol,
                "side": data.get("side"),
                "price": _as_float(data.get("price")),
                "size": _as_float(data.get("size")),
                "trade_id": data.get("tradeId") or data.get("trade_id"),
                "sequence": data.get("sequence"),
            }
        )
        total += 1
        if max_rows > 0 and total >= int(max_rows):
            break
    _write_parquet(rows, out_parquet)
    return {"rows": int(total), "source": str(src_gz), "output": str(out_parquet)}


def _convert_level2_to_parquet(src_gz: Path, out_parquet: Path, *, max_rows: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total = 0
    for msg in _iter_ndjson_gz(src_gz):
        if msg.get("type") != "message":
            continue
        data = msg.get("data") if isinstance(msg.get("data"), dict) else {}
        symbol = _topic_symbol(str(msg.get("topic") or "")) or data.get("symbol")
        ts_ms = data.get("time")
        seq_start = data.get("sequenceStart")
        seq_end = data.get("sequenceEnd")
        changes = data.get("changes") if isinstance(data.get("changes"), dict) else {}
        for side in ("bids", "asks"):
            for px_sz in (changes.get(side) or []):
                if not isinstance(px_sz, (list, tuple)) or len(px_sz) < 2:
                    continue
                rows.append(
                    {
                        "ts_ms": int(ts_ms) if ts_ms is not None else None,
                        "received_at": msg.get("_received_at"),
                        "exchange": msg.get("_exchange"),
                        "symbol": symbol,
                        "book_side": "bid" if side == "bids" else "ask",
                        "price": _as_float(px_sz[0]),
                        "size": _as_float(px_sz[1]),
                        "seq_start": int(seq_start) if seq_start is not None else None,
                        "seq_end": int(seq_end) if seq_end is not None else None,
                    }
                )
                total += 1
                if max_rows > 0 and total >= int(max_rows):
                    break
            if max_rows > 0 and total >= int(max_rows):
                break
        if max_rows > 0 and total >= int(max_rows):
            break
    _write_parquet(rows, out_parquet)
    return {"rows": int(total), "source": str(src_gz), "output": str(out_parquet)}


def _safe_unlink(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
            return
        if path.is_dir():
            shutil.rmtree(path)
    except Exception:
        pass


def _copy_or_link(src: Path, dst: Path, *, link: bool, overwrite: bool) -> None:
    src = Path(src).resolve()
    dst = Path(dst).resolve()
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {dst}")
        _safe_unlink(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            dst.symlink_to(src, target_is_directory=src.is_dir())
            return
        except Exception:
            # Fall back to copy
            pass
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export a run_id into plan.md (data/ + results/) layout.")
    parser.add_argument("--run-id", required=True, help="AgentFlow run_id")
    parser.add_argument("--out-root", default=".", help="Destination root (default: repo root)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing exported files")
    parser.add_argument("--max-rows", type=int, default=200_000, help="Max rows to convert for raw capture parquet (0=unlimited)")
    parser.add_argument("--copy-model", action="store_true", help="Copy model directory instead of symlink")
    args = parser.parse_args(argv)

    run_id = str(args.run_id).strip()
    if not run_id:
        raise SystemExit("run_id must not be empty")

    out_root = paths.resolve_repo_path(args.out_root)
    data_root = (out_root / "data").resolve()
    results_root = (out_root / "results").resolve()

    run_meta_path = paths.run_meta_path(run_id)
    if not run_meta_path.exists():
        raise SystemExit(f"run_meta not found: {run_meta_path}")
    meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    artifacts = meta.get("artifacts") if isinstance(meta.get("artifacts"), dict) else {}
    requested_steps = meta.get("requested_steps") if isinstance(meta.get("requested_steps"), list) else []
    requested = {str(s).strip().lower() for s in requested_steps if str(s).strip()}

    exported: list[dict[str, Any]] = []

    # Resolve capture + lob meta for path templating.
    capture_manifest_path = paths.resolve_under_repo(artifacts.get("capture_manifest")) if artifacts else None
    rebuild_report_path = paths.resolve_under_repo(artifacts.get("rebuild_report")) if artifacts else None

    exchange = None
    symbol = None
    date = None
    if capture_manifest_path and capture_manifest_path.exists():
        try:
            cap = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
            exchange = cap.get("exchange")
            symbols = cap.get("symbols") if isinstance(cap.get("symbols"), list) else []
            symbol = symbols[0] if symbols else None
            date = _date_from_iso(cap.get("started_at") or cap.get("ended_at"))
        except Exception:
            pass
    if rebuild_report_path and rebuild_report_path.exists():
        try:
            rep = json.loads(rebuild_report_path.read_text(encoding="utf-8"))
            exchange = exchange or rep.get("exchange")
            symbol = symbol or rep.get("symbol")
            date = date or _date_from_iso(rep.get("started_at"))
        except Exception:
            pass

    exchange = str(exchange or "unknown").strip().lower()
    symbol = str(symbol or "unknown").strip()
    date = str(date or datetime.now(timezone.utc).strftime("%Y%m%d")).strip()

    # 1) capture -> data/raw/{ex}/{sym}/{date}/...
    match_path = paths.resolve_under_repo(artifacts.get("capture_match_path")) if artifacts else None
    level2_path = paths.resolve_under_repo(artifacts.get("capture_level2_path")) if artifacts else None
    if match_path and match_path.exists():
        raw_dir = (data_root / "raw" / exchange / symbol / date).resolve()
        trades_parquet = (raw_dir / "trades.parquet").resolve()
        info = _convert_match_to_parquet(match_path, trades_parquet, max_rows=int(args.max_rows))
        exported.append({"kind": "capture_trades", **info})
        if capture_manifest_path and capture_manifest_path.exists():
            dst = (raw_dir / "meta.json").resolve()
            _copy_or_link(capture_manifest_path, dst, link=False, overwrite=bool(args.overwrite))
            exported.append({"kind": "capture_meta", "source": str(capture_manifest_path), "output": str(dst)})
    if level2_path and level2_path.exists():
        raw_dir = (data_root / "raw" / exchange / symbol / date).resolve()
        deltas_parquet = (raw_dir / "lob_deltas.parquet").resolve()
        info = _convert_level2_to_parquet(level2_path, deltas_parquet, max_rows=int(args.max_rows))
        exported.append({"kind": "capture_lob_deltas", **info})

    # 2) lob_rebuild -> data/lob/{ex}/{sym}/{date}/lob_state.parquet
    lob_state_path = paths.resolve_under_repo(artifacts.get("lob_state_parquet")) if artifacts else None
    if lob_state_path and lob_state_path.exists():
        lob_dir = (data_root / "lob" / exchange / symbol / date).resolve()
        dst = (lob_dir / "lob_state.parquet").resolve()
        _copy_or_link(lob_state_path, dst, link=True, overwrite=bool(args.overwrite))
        exported.append({"kind": "lob_state", "source": str(lob_state_path), "output": str(dst)})

    # 3) micro_feature -> data/features/{run_id}/micro_features.parquet
    micro_feat_path = paths.resolve_under_repo(artifacts.get("micro_feature_parquet")) if artifacts else None
    if micro_feat_path and micro_feat_path.exists():
        feat_dir = (data_root / "features" / run_id).resolve()
        dst = (feat_dir / "micro_features.parquet").resolve()
        _copy_or_link(micro_feat_path, dst, link=True, overwrite=bool(args.overwrite))
        exported.append({"kind": "micro_features", "source": str(micro_feat_path), "output": str(dst)})

    # 4) factor_compile -> data/features/{run_id}/factor_{name}.parquet + factor_ast.json
    factor_name = None
    factor_spec_path = paths.resolve_under_repo(artifacts.get("factor_spec_json")) if artifacts else None
    if factor_spec_path and factor_spec_path.exists():
        try:
            spec = json.loads(factor_spec_path.read_text(encoding="utf-8"))
            factor_name = str(spec.get("name") or "").strip() or None
        except Exception:
            factor_name = None

    factor_ast_path = paths.resolve_under_repo(artifacts.get("factor_ast_json")) if artifacts else None
    if factor_ast_path and factor_ast_path.exists():
        feat_dir = (data_root / "features" / run_id).resolve()
        dst = (feat_dir / "factor_ast.json").resolve()
        _copy_or_link(factor_ast_path, dst, link=False, overwrite=bool(args.overwrite))
        exported.append({"kind": "factor_ast", "source": str(factor_ast_path), "output": str(dst)})

    factor_eval_meta_path = paths.resolve_under_repo(artifacts.get("factor_eval_meta")) if artifacts else None
    if factor_eval_meta_path and factor_eval_meta_path.exists():
        factor_eval_dir = factor_eval_meta_path.parent
        values = (factor_eval_dir / "factor_values.parquet").resolve()
        if values.exists():
            feat_dir = (data_root / "features" / run_id).resolve()
            suffix = factor_name or "factor"
            dst = (feat_dir / f"factor_{suffix}.parquet").resolve()
            _copy_or_link(values, dst, link=True, overwrite=bool(args.overwrite))
            exported.append({"kind": "factor_values", "source": str(values), "output": str(dst)})

    # 5) factor_eval -> results/{run_id}/factor_scores.json + pareto.csv
    factor_scores_path = paths.resolve_under_repo(artifacts.get("factor_scores_json")) if artifacts else None
    if factor_scores_path and factor_scores_path.exists():
        out_dir = (results_root / run_id).resolve()
        dst = (out_dir / "factor_scores.json").resolve()
        _copy_or_link(factor_scores_path, dst, link=False, overwrite=bool(args.overwrite))
        exported.append({"kind": "factor_scores", "source": str(factor_scores_path), "output": str(dst)})
    pareto_path = paths.resolve_under_repo(artifacts.get("factor_pareto_csv")) if artifacts else None
    if pareto_path and pareto_path.exists():
        out_dir = (results_root / run_id).resolve()
        dst = (out_dir / "pareto.csv").resolve()
        _copy_or_link(pareto_path, dst, link=False, overwrite=bool(args.overwrite))
        exported.append({"kind": "pareto_csv", "source": str(pareto_path), "output": str(dst)})

    # 6) train -> results/{run_id}/model/
    model_dirs = artifacts.get("model_dirs") if isinstance(artifacts.get("model_dirs"), list) else []
    if model_dirs:
        model_dir = paths.resolve_repo_path(str(model_dirs[0]))
        if model_dir.exists():
            out_dir = (results_root / run_id / "model").resolve()
            _copy_or_link(model_dir, out_dir, link=not bool(args.copy_model), overwrite=bool(args.overwrite))
            exported.append({"kind": "model_dir", "source": str(model_dir), "output": str(out_dir), "mode": "copy" if args.copy_model else "symlink"})

    # 7) backtest -> results/{run_id}/backtest.zip
    results_dir = paths.resolve_repo_path(str(artifacts.get("backtest_results_dir") or "user_data/backtest_results"))
    try:
        from agent_market.backtest_results import find_latest_backtest_zip  # noqa: WPS433
    except Exception:
        find_latest_backtest_zip = None  # type: ignore[assignment]
    if "backtest" in requested and results_dir.exists() and find_latest_backtest_zip is not None:
        zp = find_latest_backtest_zip(results_dir)
        if zp and zp.exists():
            out_dir = (results_root / run_id).resolve()
            dst = (out_dir / "backtest.zip").resolve()
            _copy_or_link(zp, dst, link=False, overwrite=bool(args.overwrite))
            exported.append({"kind": "backtest_zip", "source": str(zp), "output": str(dst)})

    # 8) tca -> results/{run_id}/tca_report.json (+ html)
    tca_report_path = paths.resolve_under_repo(artifacts.get("tca_report")) if artifacts else None
    if tca_report_path and tca_report_path.exists():
        out_dir = (results_root / run_id).resolve()
        dst = (out_dir / "tca_report.json").resolve()
        _copy_or_link(tca_report_path, dst, link=False, overwrite=bool(args.overwrite))
        exported.append({"kind": "tca_report", "source": str(tca_report_path), "output": str(dst)})
    tca_html_path = paths.resolve_under_repo(artifacts.get("tca_html")) if artifacts else None
    if tca_html_path and tca_html_path.exists():
        out_dir = (results_root / run_id).resolve()
        dst = (out_dir / "tca_report.html").resolve()
        _copy_or_link(tca_html_path, dst, link=False, overwrite=bool(args.overwrite))
        exported.append({"kind": "tca_html", "source": str(tca_html_path), "output": str(dst)})

    # 9) report -> results/{run_id}/bundle.zip
    bundle_path = paths.resolve_under_repo(artifacts.get("bundle_zip")) if artifacts else None
    if bundle_path and bundle_path.exists():
        out_dir = (results_root / run_id).resolve()
        dst = (out_dir / "bundle.zip").resolve()
        _copy_or_link(bundle_path, dst, link=False, overwrite=bool(args.overwrite))
        exported.append({"kind": "bundle_zip", "source": str(bundle_path), "output": str(dst)})

    export_manifest = {
        "version": 1,
        "exported_at": _utc_now_iso(),
        "run_id": run_id,
        "out_root": str(out_root),
        "data_root": str(data_root),
        "results_root": str(results_root),
        "exchange": exchange,
        "symbol": symbol,
        "date": date,
        "exported": exported,
    }
    manifest_path = (results_root / run_id / "export_manifest.json").resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(export_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[export_planmd_layout] run_id={run_id}")
    print(f"[export_planmd_layout] wrote: {manifest_path.relative_to(ROOT)}")
    print(f"[export_planmd_layout] exported: {len(exported)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
