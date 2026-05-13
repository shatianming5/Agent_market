from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from fastapi import APIRouter, Body

from agent_market import paths  # type: ignore

from ..errors import error, not_found
from ..models import CaptureReq, LobRebuildReq, MicroFeatureReq
from ..validators import validate_timeframe
from ...runtime import ROOT, SRC, jobs
from ...job_manager import JobQueueFullError
from ._helpers import parse_csv as _parse_csv
from ._run_common import detect_kucoin_level2_sequence_gaps

router = APIRouter()


@router.post("/run/capture")
def run_capture(req: CaptureReq = Body(...)):
    exchange = str(req.exchange or "").strip().lower()
    if exchange != "kucoin":
        return error("UNSUPPORTED_EXCHANGE", f"Unsupported exchange: {exchange!r}")

    channels = [c.lower() for c in _parse_csv(req.channels)]
    allowed_channels = {"match", "level2"}
    unknown = sorted(set(channels) - allowed_channels)
    if not channels:
        return error("INVALID_CHANNELS", "channels must not be empty")
    if unknown:
        return error(
            "INVALID_CHANNELS",
            f"Unsupported channels: {unknown}. Allowed: {sorted(allowed_channels)}",
        )

    duration_sec = int(req.duration_sec or 0)
    if duration_sec <= 0:
        duration_sec = 60
    duration_sec = max(1, min(duration_sec, 24 * 3600))

    fixture_path = None
    if req.fixture:
        try:
            fixture_path = paths.safe_resolve(req.fixture, allow_absolute=True)
        except ValueError as exc:
            return error("INVALID_PATH", str(exc))
    if fixture_path is not None and not fixture_path.exists():
        return error("FIXTURE_NOT_FOUND", f"Fixture not found: {fixture_path}")

    out_dir = None
    if req.out_dir:
        try:
            out_dir = paths.safe_resolve(req.out_dir, allow_absolute=True)
        except ValueError as exc:
            return error("INVALID_PATH", str(exc))

    script = ROOT / "scripts" / "micro_capture.py"
    if not script.exists():
        return error("SCRIPT_NOT_FOUND", f"micro_capture script not found: {script}")

    cmd: list[str] = [
        sys.executable,
        str(script),
        "--exchange",
        exchange,
        "--channels",
        ",".join(channels),
    ]
    if out_dir is not None:
        cmd += ["--out-dir", str(out_dir)]
    if req.max_reconnects is not None:
        cmd += ["--max-reconnects", str(int(req.max_reconnects))]
    if fixture_path is not None:
        cmd += ["--fixture", str(fixture_path)]
        timeout_sec = 120
    else:
        symbols = _parse_csv(req.symbols)
        if not symbols:
            return error("INVALID_SYMBOLS", "symbols must not be empty")
        cmd += ["--symbols", " ".join(symbols), "--duration-sec", str(duration_sec)]
        timeout_sec = int(duration_sec) + 120

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    meta = {"exchange": exchange, "channels": channels}
    if req.max_reconnects is not None:
        meta["max_reconnects"] = int(req.max_reconnects)
    try:
        job_id = jobs.start(
            cmd,
            cwd=ROOT,
            env=env,
            timeout_sec=timeout_sec,
            kind="capture",
            meta=meta,
        )
    except JobQueueFullError as exc:
        return error("JOB_QUEUE_FULL", str(exc), status_code=429)
    return {"status": "started", "job_id": job_id, "kind": "capture", "cmd": cmd}


@router.post("/run/lob_rebuild")
def run_lob_rebuild(req: LobRebuildReq = Body(...)):
    exchange = str(req.exchange or "").strip().lower()
    if not exchange:
        exchange = "kucoin"
    if exchange != "kucoin":
        return error("UNSUPPORTED_EXCHANGE", f"Unsupported exchange: {exchange!r}")

    try:
        capture_dir = paths.safe_resolve(req.capture_dir, allow_absolute=True)
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))
    if not capture_dir.exists():
        return error(
            "DATA_NOT_FOUND",
            f"Capture dir not found: {capture_dir}",
            legacy_code="CAPTURE_DIR_NOT_FOUND",
            details={"resource": "capture_dir", "path": str(capture_dir)},
        )

    try:
        snapshot_path = paths.safe_resolve(req.snapshot, allow_absolute=True)
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))
    if not snapshot_path.exists():
        return error(
            "DATA_NOT_FOUND",
            f"Snapshot not found: {snapshot_path}",
            legacy_code="SNAPSHOT_NOT_FOUND",
            details={"resource": "snapshot", "path": str(snapshot_path)},
        )

    symbol = str(req.symbol or "").strip()
    if not symbol:
        return error("INVALID_SYMBOL", "symbol must not be empty")

    try:
        depth = int(req.depth or 0)
    except Exception:
        return error("INVALID_DEPTH", f"depth must be an integer, got {req.depth!r}")
    if depth <= 0:
        depth = 20
    depth = max(1, min(depth, 200))

    try:
        out_dir = (
            paths.safe_resolve(req.out_dir, allow_absolute=True)
            if req.out_dir
            else (paths.artifacts_root() / "lob_rebuild" / uuid.uuid4().hex[:12]).resolve()
        )
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))

    # Precheck: ensure the capture directory contains level2 data and has no sequence gaps.
    level2_path = (capture_dir / "level2.ndjson.gz").resolve()
    if not level2_path.exists():
        return error(
            "DATA_NOT_FOUND",
            f"level2.ndjson.gz not found under capture_dir: {capture_dir}",
            legacy_code="DATA_NOT_FOUND",
            details={"resource": "level2_ndjson_gz", "path": str(level2_path)},
        )
    try:
        snap = json.loads(snapshot_path.read_text(encoding="utf-8-sig"))
        snap_seq = int((snap.get("sequence") if isinstance(snap, dict) else None) or 0)
    except Exception:
        snap_seq = 0
    if snap_seq:
        gaps = detect_kucoin_level2_sequence_gaps(level2_path=level2_path, snapshot_sequence=snap_seq)
        if int(gaps.get("count") or 0) > 0:
            return error(
                "LOB_SEQUENCE_GAP",
                f"KuCoin level2 sequence gap detected (count={gaps.get('count')})",
                legacy_code="LOB_SEQUENCE_GAP",
                details={"level2_path": str(level2_path), "snapshot_sequence": int(snap_seq), **gaps},
            )

    script = ROOT / "scripts" / "lob_rebuild.py"
    if not script.exists():
        return error("SCRIPT_NOT_FOUND", f"lob_rebuild script not found: {script}")

    cmd: list[str] = [
        sys.executable,
        str(script),
        "--capture-dir",
        str(capture_dir),
        "--snapshot",
        str(snapshot_path),
        "--symbol",
        symbol,
        "--depth",
        str(depth),
        "--out-dir",
        str(out_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    try:
        job_id = jobs.start(
            cmd,
            cwd=ROOT,
            env=env,
            timeout_sec=600,
            kind="lob_rebuild",
            meta={"exchange": exchange, "symbol": symbol, "depth": depth},
        )
    except JobQueueFullError as exc:
        return error("JOB_QUEUE_FULL", str(exc), status_code=429)
    return {"status": "started", "job_id": job_id, "kind": "lob_rebuild", "cmd": cmd, "out_dir": str(out_dir)}


@router.post("/run/micro_feature")
def run_micro_feature(req: MicroFeatureReq = Body(...)):
    try:
        cfg_path = paths.safe_resolve(req.config, allow_absolute=True)
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))
    if not cfg_path.exists():
        return not_found("CONFIG_NOT_FOUND", f"Config file not found: {cfg_path}")

    if req.timeframe and not validate_timeframe(str(req.timeframe)):
        return error("INVALID_TIMEFRAME", f"Invalid timeframe: {req.timeframe}")

    run_id = (str(req.run_id).strip().lower() if req.run_id else "") or uuid.uuid4().hex[:12]
    try:
        out_dir = paths.safe_resolve(req.out_dir, allow_absolute=True) if req.out_dir else (paths.run_dir(run_id) / "micro_feature")
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))

    script = ROOT / "scripts" / "micro_features.py"
    if not script.exists():
        return error("SCRIPT_NOT_FOUND", f"micro_features script not found: {script}")

    cmd: list[str] = [
        sys.executable,
        str(script),
        "--config",
        str(cfg_path),
        "--run-id",
        run_id,
        "--out-dir",
        str(out_dir),
    ]
    if req.timeframe:
        cmd += ["--timeframe", str(req.timeframe)]
    if req.timerange:
        cmd += ["--timerange", str(req.timerange)]
    if req.data_dir:
        cmd += ["--data-dir", str(req.data_dir)]
    if req.pairs:
        cmd += ["--pairs"] + [str(p) for p in req.pairs if str(p).strip()]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    try:
        job_id = jobs.start(
            cmd,
            cwd=ROOT,
            env=env,
            timeout_sec=900,
            kind="micro_feature",
            meta={"run_id": run_id},
        )
    except JobQueueFullError as exc:
        return error("JOB_QUEUE_FULL", str(exc), status_code=429)
    return {"status": "started", "job_id": job_id, "kind": "micro_feature", "cmd": cmd, "run_id": run_id}
