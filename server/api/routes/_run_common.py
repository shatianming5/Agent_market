from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Optional

from agent_market import paths  # type: ignore

from ...runtime import ROOT, SRC


def ensure_src_on_path() -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def resolve_executable(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    try:
        sibling = Path(sys.executable).with_name(name)
        if sibling.exists():
            return str(sibling)
    except Exception:
        pass
    for cand in [
        ROOT / ".venv" / "bin" / name,
        ROOT / ".venv" / "Scripts" / (name + ".exe"),
        ROOT / "venv" / "bin" / name,
        ROOT / "venv" / "Scripts" / (name + ".exe"),
    ]:
        if cand.exists():
            return str(cand)
    return None


def load_latest_flow_run_id() -> Optional[str]:
    meta_path = paths.run_meta_latest_path()
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    run_id = str(payload.get("run_id") or "").strip().lower()
    if run_id and all(ch in "0123456789abcdef" for ch in run_id) and 8 <= len(run_id) <= 64:
        return run_id
    return None


def detect_kucoin_level2_sequence_gaps(
    *,
    level2_path: Path,
    snapshot_sequence: int,
    max_updates: int = 5000,
) -> dict:
    """
    Best-effort KuCoin L2 sequence gap precheck.

    This is intentionally lightweight and only scans sequenceStart/sequenceEnd fields.
    """

    import gzip

    last_seq = int(snapshot_sequence) if snapshot_sequence else None
    gaps: list[dict] = []
    updates = 0
    with gzip.open(Path(level2_path), "rt", encoding="utf-8") as fh:
        for line in fh:
            if updates >= int(max_updates):
                break
            s = line.strip()
            if not s:
                continue
            try:
                msg = json.loads(s)
            except Exception:
                continue
            if not isinstance(msg, dict) or msg.get("type") != "message":
                continue
            data = msg.get("data") if isinstance(msg.get("data"), dict) else {}
            seq_start = data.get("sequenceStart")
            seq_end = data.get("sequenceEnd")
            if seq_start is None or seq_end is None:
                continue
            try:
                seq_start_i = int(seq_start)
                seq_end_i = int(seq_end)
            except Exception:
                continue
            if last_seq is not None and seq_start_i != last_seq + 1:
                gaps.append({"expected_start": last_seq + 1, "got_start": seq_start_i, "got_end": seq_end_i})
                if len(gaps) >= 5:
                    break
            last_seq = seq_end_i
            updates += 1
    return {"count": int(len(gaps)), "examples": gaps, "scanned_updates": int(updates)}
