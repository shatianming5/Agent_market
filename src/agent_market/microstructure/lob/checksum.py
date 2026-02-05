from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional


def sha256_file(path: Path) -> str:
    p = Path(path).expanduser().resolve()
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def checksum_capture_session(out_dir: Path) -> Dict[str, Optional[str]]:
    """
    Compute best-effort checksums for a capture session directory.
    """

    d = Path(out_dir).expanduser().resolve()
    out: Dict[str, Optional[str]] = {"manifest_json": None, "match_ndjson_gz": None, "level2_ndjson_gz": None}
    for key, name in [
        ("manifest_json", "manifest.json"),
        ("match_ndjson_gz", "match.ndjson.gz"),
        ("level2_ndjson_gz", "level2.ndjson.gz"),
    ]:
        p = (d / name).resolve()
        if p.exists():
            out[key] = sha256_file(p)
    return out


def checksum_lob_state_parquet(path: Path) -> str:
    """
    Compute a file checksum for `lob_state.parquet`.

    This is intentionally a file-level checksum to keep the contract stable across
    parquet writer versions.
    """

    return sha256_file(path)


__all__ = ["checksum_capture_session", "checksum_lob_state_parquet", "sha256_file"]

