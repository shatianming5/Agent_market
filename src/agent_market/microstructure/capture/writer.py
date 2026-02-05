from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CaptureFile:
    channel: str
    path: Path
    count: int = 0


class CaptureWriter:
    def __init__(self, out_dir: Path, *, channels: list[str]) -> None:
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, CaptureFile] = {}
        self._handles: dict[str, Any] = {}
        for ch in channels:
            name = f"{ch}.ndjson.gz"
            path = (self.out_dir / name).resolve()
            self._files[ch] = CaptureFile(channel=ch, path=path, count=0)
            self._handles[ch] = gzip.open(path, "at", encoding="utf-8")

    @property
    def files(self) -> dict[str, CaptureFile]:
        return dict(self._files)

    def write(self, channel: str, record: Dict[str, Any]) -> None:
        if channel not in self._handles:
            return
        fh = self._handles[channel]
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._files[channel].count += 1

    def close(self) -> None:
        for fh in list(self._handles.values()):
            try:
                fh.close()
            except Exception:
                pass
        self._handles.clear()

    def write_manifest(self, payload: Dict[str, Any], *, path: Optional[Path] = None) -> Path:
        p = Path(path) if path is not None else (self.out_dir / "manifest.json")
        p = p.resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return p


__all__ = ["CaptureWriter", "CaptureFile"]

