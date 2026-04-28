from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from filelock import FileLock, Timeout as FileLockTimeout

from agent_market import paths

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    p = Path(path).resolve()
    return _sha256_bytes(p.read_bytes())


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def strategy_registry_dir() -> Path:
    return (paths.user_data_root() / "strategy_registry").resolve()


def strategy_registry_path() -> Path:
    return (strategy_registry_dir() / "registry.json").resolve()


def strategy_versions_dir() -> Path:
    return (strategy_registry_dir() / "versions").resolve()

def _registry_lock_path() -> Path:
    return (strategy_registry_dir() / ".registry.lock").resolve()


def _strategy_key(dest_path: Path) -> str:
    # Use the filename stem as the stable key (Freqtrade strategy class name usually matches).
    return str(dest_path.stem).strip() or str(dest_path.name).strip()


@dataclass
class StrategyApprovalRecord:
    strategy_key: str
    filename: str
    version: int
    created_at: str
    run_id: str
    candidate: str
    source_path: str
    dest_path: str
    sha256: str
    metrics: dict[str, Any]

def validate_strategy_file(path: Path) -> dict[str, Any]:
    """Lightweight validation for approved strategy code.

    We keep this minimal (syntax/compile) to avoid depending on freqtrade
    at approval time.
    """
    p = Path(path).resolve()
    if not p.exists():
        return {"ok": False, "error": f"missing:{p}"}
    try:
        source = p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {"ok": False, "error": f"read_failed:{exc}"}
    try:
        compile(source, str(p), "exec")
    except SyntaxError as exc:
        return {"ok": False, "error": f"syntax_error:{exc.msg}", "line": exc.lineno, "offset": exc.offset}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": f"compile_failed:{exc}"}
    return {"ok": True}


def record_strategy_approval(
    *,
    dest_path: Path,
    source_path: Path,
    run_id: str,
    candidate: str,
    metrics: Optional[dict[str, Any]] = None,
) -> StrategyApprovalRecord:
    """Record an approved strategy into a simple on-disk registry.

    This does NOT change Freqtrade's active strategy location (still under
    user_data/strategies/). It only creates immutable version snapshots under
    user_data/strategy_registry/versions/ and updates registry.json.
    """
    dest_path = Path(dest_path).resolve()
    source_path = Path(source_path).resolve()
    reg_path = strategy_registry_path()
    versions_root = strategy_versions_dir()
    lock = FileLock(str(_registry_lock_path()))
    timeout = float(os.environ.get("AGENT_MARKET_STRATEGY_REGISTRY_LOCK_TIMEOUT_SEC", "15") or "15")

    key = _strategy_key(dest_path)
    try:
        with lock.acquire(timeout=timeout):
            payload = _read_json(reg_path)
            items = list(payload.get("items") or [])
            existing = [it for it in items if isinstance(it, dict) and str(it.get("strategy_key") or "") == key]
            existing_versions = [int(it.get("version") or 0) for it in existing if str(it.get("version") or "").isdigit()]
            next_version = (max(existing_versions) + 1) if existing_versions else 1

            sha = sha256_path(dest_path)
            created_at = _utc_now()
            metrics_payload: dict[str, Any] = dict(metrics or {})

            # Snapshot immutable version.
            version_dir = (versions_root / key).resolve()
            version_dir.mkdir(parents=True, exist_ok=True)
            version_path = (version_dir / f"v{next_version}.py").resolve()
            version_path.write_bytes(dest_path.read_bytes())

            # Mark previous versions inactive, and append new one.
            for it in items:
                if isinstance(it, dict) and str(it.get("strategy_key") or "") == key:
                    it["active"] = False
            record: dict[str, Any] = {
                "strategy_key": key,
                "filename": dest_path.name,
                "version": int(next_version),
                "active": True,
                "created_at": created_at,
                "run_id": str(run_id or ""),
                "candidate": str(candidate or ""),
                "source_path": str(source_path),
                "dest_path": str(dest_path),
                "version_path": str(version_path),
                "sha256": sha,
                "metrics": metrics_payload,
            }
            items.append(record)
            payload_out = {
                "schema_version": 1,
                "updated_at": created_at,
                "items": items[-5000:],  # cap growth
            }
            _write_json_atomic(reg_path, payload_out)
    except FileLockTimeout as exc:
        logger.warning("Strategy registry lock timeout: %s", _registry_lock_path())
        raise RuntimeError("strategy_registry_lock_timeout") from exc

    return StrategyApprovalRecord(
        strategy_key=key,
        filename=dest_path.name,
        version=int(next_version),
        created_at=created_at,
        run_id=str(run_id or ""),
        candidate=str(candidate or ""),
        source_path=str(source_path),
        dest_path=str(dest_path),
        sha256=sha,
        metrics=metrics_payload,
    )


__all__ = [
    "StrategyApprovalRecord",
    "record_strategy_approval",
    "sha256_path",
    "strategy_registry_dir",
    "strategy_registry_path",
    "strategy_versions_dir",
    "validate_strategy_file",
]
