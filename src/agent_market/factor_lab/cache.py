"""Persistent cache helpers for FactorLab.

The cache is deliberately small and file based. DataFrame payloads are stored as
parquet when possible with a pickle fallback for environments without parquet
engines; numeric series bundles are stored as compressed npz files.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .paths import LAB_STATE


DEFAULT_CACHE_DIR = LAB_STATE / "cache"
CACHE_VERSION = 2


def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_fingerprint(path: Path, *, sample_bytes: int = 65536) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as fh:
        head = fh.read(sample_bytes)
        h.update(head)
        if stat.st_size > sample_bytes:
            try:
                fh.seek(max(0, stat.st_size - sample_bytes))
                h.update(fh.read(sample_bytes))
            except OSError:
                pass
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sample_sha256": h.hexdigest(),
    }


def panel_fingerprint(panel: pd.DataFrame) -> str:
    cached = panel.attrs.get("factor_lab_panel_key") or panel.attrs.get("factor_lab_panel_fingerprint")
    if cached:
        return str(cached)
    # Fallback for synthetic/in-memory tests. build_big attaches a stable key so
    # production mining does not pay this full-panel hash on every expression.
    values = pd.util.hash_pandas_object(panel, index=True).to_numpy(dtype=np.uint64, copy=False)
    h = hashlib.sha256(values.tobytes())
    h.update(json.dumps(list(map(str, panel.columns)), separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


class FactorLabCache:
    def __init__(self, root: Optional[str | Path] = None, *, enabled: bool = True) -> None:
        self.root = Path(root) if root is not None else DEFAULT_CACHE_DIR
        self.enabled = bool(enabled)
        self.stats: Dict[str, int] = {
            "panel_hits": 0,
            "panel_misses": 0,
            "panel_saves": 0,
            "exposure_hits": 0,
            "exposure_misses": 0,
            "exposure_saves": 0,
            "factor_version_hits": 0,
            "factor_version_misses": 0,
            "factor_version_saves": 0,
            "eval_hits": 0,
            "eval_misses": 0,
            "eval_saves": 0,
            "load_errors": 0,
            "save_errors": 0,
        }
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def snapshot(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "root": str(self.root), **self.stats}

    def _kind_dir(self, kind: str) -> Path:
        d = self.root / kind
        if self.enabled:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def _record(self, name: str) -> None:
        self.stats[name] = int(self.stats.get(name, 0)) + 1

    def _read_dataframe(self, base: Path) -> Tuple[pd.DataFrame, str]:
        parquet = base.with_suffix(".parquet")
        if parquet.exists():
            return pd.read_parquet(parquet), "parquet"
        pickle = base.with_suffix(".pkl")
        if pickle.exists():
            return pd.read_pickle(pickle), "pickle"
        raise FileNotFoundError(str(base))

    def _write_dataframe(self, df: pd.DataFrame, base: Path) -> str:
        parquet = base.with_suffix(".parquet")
        try:
            df.to_parquet(parquet, index=False)
            return "parquet"
        except Exception:
            pickle = base.with_suffix(".pkl")
            df.to_pickle(pickle)
            return "pickle"

    def load_panel(self, key: str) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        if not self.enabled:
            return None
        try:
            base = self._kind_dir("panel") / key
            meta_path = base.with_suffix(".json")
            if not meta_path.exists():
                self._record("panel_misses")
                return None
            df, _ = self._read_dataframe(base)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            df.attrs["factor_lab_panel_key"] = key
            df.attrs["factor_lab_cache_dir"] = str(self.root)
            self._record("panel_hits")
            return df, meta
        except Exception:
            self._record("load_errors")
            self._record("panel_misses")
            return None

    def save_panel(self, key: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            base = self._kind_dir("panel") / key
            fmt = self._write_dataframe(df, base)
            payload = {"cache_version": CACHE_VERSION, "key": key, "format": fmt, "saved_at": time.time(), **meta}
            base.with_suffix(".json").write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
            self._record("panel_saves")
        except Exception:
            self._record("save_errors")

    def load_exposure(self, key: str) -> Optional[Tuple[pd.DataFrame, list[str], Dict[str, Any]]]:
        if not self.enabled:
            return None
        try:
            base = self._kind_dir("exposure") / key
            meta_path = base.with_suffix(".json")
            if not meta_path.exists():
                self._record("exposure_misses")
                return None
            df, _ = self._read_dataframe(base)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._record("exposure_hits")
            return df, list(meta.get("exposure_cols") or []), meta
        except Exception:
            self._record("load_errors")
            self._record("exposure_misses")
            return None

    def save_exposure(self, key: str, df: pd.DataFrame, exposure_cols: list[str], meta: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            base = self._kind_dir("exposure") / key
            fmt = self._write_dataframe(df, base)
            payload = {
                "cache_version": CACHE_VERSION,
                "key": key,
                "format": fmt,
                "saved_at": time.time(),
                "exposure_cols": list(exposure_cols),
                **meta,
            }
            base.with_suffix(".json").write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
            self._record("exposure_saves")
        except Exception:
            self._record("save_errors")

    def load_npz_bundle(self, kind: str, key: str) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
        if not self.enabled:
            return None
        hit_name = f"{kind}_hits"
        miss_name = f"{kind}_misses"
        try:
            base = self._kind_dir(kind) / key
            meta_path = base.with_suffix(".json")
            npz_path = base.with_suffix(".npz")
            if not meta_path.exists() or not npz_path.exists():
                self._record(miss_name)
                return None
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            with np.load(npz_path, allow_pickle=False) as data:
                arrays = {name: data[name] for name in data.files}
            self._record(hit_name)
            return arrays, meta
        except Exception:
            self._record("load_errors")
            self._record(miss_name)
            return None

    def save_npz_bundle(self, kind: str, key: str, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        save_name = f"{kind}_saves"
        try:
            base = self._kind_dir(kind) / key
            np.savez_compressed(base.with_suffix(".npz"), **arrays)
            payload = {"cache_version": CACHE_VERSION, "key": key, "saved_at": time.time(), **meta}
            base.with_suffix(".json").write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
            self._record(save_name)
        except Exception:
            self._record("save_errors")

    def load_eval(self, key: str, *, need_oos_series: bool = False) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            base = self._kind_dir("eval") / key
            meta_path = base.with_suffix(".json")
            if not meta_path.exists():
                self._record("eval_misses")
                return None
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            metrics = dict(payload.get("metrics") or {})
            if need_oos_series:
                npz_path = base.with_suffix(".npz")
                if not npz_path.exists():
                    self._record("eval_misses")
                    return None
                with np.load(npz_path, allow_pickle=False) as data:
                    metrics["oos_series"] = data["oos_series"]
            metrics["cache_key"] = key
            metrics["cache_hit"] = True
            self._record("eval_hits")
            return metrics
        except Exception:
            self._record("load_errors")
            self._record("eval_misses")
            return None

    def save_eval(self, key: str, metrics: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            base = self._kind_dir("eval") / key
            clean_metrics = {k: v for k, v in metrics.items() if k != "oos_series"}
            payload = {
                "cache_version": CACHE_VERSION,
                "key": key,
                "saved_at": time.time(),
                "metrics": _json_safe(clean_metrics),
                "has_oos_series": "oos_series" in metrics,
            }
            base.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            if "oos_series" in metrics:
                np.savez_compressed(
                    base.with_suffix(".npz"),
                    oos_series=np.asarray(metrics["oos_series"], dtype=np.float64),
                )
            self._record("eval_saves")
        except Exception:
            self._record("save_errors")


_CACHE_INSTANCES: Dict[Tuple[str, bool], FactorLabCache] = {}


def get_cache(cache_dir: Optional[str | Path] = None, *, no_cache: bool = False) -> FactorLabCache:
    enabled = not bool(no_cache)
    root = str(Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR)
    key = (root, enabled)
    cache = _CACHE_INSTANCES.get(key)
    if cache is None:
        cache = FactorLabCache(root, enabled=enabled)
        _CACHE_INSTANCES[key] = cache
    return cache


def cache_inventory(cache_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    counts: Dict[str, int] = {}
    total_bytes = 0
    for kind in ("panel", "exposure", "factor_version", "eval"):
        d = root / kind
        files = list(d.glob("*")) if d.exists() else []
        counts[kind] = len([p for p in files if p.suffix == ".json"])
        total_bytes += sum(int(p.stat().st_size) for p in files if p.is_file())
    return {"root": str(root), "exists": root.exists(), "counts": counts, "bytes": total_bytes}


def clear_cache(cache_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    before = cache_inventory(root)
    if root.exists():
        shutil.rmtree(root)
    _CACHE_INSTANCES.clear()
    return {"cleared": str(root), "before": before}
