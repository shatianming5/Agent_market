"""Import legacy JSON factor libraries into the Factor Hub registry.

Supported shapes (auto-detected):
  1. {"expressions": [{"name": ..., "expression": ..., ...}, ...]}  (this repo's freqai_expressions*.json)
  2. {"factors": [{"name": ..., "expression": ..., ...}, ...]}
  3. {"factor_expressions": {"feat_x": "z(ema_spread)"}, ...}
  4. [{"name": ..., "expression": ...}, ...]
  5. A flat dict of name -> expression
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

from . import db

if TYPE_CHECKING:
    from .client import Client


# ============================================================
# JSON shape detection
# ============================================================

def _iter_records(data: Any) -> Iterable[Dict[str, Any]]:
    """Yield dicts of the form {name, expression, ...} from arbitrary JSON shapes."""
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "expression" in item:
                yield item
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                yield {"name": str(item[0]), "expression": str(item[1])}
        return

    if not isinstance(data, dict):
        return

    # Shape 1: {"expressions": [...]}  (this repo's freqai_expressions*.json)
    if isinstance(data.get("expressions"), list):
        for item in data["expressions"]:
            if isinstance(item, dict) and "expression" in item:
                yield item
        return

    # Shape 2: {"factors": [...]}
    if isinstance(data.get("factors"), list):
        for item in data["factors"]:
            if isinstance(item, dict) and "expression" in item:
                yield item
        return

    # Shape 2: {"factor_expressions": {...}}
    exprs = data.get("factor_expressions")
    if isinstance(exprs, dict):
        meta = data.get("metadata") or {}
        for name, expr in exprs.items():
            rec = {"name": name, "expression": expr}
            info = meta.get(name)
            if isinstance(info, dict):
                rec.update(info)
            yield rec
        return

    # Shape 3: expressions + per-name metadata at top-level
    if all(isinstance(v, str) for v in data.values()) and data:
        for name, expr in data.items():
            yield {"name": name, "expression": expr}
        return

    # Shape 4: top-level dict whose values are per-factor metadata dicts
    for name, info in data.items():
        if isinstance(info, dict) and "expression" in info:
            rec = {"name": name, **info}
            yield rec


# ============================================================
# Main migration functions
# ============================================================

def migrate_file(client: "Client", path: Path, *, lib_name: str = "",
                 origin: Optional[str] = None, status: str = "active") -> Dict[str, int]:
    """Import a single JSON factor library file. Returns counts."""
    if not path.exists():
        raise FileNotFoundError(f"Factor library not found: {path}")
    lib = lib_name or path.stem
    default_origin = origin or f"json:{lib}"

    with path.open() as f:
        data = json.load(f)

    new = 0
    dedup = 0
    skipped = 0
    errors: List[str] = []
    factor_ids: List[int] = []

    for idx, rec in enumerate(_iter_records(data), start=1):
        expr = str(rec.get("expression", "")).strip()
        if not expr:
            skipped += 1
            continue
        h = db._hash_expr(expr)
        name = str(rec.get("name") or rec.get("id") or f"{lib}_{idx}")
        category = str(rec.get("category") or rec.get("type") or "misc")
        fac_origin = str(rec.get("origin") or default_origin)
        desc = str(rec.get("description") or rec.get("notes") or "")
        meta = {k: v for k, v in rec.items()
                if k not in {"name", "expression", "category", "origin", "description", "notes", "id"}}
        try:
            with client.connect() as conn:
                row = conn.execute("SELECT id FROM factors WHERE expression_hash = ?", (h,)).fetchone()
                existed = row is not None
            fid = client.propose(
                expression=expr, name=name, category=category, origin=fac_origin,
                description=desc, status=status, source_lib=lib,
                metadata=meta if meta else None,
            )
            factor_ids.append(fid)
            if existed:
                dedup += 1
                continue

            new += 1
            # Import embedded metrics as evaluations when present (only for brand-new factors
            # to avoid double-counting on repeated migrations).
            for metric_key in ("oos_ic", "ic", "ic_mean", "lgb_gain", "profit_pct", "sharpe"):
                if metric_key in rec:
                    try:
                        value = float(rec[metric_key])
                    except (TypeError, ValueError):
                        continue
                    client.add_evaluation(
                        fid, eval_type=("ic" if "ic" in metric_key else metric_key),
                        metric_name=metric_key, metric_value=value,
                        notes=f"imported from {path.name}",
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            skipped += 1

    client.log("factor.created", payload={
        "action": "migrate_file", "library": lib, "file": str(path),
        "new": new, "dedup": dedup, "skipped": skipped,
    })

    return {"inserted": new, "dedup": dedup, "skipped": skipped, "errors": len(errors),
            "factor_ids": factor_ids, "library": lib, "error_list": errors}


SKIP_DIR_TOKENS = ("_pytest", ".pytest_cache", "__pycache__", "node_modules",
                   ".git", "backtest_results", "logs", "_journal")


def discover_libraries(search_dirs: Iterable[Path]) -> List[Path]:
    """Find all plausible JSON factor libraries under the given directories."""
    out: List[Path] = []
    seen = set()
    for d in search_dirs:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.json")):
            if p in seen:
                continue
            if any(tok in p.parts for tok in SKIP_DIR_TOKENS):
                continue
            name = p.name.lower()
            if any(tok in name for tok in
                   ("freqai_expression", "factor_lib", "factor_expressions",
                    "expressions_snapshot")):
                out.append(p)
                seen.add(p)
    return out


def migrate_all(client: "Client", search_dirs: Optional[Iterable[Path]] = None,
                *, status: str = "active") -> Dict[str, Any]:
    """Bulk-import every JSON library found under search_dirs."""
    if search_dirs is None:
        from pathlib import Path as _P
        root = _P(__file__).resolve().parents[3]
        search_dirs = [root / "user_data", root / "artifacts" / "models"]

    files = discover_libraries(search_dirs)
    summary = {"files": len(files), "inserted": 0, "dedup": 0, "skipped": 0,
               "errors": 0, "per_file": []}
    for f in files:
        r = migrate_file(client, f, lib_name=f.stem, status=status)
        summary["inserted"] += r["inserted"]
        summary["dedup"] += r["dedup"]
        summary["skipped"] += r["skipped"]
        summary["errors"] += r["errors"]
        summary["per_file"].append(
            {"file": str(f), **{k: r[k] for k in ("inserted", "dedup", "skipped", "errors")}}
        )
    return summary


# ============================================================
# CLI entry (allow `python -m agent_market.factor_hub.migrate`)
# ============================================================

def main() -> None:
    import argparse
    from .client import Client

    p = argparse.ArgumentParser(prog="factor_hub.migrate",
                                description="Import JSON factor libraries into Factor Hub")
    p.add_argument("paths", nargs="*",
                   help="explicit files or dirs to import (default: user_data/ + artifacts/models/)")
    p.add_argument("--db", default=None, help="override FACTOR_HUB_DB path")
    p.add_argument("--status", default="active")
    args = p.parse_args()

    client = Client(db_path=args.db)
    client.init_db()

    if not args.paths:
        summary = migrate_all(client, status=args.status)
    else:
        summary = {"files": 0, "inserted": 0, "dedup": 0, "skipped": 0,
                   "errors": 0, "per_file": []}
        for raw in args.paths:
            path = Path(raw)
            if path.is_dir():
                files = discover_libraries([path])
            elif path.is_file():
                files = [path]
            else:
                print(f"[migrate] skip (not found): {path}")
                continue
            for f in files:
                r = migrate_file(client, f, status=args.status)
                summary["files"] += 1
                summary["inserted"] += r["inserted"]
                summary["dedup"] += r["dedup"]
                summary["skipped"] += r["skipped"]
                summary["errors"] += r["errors"]
                summary["per_file"].append(
                    {"file": str(f), **{k: r[k] for k in ("inserted", "dedup", "skipped", "errors")}}
                )

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
