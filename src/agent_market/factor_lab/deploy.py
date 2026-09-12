"""Factor library deployment: list / switch / describe."""
from __future__ import annotations

import json
import shutil
from typing import Dict, List

from .paths import USER_DATA, EXPRESSIONS_FILE


def list_factor_libs() -> List[Dict]:
    """List all available factor libraries (freqai_expressions_*.json)."""
    libs = []
    for p in sorted(USER_DATA.glob("freqai_expressions*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8-sig"))
            n = len(d.get("expressions", []) or d.get("candidates", []))
            version = d.get("version", "?")
            generated = d.get("generated_at", d.get("checkpoint_loop", "?"))
            is_current = p.samefile(EXPRESSIONS_FILE) if EXPRESSIONS_FILE.exists() else False
            libs.append({
                "name": p.name, "factors": n, "version": str(version),
                "meta": str(generated)[:30], "is_current": is_current,
            })
        except Exception as e:
            libs.append({"name": p.name, "error": str(e)[:60]})
    return libs


def current_deployment() -> Dict:
    """Describe the currently deployed factor library."""
    if not EXPRESSIONS_FILE.exists():
        return {"deployed": False}
    d = json.loads(EXPRESSIONS_FILE.read_text(encoding="utf-8-sig"))
    return {
        "deployed": True, "path": str(EXPRESSIONS_FILE),
        "version": d.get("version", "?"),
        "n_factors": len(d.get("expressions", [])),
    }


def switch_to(name: str) -> Dict:
    """Swap factor library. `name` can be a filename or a short tag."""
    p = USER_DATA / name
    if not p.exists():
        # Try with freqai_expressions_ prefix
        p = USER_DATA / f"freqai_expressions_{name}.json"
    if not p.exists():
        # Try with freqai_expressions. prefix
        p = USER_DATA / f"freqai_expressions.{name}.bak.json"
    if not p.exists():
        raise FileNotFoundError(f"Library not found: {name}")
    # Backup current
    if EXPRESSIONS_FILE.exists():
        shutil.copy(EXPRESSIONS_FILE, EXPRESSIONS_FILE.with_suffix(".json.prev.bak"))
    shutil.copy(p, EXPRESSIONS_FILE)
    d = json.loads(EXPRESSIONS_FILE.read_text())
    result = {"deployed_from": str(p), "n_factors": len(d.get("expressions", []))}
    try:
        hub = sync_to_hub(activate=True,
                          notes=f"switch_to {p.name}")
        result["hub_deployment_id"] = hub["deployment_id"]
        result["hub_new_factors"] = hub["new_factors"]
    except Exception as exc:  # noqa: BLE001
        result["hub_sync_error"] = str(exc)[:160]
    return result


def sync_to_hub(name: str = None, *, activate: bool = True,
                deployment_name: str = "production", notes: str = "") -> Dict:
    """Register the currently-deployed (or a specific) JSON library into Factor Hub
    and create/activate a matching `deployment` entry.

    Returns {deployment_id, n_factors, new_factors, existing_factors, library_path}.
    """
    from agent_market.factor_hub import Client
    from agent_market.factor_hub import db as _db
    from .paths import EXPRESSIONS_FILE

    p = EXPRESSIONS_FILE if not name else (USER_DATA / name)
    if not p.exists():
        candidate = USER_DATA / f"freqai_expressions_{name}.json"
        if candidate.exists(): p = candidate
    if not p.exists():
        raise FileNotFoundError(f"Library not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8-sig"))
    exprs = data.get("expressions") or []
    if not exprs:
        raise ValueError(f"No expressions in {p}")

    client = Client()
    client.init_db()

    factor_ids: List[int] = []
    new_count = 0
    existing_count = 0
    source_lib = p.stem

    for rec in exprs:
        expression = str(rec.get("expression", "")).strip()
        if not expression: continue
        with client.connect() as conn:
            row = conn.execute("SELECT id FROM factors WHERE expression_hash = ?",
                               (_db._hash_expr(expression),)).fetchone()
        if row:
            fid = row["id"]; existing_count += 1
        else:
            fid = client.propose(
                expression=expression,
                name=rec.get("name") or f"{source_lib}_{len(factor_ids)+1}",
                category=rec.get("category") or "misc",
                origin=rec.get("origin") or f"deployment:{source_lib}",
                description=rec.get("description") or "",
                status="active", source_lib=source_lib,
                metadata={k: v for k, v in rec.items()
                         if k not in {"name", "expression", "category", "origin",
                                      "description"}},
            )
            new_count += 1
            # Capture imported IC if present
            for key in ("oos_ic", "train_ic", "ic", "combined"):
                if key in rec:
                    try: val = float(rec[key])
                    except (TypeError, ValueError): continue
                    client.add_evaluation(
                        fid, eval_type=("ic" if "ic" in key else key),
                        metric_name=key, metric_value=val,
                        notes=f"sync_to_hub from {p.name}",
                    )
        factor_ids.append(fid)

    did = client.deploy(
        deployment_name, factor_ids, activate=activate,
        deployed_by="factor_lab",
        notes=notes or f"sync_to_hub from {p.name} ({len(factor_ids)} factors)",
    )
    client.log("deployment.switched", payload={
        "action": "sync_to_hub", "library": str(p),
        "deployment_id": did, "deployment_name": deployment_name,
        "n_factors": len(factor_ids), "new": new_count, "existing": existing_count,
        "activated": activate,
    })
    return {"deployment_id": did, "n_factors": len(factor_ids),
            "new_factors": new_count, "existing_factors": existing_count,
            "library_path": str(p), "deployment_name": deployment_name,
            "activated": activate}


def describe(name: str = None) -> Dict:
    """Show detailed info about a factor library."""
    p = EXPRESSIONS_FILE if not name else USER_DATA / name
    if not p.exists():
        raise FileNotFoundError(f"{p}")
    d = json.loads(p.read_text(encoding="utf-8-sig"))
    exprs = d.get("expressions", [])
    if not exprs: return {"empty": True}
    import numpy as np
    ics = [abs(e.get("oos_ic", 0)) for e in exprs if "oos_ic" in e]
    origins = {}
    for e in exprs:
        o = e.get("origin", "?").split("_")[0]
        origins[o] = origins.get(o, 0) + 1
    return {
        "path": str(p), "version": d.get("version", "?"),
        "n_factors": len(exprs),
        "ic_min": min(ics) if ics else 0, "ic_max": max(ics) if ics else 0,
        "ic_mean": float(np.mean(ics)) if ics else 0,
        "origins": origins,
        "sample_exprs": [e.get("expression", "")[:90] for e in exprs[:3]],
    }
