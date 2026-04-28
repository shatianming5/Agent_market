#!/usr/bin/env python3
"""Scan Factor Hub and tag every factor with a `snoop_level` based on how it
was mined. Rules:

    - clean:       mined with strict train-only fitness (v6 composite, combo_ga
                   with dedupe, composite_sanity etc.); never saw real OOS.
    - oos_snooped: mined with OOS-IC-as-gate (run500 / run500nov / run500_real
                   / hubtest_real*); 或 imported from historical freqai JSON
                   libraries that were filtered by remine_factors.py using OOS.
    - unknown:     anything else (e.g. imported factor libraries without a
                   known filter history).

The tag is stored in factors.metadata JSON so existing queries aren't broken
and downstream tools (combo_ga pool pull, Hub UI) can filter by it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

from agent_market.factor_hub import Client


OOS_SNOOPED_PREFIXES = (
    "mining:run500",          # legacy IC-gate miner
    "mining:hubtest_real",    # early IC-gate smoke tests
    "json:freqai_expressions",  # hand-curated libs from remine_factors (OOS gate)
)
OOS_SNOOPED_LIB_KEYWORDS = (
    "freqai_expressions_v",     # v2/v3/v4/v5 archived libs all used OOS gate
    "freqai_expressions.g",     # g-factors backups
    "freqai_expressions_run500",
    "freqai_expressions_hubtest",
    "freqai_expressions_combo_ga",  # combo_ga seeded from oos-snooped pool
    "freqai_expressions_scored_all",
)
CLEAN_PREFIXES = (
    "mining:v6",                # composite fitness, train3 holdout only
    "mining:v7",                # v7 family: 1h+funding, 4h — composite TRAIN3/VAL3 only
    "mining:composite_",
    "mining:nov_sanity",
    "mining:run500nov",         # novelty gate but STILL uses OOS IC — see note
)
# NOTE: run500nov was labelled "nov" for the novelty gate; but the IC-gate
# itself still uses DEFAULT_OOS. So run500nov is technically also oos-snooped.
# We keep the explicit rule below to demote it.
DEMOTE_RUN500NOV = True


def _classify(origin: str, source_lib: str, metadata_md: dict | None = None) -> str:
    o = (origin or "").lower()
    s = (source_lib or "").lower()

    # Authoritative tag: eval_mode=composite in metadata never uses REAL_TEST3.
    if metadata_md and metadata_md.get("eval_mode") == "composite":
        return "clean"

    # Catch explicitly-clean composite runs first
    if o.startswith("mining:v6") or s.startswith("mining:v6"):
        return "clean"
    if o.startswith("mining:v7") or s.startswith("mining:v7"):
        return "clean"
    if o.startswith("mining:composite_") or s.startswith("mining:composite_"):
        return "clean"
    if o.startswith("mining:nov_sanity") or s.startswith("mining:nov_sanity"):
        return "clean"

    # run500nov looks clean (uses novelty) but still uses OOS-IC gate:
    if DEMOTE_RUN500NOV and ("run500nov" in o or "run500nov" in s):
        return "oos_snooped"

    # Explicit OOS-snooped prefixes
    for p in OOS_SNOOPED_PREFIXES:
        if o.startswith(p) or s.startswith(p):
            return "oos_snooped"
    for kw in OOS_SNOOPED_LIB_KEYWORDS:
        if kw in s:
            return "oos_snooped"
    if s.startswith("freqai_expressions"):
        return "oos_snooped"
    return "unknown"


def main() -> int:
    c = Client()
    with c.connect() as conn:
        rows = conn.execute(
            "SELECT id, origin, source_lib, metadata FROM factors"
        ).fetchall()

        counts = {"clean": 0, "oos_snooped": 0, "unknown": 0}
        updates = []
        for r in rows:
            try:
                md = json.loads(r["metadata"] or "{}")
            except Exception:
                md = {}
            level = _classify(r["origin"], r["source_lib"], md)
            counts[level] += 1
            if md.get("snoop_level") == level:
                continue
            md["snoop_level"] = level
            updates.append((json.dumps(md, ensure_ascii=False), int(r["id"])))

        for md_json, fid in updates:
            conn.execute(
                "UPDATE factors SET metadata = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
                "WHERE id = ?", (md_json, fid))
        c.log("snoop.tagged", payload={
            "total": len(rows), "updated": len(updates), **counts})

    print(f"scanned {len(rows)} factors, tagged {len(updates)} updates")
    print(f"  clean       : {counts['clean']}")
    print(f"  oos_snooped : {counts['oos_snooped']}")
    print(f"  unknown     : {counts['unknown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
