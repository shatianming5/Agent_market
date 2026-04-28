"""Market data capture (REST polling or fixture replay).

Phase 1: KuCoin spot (public) capture for:
- trades (`match`)
- level2 orderbook deltas (`level2`, synthesized from snapshot diffs in REST mode)

Capture output is ndjson.gz per channel + a manifest.json for reproducibility.
"""
