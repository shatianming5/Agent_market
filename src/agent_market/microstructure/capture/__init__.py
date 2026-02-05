"""Market data capture (WebSocket).

Phase 1: KuCoin spot (public) capture for:
- trades (`match`)
- level2 orderbook deltas (`level2`)

Capture output is ndjson.gz per channel + a manifest.json for reproducibility.
"""

