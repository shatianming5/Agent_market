"""LOB rebuild utilities (snapshot + deltas).

Phase 1 scope:
- Read KuCoin level2 deltas (from capture output) + a snapshot JSON fixture.
- Rebuild a top-N L2 orderbook and persist a parquet series + a rebuild report.
"""

from .rebuild import rebuild_kucoin_lob  # noqa: F401

