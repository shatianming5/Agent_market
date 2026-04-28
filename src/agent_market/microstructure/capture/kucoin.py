from __future__ import annotations

"""KuCoin capture helpers (non-WS).

WebSocket capture has been removed; this module only keeps small helpers used by:
- offline fixture replay (`scripts/micro_capture.py --fixture ...`)
- channel/topic parsing
- lightweight level2 sequence-gap tracking

For live capture, use REST polling via `agent_market.microstructure.capture.kucoin_rest`.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_channel_from_topic(topic: str) -> Optional[str]:
    t = str(topic or "")
    if t.startswith("/market/match:"):
        return "match"
    if t.startswith("/market/level2:"):
        return "level2"
    return None


def _symbol_from_topic(topic: str) -> Optional[str]:
    t = str(topic or "")
    if ":" not in t:
        return None
    _, sym = t.split(":", 1)
    sym = sym.strip()
    return sym or None


class KuCoinLevel2SeqGapTracker:
    """Best-effort sequence-gap tracker for KuCoin level2 updates."""

    def __init__(self, *, example_limit: int = 5) -> None:
        self._example_limit = max(0, int(example_limit))
        self._last_seq_end: dict[str, int] = {}
        self._per_symbol_updates: dict[str, int] = {}
        self._per_symbol_gaps: dict[str, int] = {}
        self._gap_count = 0
        self._examples: list[dict] = []
        self._scanned_updates = 0

    def observe(self, *, topic: str, data: Any) -> None:
        if not isinstance(data, dict):
            return
        seq_start = data.get("sequenceStart")
        seq_end = data.get("sequenceEnd")
        if seq_start is None or seq_end is None:
            return
        try:
            seq_start_i = int(seq_start)
            seq_end_i = int(seq_end)
        except Exception:
            return

        sym = str(data.get("symbol") or "").strip() or (_symbol_from_topic(topic) or "")
        if not sym:
            return

        self._scanned_updates += 1
        self._per_symbol_updates[sym] = int(self._per_symbol_updates.get(sym) or 0) + 1

        last_end = self._last_seq_end.get(sym)
        if last_end is not None and seq_start_i != int(last_end) + 1:
            self._gap_count += 1
            self._per_symbol_gaps[sym] = int(self._per_symbol_gaps.get(sym) or 0) + 1
            if self._example_limit and len(self._examples) < self._example_limit:
                self._examples.append(
                    {
                        "symbol": sym,
                        "expected_start": int(last_end) + 1,
                        "got_start": int(seq_start_i),
                        "got_end": int(seq_end_i),
                        "observed_at": _utc_now_iso(),
                    }
                )

        self._last_seq_end[sym] = int(seq_end_i)

    def meta(self) -> Dict[str, Any]:
        per_symbol = {}
        for sym in sorted(set(self._per_symbol_updates.keys()) | set(self._per_symbol_gaps.keys())):
            per_symbol[sym] = {
                "updates": int(self._per_symbol_updates.get(sym) or 0),
                "gaps": int(self._per_symbol_gaps.get(sym) or 0),
                "last_seq_end": int(self._last_seq_end.get(sym) or 0),
            }
        return {
            "count": int(self._gap_count),
            "examples": list(self._examples),
            "per_symbol": per_symbol,
            "scanned_updates": int(self._scanned_updates),
        }


__all__ = ["KuCoinLevel2SeqGapTracker", "infer_channel_from_topic"]

