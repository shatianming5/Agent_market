"""Lightweight retrieval-augmented knowledge base for WQ Brain."""
from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_KB_SAVE_LOCK = threading.Lock()

# ── Seed library: known expressions with verified/estimated metrics ───────────
_SEEDS: list[dict] = [
    {"expr": "rank(-ts_mean(returns, 60))", "desc": "60-day contrarian reversal", "sharpe": 1.2, "to": 0.10},
    {"expr": "rank(-ts_mean(returns, 120))", "desc": "120-day contrarian reversal", "sharpe": 1.0, "to": 0.08},
    {"expr": "rank(ts_rank(close, 252))", "desc": "12-month price momentum rank", "sharpe": 1.1, "to": 0.12},
    {"expr": "rank(close / ts_mean(close, 60) - 1)", "desc": "price deviation from 60-day MA", "sharpe": 0.9, "to": 0.13},
    {"expr": "rank(ts_mean(volume, 20) / ts_mean(volume, 60))", "desc": "short-term volume surge", "sharpe": 0.8, "to": 0.15},
    {"expr": "rank(-ts_delta(close,1)/ts_mean(volume,20) - ts_delta(close,5)/ts_mean(volume,60))",
     "desc": "Amihud illiquidity (multi-window)", "sharpe": 1.47, "to": 0.39},
    {"expr": "rank(-ts_delta(close,1)/close*volume/adv20)",
     "desc": "Amihud normalized by adv20", "sharpe": 1.77, "to": 0.71},
    {"expr": "rank(group_zscore(-returns, sector))",
     "desc": "sector-neutral short-term reversal", "sharpe": 1.70, "to": 0.72},
    {"expr": "rank(group_zscore(ts_mean(returns, 20), sector))",
     "desc": "sector-adjusted 20-day momentum", "sharpe": 1.1, "to": 0.18},
    {"expr": "rank(ts_zscore(close, 60))", "desc": "60-day price z-score", "sharpe": 0.9, "to": 0.13},
    {"expr": "rank(-ts_delta(close, 5) / close)", "desc": "5-day price reversal", "sharpe": 1.0, "to": 0.25},
    {"expr": "rank(ts_mean(close / vwap - 1, 20))", "desc": "20-day close-to-VWAP mean", "sharpe": 0.7, "to": 0.25},
    {"expr": "rank(ts_decay_linear(returns, 20))", "desc": "linearly-weighted recent returns", "sharpe": 0.8, "to": 0.20},
    {"expr": "rank(ts_corr(close, volume, 20))", "desc": "price-volume correlation 20d", "sharpe": 0.7, "to": 0.20},
    {"expr": "rank(ts_rank(volume, 20) / ts_rank(close, 20))", "desc": "volume rank over price rank", "sharpe": 0.6, "to": 0.62},
    {"expr": "rank(-ts_delta(close, 1) / close)", "desc": "overnight reversal signal", "sharpe": 0.8, "to": 0.40},
    {"expr": "rank(ts_mean(returns, 60) - group_zscore(returns, sector))",
     "desc": "idiosyncratic momentum vs sector", "sharpe": 1.0, "to": 0.15},
    # Smoothed Amihud variants — promising for lower turnover
    {"expr": "rank(-ts_mean(returns * volume / adv20, 5))",
     "desc": "5-day smoothed Amihud — lower turnover than daily", "sharpe": 1.5, "to": 0.35},
    {"expr": "rank(-ts_mean(returns * volume / adv20, 10))",
     "desc": "10-day smoothed Amihud — reduced noise", "sharpe": 1.4, "to": 0.25},
    {"expr": "rank(-ts_decay_linear(returns * volume / adv20, 10))",
     "desc": "decay-weighted Amihud — recent days weighted more", "sharpe": 1.5, "to": 0.30},
    {"expr": "rank(group_zscore(-ts_mean(returns * volume / adv20, 5), sector))",
     "desc": "sector-adjusted 5-day Amihud — consistent cross-sector", "sharpe": 1.6, "to": 0.33},
    # Long-term momentum/reversal hybrids
    {"expr": "rank(ts_rank(close, 252) - ts_rank(close, 20))",
     "desc": "12m momentum minus 1m reversal (skip recent reversal)", "sharpe": 1.2, "to": 0.08},
    {"expr": "rank(-ts_delta(close, 60) / close)",
     "desc": "60-day price reversal — low turnover medium-term", "sharpe": 1.15, "to": 0.10},
]

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class KBEntry:
    source: str
    text: str
    metadata: dict = field(default_factory=dict)
    _toks: Optional[frozenset] = field(default=None, repr=False, compare=False)

    @property
    def tokens(self) -> frozenset:
        if self._toks is None:
            self._toks = frozenset(_tokenize(self.text))
        return self._toks

    def to_dict(self) -> dict:
        return {"source": self.source, "text": self.text, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, d: dict) -> KBEntry:
        return cls(source=d["source"], text=d["text"], metadata=d.get("metadata", {}))


def _build_idf(entries: list[KBEntry]) -> dict[str, float]:
    N = max(len(entries), 1)
    df: Counter = Counter()
    for e in entries:
        for t in e.tokens:
            df[t] += 1
    return {t: math.log((N + 1) / (cnt + 1)) + 1.0 for t, cnt in df.items()}


def _tfidf(q_tokens: list[str], entry: KBEntry, idf: dict[str, float]) -> float:
    q_tf = Counter(q_tokens)
    e_tf = Counter(_tokenize(entry.text))
    return sum(
        math.sqrt(q_tf[t]) * math.sqrt(e_tf[t]) * (idf.get(t, 1.0) ** 2)
        for t in q_tf if t in e_tf
    )


def _jaccard(a: frozenset, b: frozenset) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


class KnowledgeBase:
    def __init__(self, index_path: Optional[Path] = None) -> None:
        self._entries: list[KBEntry] = []
        self._idf: dict[str, float] = {}
        self._path = index_path
        self._dirty = False
        if index_path and index_path.exists():
            self._load()
        self._seed()

    def _seed(self) -> None:
        existing = {e.text for e in self._entries}
        for item in _SEEDS:
            if item["expr"] not in existing:
                self._entries.append(KBEntry(
                    source="seed",
                    text=item["expr"],
                    metadata={"desc": item["desc"], "sharpe": item["sharpe"], "turnover": item["to"]},
                ))
        self._rebuild_idf()

    def _rebuild_idf(self) -> None:
        self._idf = _build_idf(self._entries)

    def add_alpha(self, expr: str, sharpe: float, fitness: float, turnover: float) -> None:
        if any(e.text == expr for e in self._entries):
            return
        self._entries.append(KBEntry(
            source="pool",
            text=expr,
            metadata={"sharpe": sharpe, "fitness": fitness, "turnover": turnover},
        ))
        self._rebuild_idf()
        self._dirty = True

    def search(self, query: str, top_k: int = 5) -> list[KBEntry]:
        if not self._entries:
            return []
        q_tokens = _tokenize(query)
        q_set = frozenset(q_tokens)
        scored = [
            (e, 0.4 * _jaccard(q_set, e.tokens) + 0.6 * _tfidf(q_tokens, e, self._idf))
            for e in self._entries
        ]
        scored.sort(key=lambda x: -x[1])
        return [e for e, _ in scored[:top_k]]

    def save(self) -> None:
        if not (self._path and self._dirty):
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with _KB_SAVE_LOCK:
            # Merge-on-save: reload disk contents first to avoid overwriting concurrent writes
            existing_texts: set[str] = set()
            merged: list[dict] = []
            if self._path.exists():
                try:
                    disk = json.loads(self._path.read_text(encoding="utf-8"))
                    for d in disk:
                        if d.get("text") not in existing_texts:
                            merged.append(d)
                            existing_texts.add(d["text"])
                except Exception:
                    pass
            for e in self._entries:
                if e.text not in existing_texts:
                    merged.append(e.to_dict())
                    existing_texts.add(e.text)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
            tmp.replace(self._path)
            self._dirty = False

    def _load(self) -> None:
        data = json.loads(self._path.read_text(encoding="utf-8"))
        self._entries = [KBEntry.from_dict(d) for d in data]
        self._rebuild_idf()

    def __len__(self) -> int:
        return len(self._entries)
