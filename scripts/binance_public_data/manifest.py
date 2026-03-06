from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ManifestEntry:
    url: str
    out_path: str
    dataset: str
    market: str
    freq: str
    symbol: str
    interval: str
    yyyymm: str
    checksum_url: Optional[str] = None


class ManifestDB:
    """SQLite manifest for resumable downloads."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS downloads (
              url TEXT PRIMARY KEY,
              out_path TEXT NOT NULL,
              dataset TEXT,
              market TEXT,
              freq TEXT,
              symbol TEXT,
              interval TEXT,
              yyyymm TEXT,
              checksum_url TEXT,
              status TEXT,
              size_bytes INTEGER,
              sha256 TEXT,
              attempts INTEGER DEFAULT 0,
              last_error TEXT,
              updated_at INTEGER
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def upsert_planned(self, e: ManifestEntry) -> None:
        now = int(time.time())
        self._conn.execute(
            """
            INSERT INTO downloads (url, out_path, dataset, market, freq, symbol, interval, yyyymm, checksum_url, status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'planned', ?)
            ON CONFLICT(url) DO UPDATE SET
              out_path=excluded.out_path,
              dataset=excluded.dataset,
              market=excluded.market,
              freq=excluded.freq,
              symbol=excluded.symbol,
              interval=excluded.interval,
              yyyymm=excluded.yyyymm,
              checksum_url=excluded.checksum_url,
              updated_at=excluded.updated_at;
            """,
            (
                e.url,
                e.out_path,
                e.dataset,
                e.market,
                e.freq,
                e.symbol,
                e.interval,
                e.yyyymm,
                e.checksum_url,
                now,
            ),
        )
        self._conn.commit()

    def mark_ok(self, url: str, *, size_bytes: int, sha256: str) -> None:
        now = int(time.time())
        self._conn.execute(
            """
            UPDATE downloads
            SET status='ok', size_bytes=?, sha256=?, last_error=NULL, updated_at=?
            WHERE url=?;
            """,
            (size_bytes, sha256, now, url),
        )
        self._conn.commit()

    def mark_failed(self, url: str, err: str) -> None:
        now = int(time.time())
        self._conn.execute(
            """
            UPDATE downloads
            SET status='failed', attempts=attempts+1, last_error=?, updated_at=?
            WHERE url=?;
            """,
            (err[:2000], now, url),
        )
        self._conn.commit()

    def is_ok(self, url: str) -> bool:
        cur = self._conn.execute("SELECT status FROM downloads WHERE url=?", (url,))
        row = cur.fetchone()
        return bool(row and row[0] == 'ok')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_checksum_file(text: str) -> Optional[str]:
    # Typical format: "<sha256>  <filename>" (or *filename)
    for line in (text or '').splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            return parts[0].lower()
    return None
