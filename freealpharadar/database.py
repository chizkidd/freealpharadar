"""SQLite-backed cache and persistence layer.

Everything FreeAlphaRadar fetches is cached here so that:

* free-tier rate limits are respected (we only hit the network when a cache
  entry is missing or its TTL has expired);
* the application remains **fully functional offline**, serving the last known
  good data;
* the Streamlit app can be stateless and simply read from this database, while
  :mod:`run_scorer` refreshes it out of band.

The cache stores arbitrary JSON payloads keyed by ``(source, key)`` together
with a fetch timestamp. Watchlist membership and computed scores are stored in
dedicated tables.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from freealpharadar.config import settings
from freealpharadar.utils import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    source      TEXT NOT NULL,
    key         TEXT NOT NULL,
    payload     TEXT NOT NULL,
    fetched_at  REAL NOT NULL,
    PRIMARY KEY (source, key)
);

CREATE TABLE IF NOT EXISTS watchlist (
    ticker      TEXT PRIMARY KEY,
    added_at    REAL NOT NULL,
    note        TEXT
);

CREATE TABLE IF NOT EXISTS scores (
    ticker      TEXT NOT NULL,
    computed_at REAL NOT NULL,
    payload     TEXT NOT NULL,
    PRIMARY KEY (ticker, computed_at)
);

CREATE INDEX IF NOT EXISTS idx_cache_source ON cache(source);
CREATE INDEX IF NOT EXISTS idx_scores_ticker ON scores(ticker);
"""


class Database:
    """Thin, thread-safe wrapper around a SQLite cache database.

    A single connection is shared across threads (``check_same_thread=False``)
    and guarded by a lock, which is more than sufficient for Streamlit's
    workload and avoids the overhead of per-call connections.

    Args:
        db_path: Path to the SQLite file. Defaults to the configured location.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = Path(db_path or settings.db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.debug("Database initialised at %s", self._path)

    # ------------------------------------------------------------------ #
    # Connection plumbing
    # ------------------------------------------------------------------ #
    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        """Yield a cursor under the shared lock, committing on success."""
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            finally:
                cur.close()

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ #
    # Generic JSON cache
    # ------------------------------------------------------------------ #
    def set_cache(self, source: str, key: str, payload: Any) -> None:
        """Insert or replace a cache entry.

        Args:
            source: Logical data source name (e.g. ``"yfinance:prices"``).
            key: Identifier within the source (typically a ticker).
            payload: JSON-serialisable object to store.
        """
        blob = json.dumps(payload, default=str)
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO cache (source, key, payload, fetched_at) "
                "VALUES (?, ?, ?, ?)",
                (source, key, blob, time.time()),
            )

    def get_cache(self, source: str, key: str) -> Optional[Dict[str, Any]]:
        """Return a cache entry's payload and metadata, or ``None`` if missing.

        Returns:
            A dict with keys ``payload``, ``fetched_at`` and ``age`` (seconds),
            or ``None`` when there is no entry.
        """
        with self._cursor() as cur:
            cur.execute(
                "SELECT payload, fetched_at FROM cache WHERE source = ? AND key = ?",
                (source, key),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "payload": json.loads(row["payload"]),
            "fetched_at": row["fetched_at"],
            "age": time.time() - row["fetched_at"],
        }

    def is_fresh(self, source: str, key: str, ttl: int) -> bool:
        """Return whether a cache entry exists and is younger than ``ttl``."""
        entry = self.get_cache(source, key)
        return entry is not None and entry["age"] < ttl

    def cached_payload(self, source: str, key: str) -> Optional[Any]:
        """Return only the payload for a cache entry, or ``None``."""
        entry = self.get_cache(source, key)
        return entry["payload"] if entry else None

    # ------------------------------------------------------------------ #
    # Watchlist
    # ------------------------------------------------------------------ #
    def add_to_watchlist(self, ticker: str, note: str = "") -> None:
        """Add (or update) a ticker on the watchlist."""
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO watchlist (ticker, added_at, note) "
                "VALUES (?, ?, ?)",
                (ticker.upper(), time.time(), note),
            )

    def remove_from_watchlist(self, ticker: str) -> None:
        """Remove a ticker from the watchlist."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))

    def get_watchlist(self) -> List[str]:
        """Return the list of watchlisted tickers, most recently added first."""
        with self._cursor() as cur:
            cur.execute("SELECT ticker FROM watchlist ORDER BY added_at DESC")
            return [r["ticker"] for r in cur.fetchall()]

    # ------------------------------------------------------------------ #
    # Scores
    # ------------------------------------------------------------------ #
    def save_score(self, ticker: str, payload: Dict[str, Any]) -> None:
        """Persist a computed score snapshot for a ticker."""
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO scores (ticker, computed_at, payload) "
                "VALUES (?, ?, ?)",
                (ticker.upper(), time.time(), json.dumps(payload, default=str)),
            )

    def latest_score(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return the most recent stored score snapshot for ``ticker``."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT payload, computed_at FROM scores WHERE ticker = ? "
                "ORDER BY computed_at DESC LIMIT 1",
                (ticker.upper(),),
            )
            row = cur.fetchone()
        if row is None:
            return None
        data = json.loads(row["payload"])
        data["_computed_at"] = row["computed_at"]
        return data


# Module-level singleton used throughout the app.
_DB_SINGLETON: Optional[Database] = None
_SINGLETON_LOCK = threading.Lock()


def get_db() -> Database:
    """Return the process-wide :class:`Database` singleton."""
    global _DB_SINGLETON
    if _DB_SINGLETON is None:
        with _SINGLETON_LOCK:
            if _DB_SINGLETON is None:
                _DB_SINGLETON = Database()
    return _DB_SINGLETON
