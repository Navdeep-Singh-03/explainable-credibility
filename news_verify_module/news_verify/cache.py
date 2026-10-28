"""Caches for provider responses (not for verdicts: stance depends on the claim and can change with config).

Only SUCCESSFUL provider responses are cached. Failures never are, so an outage cannot turn into a
remembered "no evidence". Cached content is re-sanitised on load, so a tampered cache file is not trusted.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from typing import Any, Dict, Optional, Protocol, Tuple


class Cache(Protocol):
    def get(self, key: str) -> Optional[Any]: ...
    def set(self, key: str, value: Any, ttl_seconds: float) -> None: ...


def make_cache_key(provider: str, fingerprint: str, query: str, limit: int) -> str:
    raw = json.dumps([provider, fingerprint, query.strip().lower(), limit], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class NullCache:
    def get(self, key: str) -> Optional[Any]:
        return None

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        return None


class MemoryCache:
    def __init__(self, clock=time.time) -> None:
        self._d: Dict[str, Tuple[float, str]] = {}
        self._lock = threading.Lock()
        self._clock = clock

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._d.get(key)
            if not item:
                return None
            if item[0] <= self._clock():
                del self._d[key]
                return None
            try:
                return json.loads(item[1])
            except ValueError:
                return None

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        with self._lock:
            self._d[key] = (self._clock() + ttl_seconds, json.dumps(value))


class SqliteCache:
    """Persistent cache using only the standard library."""

    def __init__(self, path: str, clock=time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        with self._lock, self._db:
            self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, expires_at REAL NOT NULL)")

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            row = self._db.execute("SELECT value, expires_at FROM cache WHERE key = ?", (key,)).fetchone()
        if not row or row[1] <= self._clock():
            return None
        try:
            return json.loads(row[0])
        except ValueError:
            return None

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
                             (key, json.dumps(value), self._clock() + ttl_seconds))

    def purge_expired(self) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM cache WHERE expires_at <= ?", (self._clock(),))

    def close(self) -> None:
        with self._lock:
            self._db.close()
