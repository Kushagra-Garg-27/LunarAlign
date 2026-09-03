"""
SIH26166 — Temporary in-memory result storage for registration artifacts.

Stores registration pipeline results (images, metrics) keyed by UUID
result IDs.  This is a simple dict-backed store — no database, no
persistence, no background workers.

Cleanup: results are evicted when the store exceeds ``MAX_RESULTS``.
The oldest result is removed first (FIFO).

This is intentionally minimal.  Persistent storage, job management,
and authentication belong to later stages.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any

logger = logging.getLogger("sih26166.core.result_store")

MAX_RESULTS: int = 50


class ResultStore:
    """Thread-safe in-memory result store with FIFO eviction."""

    def __init__(self, max_results: int = MAX_RESULTS) -> None:
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._max = max_results
        self._lock = threading.Lock()

    def put(self, result_id: str, result: Any) -> None:
        """Store a result, evicting the oldest if at capacity."""
        with self._lock:
            if result_id in self._store:
                self._store.move_to_end(result_id)
            else:
                if len(self._store) >= self._max:
                    evicted_id, _ = self._store.popitem(last=False)
                    logger.info("Evicted result %s (store full)", evicted_id)
                self._store[result_id] = result

    def get(self, result_id: str) -> Any | None:
        """Retrieve a result by ID, or None if not found."""
        with self._lock:
            return self._store.get(result_id)

    def remove(self, result_id: str) -> bool:
        """Remove a result. Returns True if it existed."""
        with self._lock:
            if result_id in self._store:
                del self._store[result_id]
                return True
            return False

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Global singleton
result_store = ResultStore()
