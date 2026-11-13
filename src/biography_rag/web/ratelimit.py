"""Per-client sliding-window limit for the endpoints that spend money on LLM/embedding calls."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float = 3600, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        """Record a hit and say whether it is within the limit. A limit of 0 disables limiting."""
        if self.limit <= 0:
            return True
        now = self._clock()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True

    def retry_after(self, client: str) -> int:
        with self._lock:
            hits = self._hits.get(client)
            if not hits:
                return 0
            return max(1, int(self.window - (self._clock() - hits[0])))
