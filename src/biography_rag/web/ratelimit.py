"""Per-client sliding-window limit for the endpoints that spend money on LLM/embedding calls."""

from __future__ import annotations

import ipaddress
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

    def allow(self, client: str, cost: int = 1) -> bool:
        """Record `cost` hits if they all fit within the limit; otherwise record none.
        A limit of 0 disables limiting."""
        if self.limit <= 0:
            return True
        now = self._clock()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) + cost > self.limit:
                return False
            hits.extend([now] * cost)
            return True

    def retry_after(self, client: str) -> int:
        with self._lock:
            hits = self._hits.get(client)
            if not hits:
                return 0
            return max(1, int(self.window - (self._clock() - hits[0])))


# Cloudflare's published edge ranges (https://www.cloudflare.com/ips/). The custom domain is
# proxied through Cloudflare, which appends the address that connected to it.
CLOUDFLARE_RANGES = tuple(
    ipaddress.ip_network(n)
    for n in (
        "173.245.48.0/20", "103.21.244.0/22", "103.22.200.0/22", "103.31.4.0/22", "141.101.64.0/18",
        "108.162.192.0/18", "190.93.240.0/20", "188.114.96.0/20", "197.234.240.0/22", "198.41.128.0/17",
        "162.158.0.0/15", "104.16.0.0/13", "104.24.0.0/14", "172.64.0.0/13", "131.0.72.0/22",
        "2400:cb00::/32", "2606:4700::/32", "2803:f800::/32", "2405:b500::/32", "2405:8100::/32",
        "2a06:98c0::/29", "2c0f:f248::/32",
    )
)  # fmt: skip


def _is_cloudflare(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return any(ip in net for net in CLOUDFLARE_RANGES)


def client_ip(forwarded_for: str | None, peer: str | None, *, trust_forwarded_for: bool) -> str:
    """The address to rate-limit on. Entries a client writes into X-Forwarded-For sit on the
    left; each proxy appends the address that connected to it on the right. So read from the
    right: the last entry is whoever reached Render's edge, and if that is Cloudflare, the
    entry before it is whoever reached Cloudflare. Anything further left is client-supplied."""
    if not trust_forwarded_for or not forwarded_for:
        return peer or "unknown"
    hops = [h.strip() for h in forwarded_for.split(",") if h.strip()]
    if not hops:
        return peer or "unknown"
    if len(hops) >= 2 and _is_cloudflare(hops[-1]):
        return hops[-2]
    return hops[-1]
