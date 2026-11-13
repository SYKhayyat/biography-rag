"""Background work. Ingestion and story runs take tens of seconds, so the web layer hands
them to a runner and the browser polls for status."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

log = logging.getLogger(__name__)


class JobRunner(Protocol):
    def submit(self, fn: Callable[..., Any], *args: Any) -> None: ...


class ThreadJobRunner:
    def __init__(self, max_workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="job")

    def submit(self, fn: Callable[..., Any], *args: Any) -> None:
        future = self._pool.submit(fn, *args)
        future.add_done_callback(lambda f: f.exception() and log.error("job failed", exc_info=f.exception()))

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class InlineJobRunner:
    """Runs the job immediately; for tests and scripts."""

    def submit(self, fn: Callable[..., Any], *args: Any) -> None:
        fn(*args)
