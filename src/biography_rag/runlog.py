"""Audit log: every query, the chunk ids it retrieved, and every prompt sent to an LLM.

Events go to the run repository (so the live site can show them) and to a JSONL file per
run under ``log_dir`` (so they can be committed and reviewed offline)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Protocol

from biography_rag.domain import utcnow
from biography_rag.store.base import RunRepository


class EventSink(Protocol):
    def write(self, run_id: str, event: dict[str, Any]) -> None: ...


# Always stored, whatever the cap: they say how the run started, ended and was checked.
_ESSENTIAL = frozenset({"run_started", "weave", "verification", "run_failed", "run_finished"})


class RepositorySink:
    """Stores events in the database, at most ``max_events`` per run (0 = no cap), so a runaway
    run cannot fill the free database quota. Past the cap one ``events_dropped`` marker is
    written, and only essential events after it."""

    def __init__(self, runs: RunRepository, max_events: int = 0) -> None:
        self._runs = runs
        self._max = max_events
        self._counts: dict[str, int] = {}
        self._marked: set[str] = set()
        self._lock = threading.Lock()

    def write(self, run_id: str, event: dict[str, Any]) -> None:
        if self._max > 0:
            with self._lock:
                n = self._counts.get(run_id, 0) + 1
                self._counts[run_id] = n
                drop = n > self._max and event.get("type") not in _ESSENTIAL
                mark = drop and run_id not in self._marked
                if mark:
                    self._marked.add(run_id)
                if event.get("type") == "run_finished":
                    self._counts.pop(run_id, None)
                    self._marked.discard(run_id)
            if mark:
                marker = {"type": "events_dropped", "run_id": run_id, "at": event.get("at"), "limit": self._max}
                self._runs.append_event(run_id, marker)
            if drop:
                return
        self._runs.append_event(run_id, event)


class JsonlFileSink:
    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._lock = threading.Lock()

    def write(self, run_id: str, event: dict[str, Any]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock, (self._dir / f"{run_id}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


class RunLogger:
    def __init__(self, sinks: list[EventSink]) -> None:
        self._sinks = sinks

    def log(self, run_id: str, event_type: str, **fields: Any) -> None:
        event = {"type": event_type, "run_id": run_id, "at": utcnow().isoformat(), **fields}
        for sink in self._sinks:
            sink.write(run_id, event)
