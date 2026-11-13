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


class RepositorySink:
    def __init__(self, runs: RunRepository) -> None:
        self._runs = runs

    def write(self, run_id: str, event: dict[str, Any]) -> None:
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
