"""In-process implementations: used by the tests and for running locally without Postgres."""

from __future__ import annotations

import copy
import math
import threading
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from biography_rag.domain import Chunk, Corpus, StoryRun
from biography_rag.text import content_terms


class MemoryCorpusRepository:
    def __init__(self) -> None:
        self._items: dict[str, Corpus] = {}
        self._lock = threading.Lock()

    def save(self, corpus: Corpus) -> None:
        with self._lock:
            self._items[corpus.corpus_id] = copy.deepcopy(corpus)

    def get(self, corpus_id: str) -> Corpus | None:
        with self._lock:
            item = self._items.get(corpus_id)
            return copy.deepcopy(item) if item else None

    def list(self) -> list[Corpus]:
        with self._lock:
            return sorted((copy.deepcopy(c) for c in self._items.values()), key=lambda c: c.created_at, reverse=True)

    def delete(self, corpus_id: str) -> None:
        with self._lock:
            self._items.pop(corpus_id, None)


class _Bm25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(content_terms(c.text)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = set(content_terms(query))
        out = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for t in terms:
                f = doc.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out


class MemoryChunkIndex:
    def __init__(self) -> None:
        self._chunks: dict[str, list[Chunk]] = defaultdict(list)
        self._vectors: dict[str, list[list[float]]] = defaultdict(list)
        self._bm25: dict[str, _Bm25] = {}
        self._lock = threading.Lock()

    def add(self, corpus_id: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        with self._lock:
            self._chunks[corpus_id].extend(chunks)
            self._vectors[corpus_id].extend(list(v) for v in vectors)
            self._bm25.pop(corpus_id, None)

    def vector_search(self, corpus_id: str, vector: Sequence[float], k: int) -> list[tuple[Chunk, float]]:
        with self._lock:
            chunks, vectors = self._chunks.get(corpus_id, []), self._vectors.get(corpus_id, [])
            if not chunks:
                return []
            sims = np.asarray(vectors, dtype=np.float32) @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-sims)[:k]
        return [(chunks[i], float(sims[i])) for i in order]

    def keyword_search(self, corpus_id: str, text: str, k: int) -> list[tuple[Chunk, float]]:
        with self._lock:
            chunks = self._chunks.get(corpus_id, [])
            if not chunks:
                return []
            bm25 = self._bm25.get(corpus_id) or self._bm25.setdefault(corpus_id, _Bm25(chunks))
        scores = bm25.scores(text)
        ranked = sorted(((s, i) for i, s in enumerate(scores) if s > 0), reverse=True)[:k]
        return [(chunks[i], s) for s, i in ranked]

    def get_many(self, corpus_id: str, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        wanted = set(chunk_ids)
        with self._lock:
            return {c.chunk_id: c for c in self._chunks.get(corpus_id, []) if c.chunk_id in wanted}

    def delete_corpus(self, corpus_id: str) -> None:
        with self._lock:
            self._chunks.pop(corpus_id, None)
            self._vectors.pop(corpus_id, None)
            self._bm25.pop(corpus_id, None)


class MemoryRunRepository:
    def __init__(self) -> None:
        self._runs: dict[str, StoryRun] = {}
        self._events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()

    def save(self, run: StoryRun) -> None:
        with self._lock:
            self._runs[run.run_id] = copy.deepcopy(run)

    def get(self, run_id: str) -> StoryRun | None:
        with self._lock:
            run = self._runs.get(run_id)
            return copy.deepcopy(run) if run else None

    def list_for_corpus(self, corpus_id: str) -> list[StoryRun]:
        with self._lock:
            runs = [copy.deepcopy(r) for r in self._runs.values() if r.corpus_id == corpus_id]
        return sorted(runs, key=lambda r: r.created_at, reverse=True)

    def append_event(self, run_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            self._events[run_id].append(copy.deepcopy(event))

    def events(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._events.get(run_id, []))
