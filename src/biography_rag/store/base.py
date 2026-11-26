"""Storage ports. Kept narrow (interface segregation): each service depends only on the
repository it needs, and each has an in-memory and a Postgres implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from biography_rag.domain import Chunk, Corpus, StoryRun


@dataclass(frozen=True)
class SearchFilters:
    """Exclusions applied by the index *before* the row limit, so a discarded hit never
    consumes one of the ``k`` slots. Filtering after ``LIMIT`` (as this code used to) means a
    question whose nearest neighbours are index pages returns fewer than ``k`` passages, or
    none at all."""

    min_similarity: float = 0.0  # dense arm: drop hits below this cosine
    max_number_density: float | None = None  # drop back-of-book index passages (None = keep)

    @property
    def active(self) -> bool:
        return self.min_similarity > 0.0 or self.max_number_density is not None


NO_FILTERS = SearchFilters()


class CorpusRepository(Protocol):
    def save(self, corpus: Corpus) -> None: ...
    def get(self, corpus_id: str) -> Corpus | None: ...
    def list(self) -> list[Corpus]: ...
    def delete(self, corpus_id: str) -> None: ...


class ChunkIndex(Protocol):
    def add(self, corpus_id: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...
    def vector_search(
        self, corpus_id: str, vector: Sequence[float], k: int, filters: SearchFilters = NO_FILTERS
    ) -> list[tuple[Chunk, float]]: ...
    def keyword_search(
        self, corpus_id: str, text: str, k: int, filters: SearchFilters = NO_FILTERS
    ) -> list[tuple[Chunk, float]]: ...
    def window(self, corpus_id: str, chunk: Chunk, radius: int) -> str:
        """The chunk plus ``radius`` neighbours on each side of it in the same document,
        in reading order. Returns ``chunk.text`` when there is nothing to widen."""
        ...

    def get_many(self, corpus_id: str, chunk_ids: Sequence[str]) -> dict[str, Chunk]: ...
    def delete_corpus(self, corpus_id: str) -> None: ...


class RunRepository(Protocol):
    def save(self, run: StoryRun) -> None: ...
    def get(self, run_id: str) -> StoryRun | None: ...
    def list_for_corpus(self, corpus_id: str) -> list[StoryRun]: ...
    def append_event(self, run_id: str, event: dict[str, Any]) -> None: ...
    def events(self, run_id: str) -> list[dict[str, Any]]: ...
