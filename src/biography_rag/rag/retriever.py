"""Hybrid retrieval: dense vectors + keyword (BM25 / Postgres FTS), fused by reciprocal rank.

Dense search finds paraphrases ("his boyhood" for "childhood"); keyword search finds rare
exact words (names, "kite", "almanac"). Reciprocal rank fusion needs no score calibration
between the two, which matters because the app must work with any embedder.

Back-of-book index pages are skipped. They match every question (they name every topic) but
state nothing: in the Franklin run they were the only evidence offered for two questions whose
answers were in the narrative chapters."""

from __future__ import annotations

from dataclasses import dataclass

from biography_rag.domain import Chunk, RetrievedChunk
from biography_rag.embeddings.base import Embedder
from biography_rag.store.base import ChunkIndex
from biography_rag.text import number_density


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 6
    candidate_k: int = 24
    min_similarity: float = 0.0
    rrf_k: int = 60
    max_number_density: float | None = 0.15  # skip index-like chunks above this; None = keep all


class HybridRetriever:
    def __init__(self, index: ChunkIndex, embedder: Embedder, config: RetrievalConfig) -> None:
        self._index = index
        self._embedder = embedder
        self._config = config

    def retrieve(self, corpus_id: str, question: str, keywords: str = "") -> list[RetrievedChunk]:
        cfg = self._config
        dense = self._index.vector_search(corpus_id, self._embedder.embed_query(question), cfg.candidate_k)
        dense = [(c, s) for c, s in dense if s >= cfg.min_similarity]
        sparse = self._index.keyword_search(corpus_id, f"{question} {keywords}".strip(), cfg.candidate_k)
        if cfg.max_number_density is not None:
            dense = [(c, s) for c, s in dense if number_density(c.text) <= cfg.max_number_density]
            sparse = [(c, s) for c, s in sparse if number_density(c.text) <= cfg.max_number_density]

        fused: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        cosine: dict[str, float] = {}
        vrank: dict[str, int] = {}
        krank: dict[str, int] = {}
        for rank, (chunk, sim) in enumerate(dense, 1):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1 / (cfg.rrf_k + rank)
            chunks[chunk.chunk_id], cosine[chunk.chunk_id], vrank[chunk.chunk_id] = chunk, sim, rank
        for rank, (chunk, _score) in enumerate(sparse, 1):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1 / (cfg.rrf_k + rank)
            chunks[chunk.chunk_id], krank[chunk.chunk_id] = chunk, rank

        best = sorted(fused, key=lambda cid: (-fused[cid], cid))[: cfg.top_k]
        return [
            RetrievedChunk(chunks[cid], fused[cid], cosine.get(cid), vrank.get(cid), krank.get(cid)) for cid in best
        ]
