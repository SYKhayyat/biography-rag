"""Hybrid retrieval: dense vectors + keyword (BM25 / Postgres FTS), fused by reciprocal rank.

Dense search finds paraphrases ("his boyhood" for "childhood"); keyword search finds rare
exact words (names, "kite", "almanac"). Reciprocal rank fusion needs no score calibration
between the two, which matters because the app must work with any embedder.

Back-of-book index pages are skipped. They match every question (they name every topic) but
state nothing: in the Franklin run they were the only evidence offered for two questions whose
answers were in the narrative chapters."""

from __future__ import annotations

import re
from collections.abc import Sequence
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
    compound_extra_k: int = 3  # more chunks for a question that asks two things


_COMPOUND = re.compile(r",\s+and\s+|;\s+|\?\s+(?=\w)", re.I)


def compound_parts(question: str) -> list[str]:
    """The clauses of a question that asks two things ("How many children did X have, and what
    were their names?"), or [] for a simple question. Each half needs its own evidence, which
    one retrieval for the whole sentence tends to miss."""
    parts = [p.strip(" ?") for p in _COMPOUND.split(question)]
    parts = [p for p in parts if len(p.split()) >= 2]
    return parts if len(parts) >= 2 else []


class HybridRetriever:
    def __init__(self, index: ChunkIndex, embedder: Embedder, config: RetrievalConfig) -> None:
        self._index = index
        self._embedder = embedder
        self._config = config

    def retrieve(
        self,
        corpus_id: str,
        question: str,
        keywords: str = "",
        alt_question: str | None = None,
        extra_phrasings: Sequence[str] = (),
    ) -> list[RetrievedChunk]:
        """``alt_question`` is a second phrasing searched alongside the first (both arms), e.g. the
        question in the first person for a memoir, where the text says "I" and rarely the name.
        ``extra_phrasings`` are more (the clauses of a compound question); when given, the result
        holds ``compound_extra_k`` more chunks, so each clause can bring its own evidence."""
        cfg = self._config
        phrasings = list(dict.fromkeys(p for p in [question, alt_question, *extra_phrasings] if p))
        top_k = cfg.top_k + (cfg.compound_extra_k if extra_phrasings else 0)
        dense_arms, sparse_arms = [], []
        for q in phrasings:
            dense = self._index.vector_search(corpus_id, self._embedder.embed_query(q), cfg.candidate_k)
            dense = [(c, s) for c, s in dense if s >= cfg.min_similarity]
            sparse = self._index.keyword_search(corpus_id, f"{q} {keywords}".strip(), cfg.candidate_k)
            if cfg.max_number_density is not None:
                dense = [(c, s) for c, s in dense if number_density(c.text) <= cfg.max_number_density]
                sparse = [(c, s) for c, s in sparse if number_density(c.text) <= cfg.max_number_density]
            dense_arms.append(dense)
            sparse_arms.append(sparse)

        fused: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        cosine: dict[str, float] = {}
        vrank: dict[str, int] = {}
        krank: dict[str, int] = {}
        for dense in dense_arms:
            for rank, (chunk, sim) in enumerate(dense, 1):
                cid = chunk.chunk_id
                fused[cid] = fused.get(cid, 0.0) + 1 / (cfg.rrf_k + rank)
                chunks[cid] = chunk
                cosine[cid] = max(sim, cosine.get(cid, sim))
                vrank[cid] = min(rank, vrank.get(cid, rank))
        for sparse in sparse_arms:
            for rank, (chunk, _score) in enumerate(sparse, 1):
                cid = chunk.chunk_id
                fused[cid] = fused.get(cid, 0.0) + 1 / (cfg.rrf_k + rank)
                chunks[cid] = chunk
                krank[cid] = min(rank, krank.get(cid, rank))

        best = sorted(fused, key=lambda cid: (-fused[cid], cid))[:top_k]
        return [
            RetrievedChunk(chunks[cid], fused[cid], cosine.get(cid), vrank.get(cid), krank.get(cid)) for cid in best
        ]
