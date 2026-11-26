"""Hybrid retrieval: dense vectors + keyword (BM25 / Postgres FTS), fused by reciprocal rank.

Dense search finds paraphrases ("his boyhood" for "childhood"); keyword search finds rare
exact words (names, "kite", "almanac"). Reciprocal rank fusion needs no score calibration
between the two, which matters because the app must work with any embedder.

Back-of-book index pages are skipped. They match every question (they name every topic) but
state nothing: in the Franklin run they were the only evidence offered for two questions whose
answers were in the narrative chapters.

Five things changed after reviewing the first version:

* the exclusions (cosine floor, index pages) now go to the index and are applied *before* its row
  limit, instead of discarding rows it had already spent slots on;
* RRF is a mean over arms, not a sum. Summed, a passage ranked #20 by six phrasings beat one
  ranked #1 by a single phrasing, so multi-part questions rewarded passages that vaguely matched
  everything;
* at most ``max_per_doc`` passages come from one book, so six biographies of one person do not
  fill the context with the same anecdote told six ways;
* an optional reranker may reorder the pool before the quota is taken (see ``rag.rerank``);
* a passage is widened to its neighbours before the model sees it (see ``domain.merge_windows``).

One entry point, ``retrieve_logged``, so the whole order — fuse, rerank, diversify, widen — is
readable in one place, and the retriever keeps no state (the 20 questions run in threads).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from biography_rag.domain import Chunk, RetrievedChunk, merge_windows
from biography_rag.embeddings.base import Embedder
from biography_rag.rag.rerank import Reranker
from biography_rag.store.base import ChunkIndex, SearchFilters


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 6
    candidate_k: int = 24
    # Drop dense hits below this cosine. 0 turns the floor off; the app default (config.py) is a
    # small positive number, so "nothing relevant" can be said before the LLM is asked at all.
    min_similarity: float = 0.0
    rrf_k: int = 60
    max_number_density: float | None = 0.15  # skip index-like chunks above this; None = keep all
    compound_extra_k: int = 3  # more chunks for a question that asks two things
    max_per_doc: int = 2  # per book, never fewer than an even share of the quota
    neighbour_radius: int = 1  # passages added on each side of a hit; 0 = send the chunk alone
    rerank_pool: int = 12  # passages the reranker chooses from, when one is configured


_COMPOUND = re.compile(r",\s+and\s+|;\s+|\?\s+(?=\w)", re.I)


def compound_parts(question: str) -> list[str]:
    """The clauses of a question that asks two things ("How many children did X have, and what
    were their names?"), or [] for a simple question. Each half needs its own evidence, which
    one retrieval for the whole sentence tends to miss."""
    parts = [p.strip(" ?") for p in _COMPOUND.split(question)]
    parts = [p for p in parts if len(p.split()) >= 2]
    return parts if len(parts) >= 2 else []


def _choose(
    ranked: list[str],
    chunks: dict[str, Chunk],
    k: int,
    max_per_doc: int,
    radius: int,
) -> list[str]:
    """The passages to keep, in rank order, at most ``max_per_doc`` per book.

    A chunk that already sits inside a kept chunk's neighbour window is passed over rather than
    kept twice: the same paragraph would otherwise reach the model twice, once as itself and
    once as its neighbour's context. So the quota counts *windows*, not chunks, and the scan
    keeps going down the ranking until it has ``k`` of them.

    The per-document cap flexes up when the corpus is small: with one document (a memoir) a flat
    cap of two would throw away four of six passages. It never falls below an even share of ``k``.
    """
    if k <= 0 or not ranked:
        return []
    docs = {chunks[cid].doc_id for cid in ranked}
    cap = min(k, max(max_per_doc, math.ceil(k / len(docs)))) if max_per_doc > 0 else k
    kept: list[str] = []
    crowded: list[str] = []  # blocked only by the per-document cap
    seen: Counter[str] = Counter()
    covered: dict[str, int] = {}  # doc_id -> highest ordinal already inside a kept window

    def inside_window(chunk: Chunk) -> bool:
        return radius > 0 and chunk.ordinal <= covered.get(chunk.doc_id, -1)

    def take(cid: str) -> None:
        chunk = chunks[cid]
        kept.append(cid)
        if radius > 0:
            covered[chunk.doc_id] = chunk.ordinal + radius

    for cid in ranked:
        if inside_window(chunks[cid]):
            continue
        if seen[chunks[cid].doc_id] >= cap:
            crowded.append(cid)
            continue
        seen[chunks[cid].doc_id] += 1
        take(cid)
        if len(kept) == k:
            return kept
    for cid in crowded:  # the cap left room: fill it rather than answer with fewer passages
        if inside_window(chunks[cid]):
            continue
        take(cid)
        if len(kept) == k:
            break
    return kept[:k]


class HybridRetriever:
    def __init__(
        self, index: ChunkIndex, embedder: Embedder, config: RetrievalConfig, reranker: Reranker | None = None
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._config = config
        self._reranker = reranker

    def quota(self, clauses: Sequence[str] = ()) -> int:
        """How many passages to send. Only a question that asks two things gets more: the extra
        phrasings an expander invents are more ways to ask *one* thing, not a second topic."""
        return self._config.top_k + (self._config.compound_extra_k if clauses else 0)

    def retrieve(
        self,
        corpus_id: str,
        question: str,
        keywords: str = "",
        alt_question: str | None = None,
        extra_phrasings: Sequence[str] = (),
        hypothetical: str = "",
        clauses: Sequence[str] = (),
    ) -> list[RetrievedChunk]:
        return self.retrieve_logged(
            corpus_id, question, keywords, alt_question, extra_phrasings, hypothetical, clauses
        )[0]

    def retrieve_logged(
        self,
        corpus_id: str,
        question: str,
        keywords: str = "",
        alt_question: str | None = None,
        extra_phrasings: Sequence[str] = (),
        hypothetical: str = "",
        clauses: Sequence[str] = (),
    ) -> tuple[list[RetrievedChunk], str | None]:
        """The passages to answer from, and a note about reranking (None when there was none).

        ``alt_question`` is a second phrasing searched alongside the first (both arms), e.g. the
        question in the first person for a memoir, where the text says "I" and rarely the name.
        ``extra_phrasings`` are more (the clauses of a compound question, or the expander's
        rephrasings); when given, the result holds ``compound_extra_k`` more chunks, so each
        clause can bring its own evidence. ``clauses`` are the halves of a compound question: they
        widen the quota, where ordinary extra phrasings do not. ``hypothetical`` is a written-out
        answer (HyDE): it is a passage, not a question, so only the dense arm searches it.
        """
        cfg = self._config
        phrasings = list(dict.fromkeys(p for p in [question, alt_question, *extra_phrasings] if p))
        filters = SearchFilters(cfg.min_similarity, cfg.max_number_density)
        dense_arms: list[list[tuple[Chunk, float]]] = []
        sparse_arms: list[list[tuple[Chunk, float]]] = []
        for q in phrasings:
            dense_arms.append(
                self._index.vector_search(corpus_id, self._embedder.embed_query(q), cfg.candidate_k, filters)
            )
            sparse_arms.append(
                self._index.keyword_search(corpus_id, f"{q} {keywords}".strip(), cfg.candidate_k, filters)
            )
        if hypothetical:
            dense_arms.append(
                self._index.vector_search(corpus_id, self._embedder.embed_query(hypothetical), cfg.candidate_k, filters)
            )

        # Mean over arms: every arm gets one vote, so a phrasing nobody agrees with is not
        # outvoted by the sheer number of phrasings that weakly match a passage.
        weight = 1.0 / max(1, len(dense_arms) + len(sparse_arms))
        fused: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        cosine: dict[str, float] = {}
        vrank: dict[str, int] = {}
        krank: dict[str, int] = {}
        for dense in dense_arms:
            for rank, (chunk, sim) in enumerate(dense, 1):
                cid = chunk.chunk_id
                fused[cid] = fused.get(cid, 0.0) + weight / (cfg.rrf_k + rank)
                chunks[cid] = chunk
                cosine[cid] = max(sim, cosine.get(cid, sim))
                vrank[cid] = min(rank, vrank.get(cid, rank))
        for sparse in sparse_arms:
            for rank, (chunk, _score) in enumerate(sparse, 1):
                cid = chunk.chunk_id
                fused[cid] = fused.get(cid, 0.0) + weight / (cfg.rrf_k + rank)
                chunks[cid] = chunk
                krank[cid] = min(rank, krank.get(cid, rank))

        ranked = sorted(fused, key=lambda cid: (-fused[cid], cid))
        note: str | None = None
        if self._reranker is not None and len(ranked) > 1:
            pool = [self._hit(cid, fused, cosine, vrank, krank, chunks) for cid in ranked[: cfg.rerank_pool]]
            # The whole pool is reordered; the quota is cut afterwards, by `_choose`.
            outcome = self._reranker.rerank_logged(question, pool, len(pool))
            note = outcome.note
            first = [r.chunk.chunk_id for r in outcome.hits]
            ranked = first + [cid for cid in ranked if cid not in set(first)]

        chosen = _choose(ranked, chunks, self.quota(clauses), cfg.max_per_doc, cfg.neighbour_radius)
        widened = {c.chunk_id: c for c in self._widen(corpus_id, [chunks[cid] for cid in chosen])}
        return [self._hit(cid, fused, cosine, vrank, krank, widened) for cid in widened], note

    def _hit(
        self,
        chunk_id: str,
        fused: dict[str, float],
        cosine: dict[str, float],
        vrank: dict[str, int],
        krank: dict[str, int],
        chunks: dict[str, Chunk],
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunks[chunk_id], fused[chunk_id], cosine.get(chunk_id), vrank.get(chunk_id), krank.get(chunk_id)
        )

    def _widen(self, corpus_id: str, chunks: Sequence[Chunk]) -> list[Chunk]:
        radius = self._config.neighbour_radius
        if radius <= 0 or not chunks:
            return list(chunks)
        texts = {c.chunk_id: self._index.window(corpus_id, c, radius) for c in chunks}
        return merge_windows(chunks, radius, texts)
