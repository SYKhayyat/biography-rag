"""Composition root: the only place that knows which concrete classes back each port."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from biography_rag.config import Settings
from biography_rag.embeddings.registry import EmbedderRegistry
from biography_rag.ingest.chunker import ParagraphChunker
from biography_rag.jobs import JobRunner, ThreadJobRunner
from biography_rag.llm.base import ChatModelSource
from biography_rag.llm.providers import ChatModelFactory
from biography_rag.rag.retriever import RetrievalConfig
from biography_rag.runlog import JsonlFileSink, RepositorySink, RunLogger
from biography_rag.services import IngestService, StoryConfig, StoryService
from biography_rag.store.base import ChunkIndex, CorpusRepository, RunRepository

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    corpora: CorpusRepository
    index: ChunkIndex
    runs: RunRepository
    embedders: EmbedderRegistry
    chats: ChatModelSource
    ingest: IngestService
    story: StoryService
    jobs: JobRunner
    closer: Any = None  # e.g. the Postgres pool

    def close(self) -> None:
        shutdown = getattr(self.jobs, "shutdown", None)
        if shutdown:
            shutdown()
        if self.closer is not None:
            self.closer.close()


def _stores(settings: Settings) -> tuple[CorpusRepository, ChunkIndex, RunRepository, Any]:
    if not settings.database_url:
        from biography_rag.store.memory import MemoryChunkIndex, MemoryCorpusRepository, MemoryRunRepository

        log.warning("DATABASE_URL is empty: using in-memory storage (lost on restart)")
        return MemoryCorpusRepository(), MemoryChunkIndex(), MemoryRunRepository(), None
    from biography_rag.store.postgres import (
        PostgresChunkIndex,
        PostgresCorpusRepository,
        PostgresRunRepository,
        create_pool,
    )

    pool = create_pool(settings.database_url)
    return PostgresCorpusRepository(pool), PostgresChunkIndex(pool), PostgresRunRepository(pool), pool


def build_container(
    settings: Settings,
    *,
    embedders: EmbedderRegistry | None = None,
    chats: ChatModelSource | None = None,
    jobs: JobRunner | None = None,
    stores: tuple[CorpusRepository, ChunkIndex, RunRepository] | None = None,
) -> Container:
    """Wire the app from settings; tests inject fakes through the keyword arguments."""
    closer = None
    if stores is None:
        corpora, index, runs, closer = _stores(settings)
    else:
        corpora, index, runs = stores
    embedders = embedders or EmbedderRegistry.from_settings(settings)
    chats = chats or ChatModelFactory(settings)
    logger = RunLogger([RepositorySink(runs), JsonlFileSink(settings.log_dir)])
    retrieval = RetrievalConfig(
        top_k=settings.top_k,
        candidate_k=settings.candidate_k,
        min_similarity=settings.min_similarity,
        max_number_density=settings.max_number_density,
    )
    ingest = IngestService(
        corpora, index, embedders, ParagraphChunker(size=settings.chunk_size, overlap=settings.chunk_overlap)
    )
    story = StoryService(
        corpora,
        index,
        runs,
        embedders,
        chats,
        logger,
        StoryConfig(retrieval, concurrency=settings.query_concurrency, verify_with_llm=settings.verify_with_llm),
    )
    return Container(
        settings,
        corpora,
        index,
        runs,
        embedders,
        chats,
        ingest,
        story,
        jobs or ThreadJobRunner(settings.job_workers),
        closer,
    )
