"""Integration tests for the Postgres/pgvector store. Skipped unless TEST_DATABASE_URL is set.

scripts/test_postgres.sh      # starts a throwaway local Postgres with pgvector (nix) and runs these
"""

from __future__ import annotations

import os
import uuid

import pytest

from biography_rag.domain import Chunk, Corpus, CorpusStatus, RunStatus, StoryRun
from biography_rag.ingest.chunker import ParagraphChunker
from biography_rag.rag.retriever import RetrievalConfig
from biography_rag.runlog import RepositorySink, RunLogger
from biography_rag.services import IngestService, StoryConfig, StoryService
from tests.fakes import FakeChatSource, HashingEmbedder, fake_registry

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = [pytest.mark.postgres, pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")]


@pytest.fixture(scope="module")
def pool():
    from biography_rag.store.postgres import create_pool

    p = create_pool(URL)
    yield p
    p.close()


@pytest.fixture
def stores(pool):
    from biography_rag.store.postgres import PostgresChunkIndex, PostgresCorpusRepository, PostgresRunRepository

    return PostgresCorpusRepository(pool), PostgresChunkIndex(pool), PostgresRunRepository(pool)


def new_corpus(corpora) -> Corpus:
    corpus = Corpus(f"t-{uuid.uuid4().hex[:8]}", "Ben", "fake:hashing", status=CorpusStatus.READY)
    corpora.save(corpus)
    return corpus


def test_schema_is_idempotent(pool):
    from biography_rag.store.postgres import create_pool

    create_pool(URL).close()  # running the schema twice must not fail


def test_corpus_round_trip(stores):
    corpora, _, _ = stores
    corpus = new_corpus(corpora)
    assert corpora.get(corpus.corpus_id) == corpus
    assert corpus.corpus_id in [c.corpus_id for c in corpora.list()]
    corpora.delete(corpus.corpus_id)
    assert corpora.get(corpus.corpus_id) is None


def test_chunk_index_search(stores):
    corpora, index, _ = stores
    corpus = new_corpus(corpora)
    other = new_corpus(corpora)
    emb = HashingEmbedder()
    chunks = [
        Chunk("d01-0000", "d01", 0, "Franklin was born in Boston in 1706."),
        Chunk("d01-0001", "d01", 1, "He flew a kite to prove lightning was electricity."),
        Chunk("d01-0002", "d01", 2, "The printing shop on Market Street did well."),
    ]
    index.add(corpus.corpus_id, chunks, emb.embed_documents([c.text for c in chunks]))
    index.add(other.corpus_id, chunks[:1], emb.embed_documents([chunks[0].text]))

    hits = index.vector_search(corpus.corpus_id, emb.embed_query("kite lightning electricity"), 2)
    assert hits[0][0].chunk_id == "d01-0001" and len(hits) == 2
    assert -1.0 <= hits[0][1] <= 1.0

    kw = index.keyword_search(corpus.corpus_id, "born birth birthplace", 5)
    assert [c.chunk_id for c, _ in kw] == ["d01-0000"]  # OR query: any matching term is enough
    assert index.keyword_search(corpus.corpus_id, "!!! & | ()", 5) == []  # tsquery syntax is neutralised

    assert set(index.get_many(corpus.corpus_id, ["d01-0002", "d09-0000"])) == {"d01-0002"}
    index.delete_corpus(corpus.corpus_id)
    assert index.vector_search(corpus.corpus_id, emb.embed_query("kite"), 5) == []
    assert len(index.vector_search(other.corpus_id, emb.embed_query("kite"), 5)) == 1  # scoped by corpus


def test_runs_and_events(stores):
    corpora, _, runs = stores
    corpus = new_corpus(corpora)
    run = StoryRun(f"r{uuid.uuid4().hex[:8]}", corpus.corpus_id, "Ben", "fake", "fake-1")
    runs.save(run)
    run.status = RunStatus.DONE
    runs.save(run)  # upsert
    assert runs.get(run.run_id).status is RunStatus.DONE
    assert [r.run_id for r in runs.list_for_corpus(corpus.corpus_id)] == [run.run_id]
    for n in range(3):
        runs.append_event(run.run_id, {"type": "query", "n": n, "text": "unicode — ok"})
    assert [e["n"] for e in runs.events(run.run_id)] == [0, 1, 2]
    corpora.delete(corpus.corpus_id)
    assert runs.get(run.run_id) is None  # cascades


def test_full_story_on_postgres(stores, bio_file, chat):
    corpora, index, runs = stores
    embedders = fake_registry()
    ingest = IngestService(corpora, index, embedders, ParagraphChunker(size=200, overlap=0))
    story = StoryService(
        corpora,
        index,
        runs,
        embedders,
        FakeChatSource(chat),
        RunLogger([RepositorySink(runs)]),
        StoryConfig(RetrievalConfig(top_k=3, candidate_k=6), concurrency=4),
    )
    corpus = ingest.create("Benjamin Franklin", [bio_file])
    assert ingest.ingest(corpus.corpus_id, [bio_file]).status is CorpusStatus.READY
    run = story.execute(story.start(corpus.corpus_id).run_id)
    assert run.status is RunStatus.DONE, run.error
    assert run.answers[0].found and run.verification.flagged == []
    assert sum(e["type"] == "query" for e in runs.events(run.run_id)) == 20
    ingest.delete(corpus.corpus_id)
