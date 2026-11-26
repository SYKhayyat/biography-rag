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
    for c in (corpus, other):
        index.delete_corpus(c.corpus_id)
        corpora.delete(c.corpus_id)


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


def test_filters_apply_before_the_row_limit(stores):
    """An index page matches every question. The floor and the index filter must run in SQL, so
    the limit counts rows that survive them."""
    from biography_rag.store.base import SearchFilters

    corpora, index, _ = stores
    corpus = new_corpus(corpora)
    emb = HashingEmbedder()
    index_page = "Franklin, Benjamin: born 2, 7, 11, 14, 19; Boston 4, 9, 12, 22, 31, 36, 41, 44."
    chunks = [
        Chunk("d01-0000", "d01", 0, index_page),
        Chunk("d01-0001", "d01", 1, "Franklin was born in Boston in the year seventeen oh six."),
        Chunk("d01-0002", "d01", 2, "He was born in Boston and went to school there."),
    ]
    index.add(corpus.corpus_id, chunks, emb.embed_documents([c.embed_input for c in chunks]))
    filters = SearchFilters(min_similarity=0.0, max_number_density=0.15)
    query = emb.embed_query(index_page)  # the index page is its own best match, by construction
    unfiltered = index.vector_search(corpus.corpus_id, query, 3)
    assert unfiltered[0][0].chunk_id == "d01-0000"
    hits = index.vector_search(corpus.corpus_id, query, 2, filters)
    assert [c.chunk_id for c, _ in hits] == ["d01-0001", "d01-0002"]  # two real passages, not one
    assert index.keyword_search(corpus.corpus_id, "born Boston", 2, filters)[0][0].chunk_id != "d01-0000"
    index.delete_corpus(corpus.corpus_id)
    corpora.delete(corpus.corpus_id)


def test_keyword_search_ranks_all_terms_first(stores):
    """An OR query lets one common word carry a hit. Documents holding every term come first."""
    corpora, index, _ = stores
    corpus = new_corpus(corpora)
    emb = HashingEmbedder()
    chunks = [
        Chunk("d01-0000", "d01", 0, "Kite kite kite kite kite kite, over and over, a kite party."),
        Chunk("d01-0001", "d01", 1, "He flew a kite in a thunderstorm to show that lightning is electricity."),
        Chunk("d01-0002", "d01", 2, "The almanac sold well in every colony for many years."),
    ]
    index.add(corpus.corpus_id, chunks, emb.embed_documents([c.embed_input for c in chunks]))
    hits = [c.chunk_id for c, _ in index.keyword_search(corpus.corpus_id, "kite lightning", 3)]
    assert hits[0] == "d01-0001"  # the only chunk with both terms
    assert "d01-0002" not in hits  # no term at all
    index.delete_corpus(corpus.corpus_id)
    corpora.delete(corpus.corpus_id)


def test_neighbours_come_back_in_reading_order(stores):
    corpora, index, _ = stores
    corpus = new_corpus(corpora)
    emb = HashingEmbedder()
    chunks = [
        Chunk("d01-0000", "d01", 0, "First passage of book one.", context="Book One"),
        Chunk("d01-0001", "d01", 1, "Second passage of book one.", context="Book One"),
        Chunk("d01-0002", "d01", 2, "Third passage of book one.", context="Book One"),
        Chunk("d02-0000", "d02", 0, "First passage of book two."),
    ]
    index.add(corpus.corpus_id, chunks, emb.embed_documents([c.embed_input for c in chunks]))
    before, after = index.neighbours(corpus.corpus_id, chunks[1], 1)
    assert (before, after) == ("First passage of book one.", "Third passage of book one.")
    assert index.neighbours(corpus.corpus_id, chunks[1], 0) == ("", "")
    assert index.neighbours(corpus.corpus_id, chunks[3], 1) == ("", "")  # nothing either side there
    assert index.get_many(corpus.corpus_id, ["d01-0000"])["d01-0000"].context == "Book One"
    index.delete_corpus(corpus.corpus_id)
    corpora.delete(corpus.corpus_id)
