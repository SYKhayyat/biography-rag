"""Postgres + pgvector implementations. Works on any Postgres with the ``vector``
extension: Supabase, AWS RDS, Neon. Safe behind a transaction-mode pooler
(prepared statements are disabled)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from biography_rag.domain import Chunk, Corpus, StoryRun
from biography_rag.serialization import corpus_from_dict, corpus_to_dict, run_from_dict, run_to_dict
from biography_rag.store.base import NO_FILTERS, SearchFilters
from biography_rag.text import STOPWORDS, number_density, words

SCHEMA = """
create extension if not exists vector;

create table if not exists corpora (
    corpus_id  text primary key,
    data       jsonb not null,
    created_at timestamptz not null default now()
);

-- The embedding column is untyped so corpora indexed by different embedders
-- (1536-d OpenAI, 384-d local) can share one table. Search is an exact scan
-- filtered by corpus, which is fast at biography scale (thousands of chunks).
create table if not exists chunks (
    corpus_id text not null references corpora(corpus_id) on delete cascade,
    chunk_id  text not null,
    doc_id    text not null,
    ordinal   int  not null,
    text      text not null,
    context   text not null default '',
    num_density real not null default 0,
    embedding vector not null,
    tsv       tsvector generated always as (to_tsvector('english', text)) stored,
    primary key (corpus_id, chunk_id)
);
-- Added after the first release; existing rows keep the defaults above.
alter table chunks add column if not exists context text not null default '';
alter table chunks add column if not exists num_density real not null default 0;
create index if not exists chunks_tsv on chunks using gin (tsv);
create index if not exists chunks_doc on chunks (corpus_id, doc_id, ordinal);

create table if not exists runs (
    run_id     text primary key,
    corpus_id  text not null references corpora(corpus_id) on delete cascade,
    data       jsonb not null,
    created_at timestamptz not null default now()
);
create index if not exists runs_by_corpus on runs (corpus_id, created_at desc);

create table if not exists run_events (
    run_id text not null references runs(run_id) on delete cascade,
    seq    bigserial,
    event  jsonb not null,
    primary key (run_id, seq)
);

-- Row-level security on, no policies: Supabase's public REST API (anon and
-- authenticated roles) sees nothing. The app connects as the table owner,
-- which bypasses RLS, so it is unaffected.
alter table corpora    enable row level security;
alter table chunks     enable row level security;
alter table runs       enable row level security;
alter table run_events enable row level security;
"""


def _configure(conn: Connection) -> None:
    register_vector(conn)


def create_pool(database_url: str, *, max_size: int = 5) -> ConnectionPool:
    # Create the extension before any connection tries to register the vector type.
    with Connection.connect(database_url, autocommit=True, prepare_threshold=None) as conn:
        conn.execute(SCHEMA)
    pool = ConnectionPool(
        database_url,
        min_size=1,
        max_size=max_size,
        kwargs={"prepare_threshold": None, "autocommit": True},
        configure=_configure,
        open=True,
    )
    pool.wait()
    return pool


class PostgresCorpusRepository:
    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def save(self, corpus: Corpus) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "insert into corpora (corpus_id, data, created_at) values (%s, %s, %s) "
                "on conflict (corpus_id) do update set data = excluded.data",
                (corpus.corpus_id, Jsonb(corpus_to_dict(corpus)), corpus.created_at),
            )

    def get(self, corpus_id: str) -> Corpus | None:
        with self._pool.connection() as conn:
            row = conn.execute("select data from corpora where corpus_id = %s", (corpus_id,)).fetchone()
        return corpus_from_dict(row[0]) if row else None

    def list(self) -> list[Corpus]:
        with self._pool.connection() as conn:
            rows = conn.execute("select data from corpora order by created_at desc").fetchall()
        return [corpus_from_dict(r[0]) for r in rows]

    def delete(self, corpus_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("delete from corpora where corpus_id = %s", (corpus_id,))


def _query(text: str, operator: str) -> str:
    """tsquery text joining the content terms with ``operator``; '' when nothing is left."""
    terms = {w for w in (t.replace("'", "") for t in words(text)) if w.isalnum() and w not in STOPWORDS and len(w) > 2}
    return f" {operator} ".join(sorted(terms))


def _or_query(text: str) -> str:
    """'Who were his siblings?' -> 'siblings' ; terms OR-ed so partial matches still rank."""
    return _query(text, "|")


def _and_query(text: str) -> str:
    """The same terms AND-ed: the passages that hold every one of them, which are the ones a
    question is usually about. The OR query then fills the tail of the list."""
    return _query(text, "&")


class PostgresChunkIndex:
    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def add(self, corpus_id: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        rows = [
            (
                corpus_id,
                c.chunk_id,
                c.doc_id,
                c.ordinal,
                c.text,
                c.context,
                number_density(c.text),
                np.asarray(v, dtype=np.float32),
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.executemany(
                "insert into chunks (corpus_id, chunk_id, doc_id, ordinal, text, context, num_density, embedding) "
                "values (%s, %s, %s, %s, %s, %s, %s, %s)",
                rows,
            )

    @staticmethod
    def _where(filters: SearchFilters) -> tuple[str, list[Any]]:
        """The exclusions as SQL, so the row limit counts only rows that survive them."""
        sql, params = "", []
        if filters.max_number_density is not None:
            sql += " and num_density <= %s"
            params.append(filters.max_number_density)
        return sql, params

    def vector_search(
        self, corpus_id: str, vector: Sequence[float], k: int, filters: SearchFilters = NO_FILTERS
    ) -> list[tuple[Chunk, float]]:
        extra, extra_params = self._where(filters)
        dense = np.asarray(vector, dtype=np.float32)
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"select chunk_id, doc_id, ordinal, text, context, 1 - (embedding <=> %s) as sim "
                f"from chunks where corpus_id = %s{extra} and 1 - (embedding <=> %s) >= %s "
                f"order by embedding <=> %s limit %s",
                (dense, corpus_id, *extra_params, dense, filters.min_similarity, dense, k),
            ).fetchall()
        return [(Chunk(r[0], r[1], r[2], r[3], r[4]), float(r[5])) for r in rows]

    def keyword_search(
        self, corpus_id: str, text: str, k: int, filters: SearchFilters = NO_FILTERS
    ) -> list[tuple[Chunk, float]]:
        query = _or_query(text)
        if not query:
            return []
        whole = _and_query(text)
        extra, extra_params = self._where(filters)
        with self._pool.connection() as conn:
            rows = conn.execute(
                "select chunk_id, doc_id, ordinal, text, context, ts_rank_cd(tsv, qany) as rank "
                "from chunks, to_tsquery('english', %s) qall, to_tsquery('english', %s) qany "
                f"where corpus_id = %s and tsv @@ qany{extra} "
                "order by (tsv @@ qall) desc, rank desc, char_length(text) asc limit %s",
                (whole, query, corpus_id, *extra_params, k),
            ).fetchall()
        return [(Chunk(r[0], r[1], r[2], r[3], r[4]), float(r[5])) for r in rows]

    def window(self, corpus_id: str, chunk: Chunk, radius: int) -> str:
        if radius <= 0:
            return chunk.text
        with self._pool.connection() as conn:
            rows = conn.execute(
                "select text from chunks where corpus_id = %s and doc_id = %s and ordinal between %s and %s "
                "order by ordinal",
                (corpus_id, chunk.doc_id, chunk.ordinal - radius, chunk.ordinal + radius),
            ).fetchall()
        return "\n\n".join(r[0] for r in rows) or chunk.text

    def get_many(self, corpus_id: str, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        with self._pool.connection() as conn:
            rows = conn.execute(
                "select chunk_id, doc_id, ordinal, text, context from chunks "
                "where corpus_id = %s and chunk_id = any(%s)",
                (corpus_id, list(chunk_ids)),
            ).fetchall()
        return {r[0]: Chunk(r[0], r[1], r[2], r[3], r[4]) for r in rows}

    def delete_corpus(self, corpus_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("delete from chunks where corpus_id = %s", (corpus_id,))


class PostgresRunRepository:
    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def save(self, run: StoryRun) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "insert into runs (run_id, corpus_id, data, created_at) values (%s, %s, %s, %s) "
                "on conflict (run_id) do update set data = excluded.data",
                (run.run_id, run.corpus_id, Jsonb(run_to_dict(run)), run.created_at),
            )

    def get(self, run_id: str) -> StoryRun | None:
        with self._pool.connection() as conn:
            row = conn.execute("select data from runs where run_id = %s", (run_id,)).fetchone()
        return run_from_dict(row[0]) if row else None

    def list_for_corpus(self, corpus_id: str) -> list[StoryRun]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "select data from runs where corpus_id = %s order by created_at desc", (corpus_id,)
            ).fetchall()
        return [run_from_dict(r[0]) for r in rows]

    def append_event(self, run_id: str, event: dict[str, Any]) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "insert into run_events (run_id, event) values (%s, %s)",
                (run_id, Jsonb(json.loads(json.dumps(event, default=str)))),
            )

    def events(self, run_id: str) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute("select event from run_events where run_id = %s order by seq", (run_id,)).fetchall()
        return [r[0] for r in rows]
