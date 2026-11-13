"""Use cases. Services depend on ports (repositories, embedder registry, chat source),
never on concrete infrastructure, so tests run them entirely in memory."""

from __future__ import annotations

import logging
import re
import threading
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import PurePath

from biography_rag.domain import (
    Chunk,
    Corpus,
    CorpusStatus,
    QueryAnswer,
    QueryTemplate,
    RunStatus,
    SourceDocument,
    StoryRun,
    utcnow,
)
from biography_rag.embeddings.registry import EmbedderRegistry
from biography_rag.ingest.chunker import Chunker
from biography_rag.ingest.loaders import LOADERS, extract_text
from biography_rag.llm.base import ChatModelSource
from biography_rag.rag.generation import QueryAnswerer, StoryWeaver
from biography_rag.rag.queries import QUERIES
from biography_rag.rag.retriever import HybridRetriever, RetrievalConfig
from biography_rag.rag.verification import JudgedVerifier, LexicalVerifier, Verifier
from biography_rag.runlog import RunLogger
from biography_rag.serialization import answer_to_dict, report_to_dict
from biography_rag.store.base import ChunkIndex, CorpusRepository, RunRepository

log = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Bad input from the user; the web layer maps it to HTTP 400."""


class NotFound(LookupError):
    pass


@dataclass(frozen=True)
class UploadedFile:
    name: str
    data: bytes


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:32] or "person"


def _new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:10]}"


class IngestService:
    def __init__(
        self,
        corpora: CorpusRepository,
        index: ChunkIndex,
        embedders: EmbedderRegistry,
        chunker: Chunker,
        embed_batch: int = 256,
    ) -> None:
        self._corpora = corpora
        self._index = index
        self._embedders = embedders
        self._chunker = chunker
        self._batch = embed_batch

    def create(self, person: str, files: Sequence[UploadedFile], embedder: str | None = None) -> Corpus:
        person = " ".join(person.split())
        if not person or len(person) > 120:
            raise ValidationError("give the person's name (1-120 characters)")
        if not files:
            raise ValidationError("upload at least one file")
        bad = [f.name for f in files if PurePath(f.name).suffix.lower() not in LOADERS]
        if bad:
            raise ValidationError(f"unsupported file type: {', '.join(bad[:5])} (use {', '.join(sorted(LOADERS))})")
        try:
            chosen = self._embedders.by_name(embedder)
        except (ValueError, RuntimeError) as exc:
            raise ValidationError(str(exc)) from exc
        corpus = Corpus(
            corpus_id=f"{_slug(person)}-{_new_id()[:6]}",
            person=person,
            embedder_id=chosen.embedder_id,
            documents=[f.name for f in files],
        )
        self._corpora.save(corpus)
        return corpus

    def ingest(self, corpus_id: str, files: Sequence[UploadedFile]) -> Corpus:
        corpus = self._corpora.get(corpus_id)
        if corpus is None:
            raise NotFound(corpus_id)
        try:
            chunks: list[Chunk] = []
            for n, file in enumerate(files, 1):
                text = extract_text(file.name, file.data)
                if text.strip():
                    chunks.extend(self._chunker.chunk(SourceDocument(f"d{n:02d}", file.name, text)))
            if not chunks:
                raise ValidationError("no text could be extracted from the uploaded files")
            embedder = self._embedders.by_id(corpus.embedder_id)
            for start in range(0, len(chunks), self._batch):
                batch = chunks[start : start + self._batch]
                self._index.add(corpus_id, batch, embedder.embed_documents([c.text for c in batch]))
            corpus = replace(corpus, status=CorpusStatus.READY, chunk_count=len(chunks), error=None)
        except Exception as exc:
            log.exception("ingest failed for %s", corpus_id)
            self._index.delete_corpus(corpus_id)
            corpus = replace(corpus, status=CorpusStatus.FAILED, chunk_count=0, error=str(exc)[:500])
        self._corpora.save(corpus)
        return corpus

    def delete(self, corpus_id: str) -> None:
        if self._corpora.get(corpus_id) is None:
            raise NotFound(corpus_id)
        self._index.delete_corpus(corpus_id)
        self._corpora.delete(corpus_id)


@dataclass(frozen=True)
class StoryConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    concurrency: int = 4
    verify_with_llm: bool = True
    max_compare_targets: int = 6


@dataclass(frozen=True)
class Target:
    """One model to run: a provider and (optionally) one of that provider's listed models."""

    provider: str | None = None
    model: str | None = None


class StoryService:
    def __init__(
        self,
        corpora: CorpusRepository,
        index: ChunkIndex,
        runs: RunRepository,
        embedders: EmbedderRegistry,
        chats: ChatModelSource,
        logger: RunLogger,
        config: StoryConfig = StoryConfig(),
        queries: Sequence[QueryTemplate] = QUERIES,
    ) -> None:
        self._corpora = corpora
        self._index = index
        self._runs = runs
        self._embedders = embedders
        self._chats = chats
        self._log = logger
        self._config = config
        self._queries = list(queries)

    # --- helpers -----------------------------------------------------------
    def _ready_corpus(self, corpus_id: str) -> Corpus:
        corpus = self._corpora.get(corpus_id)
        if corpus is None:
            raise NotFound(corpus_id)
        if corpus.status is not CorpusStatus.READY:
            raise ValidationError(f"corpus is {corpus.status.value}, not ready")
        return corpus

    def _check_target(self, target: Target) -> tuple[str, str]:
        try:
            chat = self._chats.create(target.provider, target.model)
        except (ValueError, RuntimeError) as exc:
            raise ValidationError(str(exc)) from exc
        return chat.provider, chat.model

    def _new_run(
        self,
        corpus: Corpus,
        target: Target,
        kind: str,
        total: int,
        *,
        question: str | None = None,
        comparison_id: str | None = None,
    ) -> StoryRun:
        provider, model = self._check_target(target)
        run = StoryRun(
            run_id=_new_id("r"),
            corpus_id=corpus.corpus_id,
            person=corpus.person,
            provider=provider,
            model=model,
            kind=kind,
            total=total,
            question=question,
            comparison_id=comparison_id,
        )
        self._runs.save(run)
        return run

    @staticmethod
    def _clean_question(question: str) -> str:
        question = " ".join(question.split())
        if not question or len(question) > 500:
            raise ValidationError("ask a question of 1-500 characters")
        return question

    def _verifier(self, provider: str, model: str) -> Verifier:
        lexical = LexicalVerifier()
        if not self._config.verify_with_llm:
            return lexical
        return JudgedVerifier(lexical, self._chats.create(provider, model))

    def _answer_one(
        self,
        run: StoryRun,
        retriever: HybridRetriever,
        answerer: QueryAnswerer,
        query_id: int,
        question: str,
        keywords: str,
    ) -> tuple[QueryAnswer, list[Chunk]]:
        retrieved = retriever.retrieve(run.corpus_id, question, keywords)
        chunks = [r.chunk for r in retrieved]
        answer, exchange = answerer.answer(query_id, run.person, question, chunks)
        lexical = LexicalVerifier().verify(
            answer.answer or "", {c.chunk_id: c for c in chunks}, set(answer.retrieved_ids)
        )
        self._log.log(
            run.run_id,
            "query",
            query_id=query_id,
            question=question,
            keyword_hints=keywords,
            retrieved=[
                {
                    "chunk_id": r.chunk.chunk_id,
                    "rrf": round(r.score, 5),
                    "cosine": None if r.cosine is None else round(r.cosine, 4),
                    "vector_rank": r.vector_rank,
                    "keyword_rank": r.keyword_rank,
                }
                for r in retrieved
            ],
            llm_called=bool(exchange.messages),
            prompt=exchange.messages,
            response=exchange.response,
            evidence=list(exchange.evidence),
            gate=exchange.gate_note,
            evidence_check={"prompt": exchange.check_messages, "response": exchange.check_response}
            if exchange.check_messages
            else None,
            answer=answer_to_dict(answer),
            answer_check=report_to_dict(lexical.report),
        )
        return answer, chunks

    def _context(self, run: StoryRun, corpus: Corpus) -> tuple[HybridRetriever, QueryAnswerer]:
        embedder = self._embedders.by_id(corpus.embedder_id)
        retriever = HybridRetriever(self._index, embedder, self._config.retrieval)
        return retriever, QueryAnswerer(self._chats.create(run.provider, run.model))

    def _log_start(self, run: StoryRun, corpus: Corpus) -> None:
        r = self._config.retrieval
        self._log.log(
            run.run_id,
            "run_started",
            kind=run.kind,
            question=run.question,
            comparison_id=run.comparison_id,
            corpus_id=corpus.corpus_id,
            person=corpus.person,
            embedder=corpus.embedder_id,
            chunk_count=corpus.chunk_count,
            provider=run.provider,
            model=run.model,
            retrieval={
                "top_k": r.top_k,
                "candidate_k": r.candidate_k,
                "min_similarity": r.min_similarity,
                "max_number_density": r.max_number_density,
            },
            verify_with_llm=self._config.verify_with_llm,
        )

    # --- starting runs -------------------------------------------------------
    def start(self, corpus_id: str, provider: str | None = None, model: str | None = None) -> StoryRun:
        """Queue the 20-question story; `execute` does the work."""
        return self._new_run(self._ready_corpus(corpus_id), Target(provider, model), "story", len(self._queries))

    def start_ask(
        self, corpus_id: str, question: str, provider: str | None = None, model: str | None = None
    ) -> StoryRun:
        """Queue one ad-hoc question (the canary test)."""
        question = self._clean_question(question)
        return self._new_run(self._ready_corpus(corpus_id), Target(provider, model), "ask", 1, question=question)

    def start_many(
        self, corpus_id: str, kind: str, targets: Sequence[Target], question: str | None = None
    ) -> list[StoryRun]:
        """Queue the same story or question once per model, for side-by-side comparison.
        Every target is checked before any run is created, so a bad one queues nothing."""
        if kind not in ("story", "ask"):
            raise ValidationError("kind must be 'story' or 'ask'")
        limit = self._config.max_compare_targets
        if not 1 <= len(targets) <= limit:
            raise ValidationError(f"pick 1 to {limit} models")
        corpus = self._ready_corpus(corpus_id)
        question = self._clean_question(question or "") if kind == "ask" else None
        resolved = [self._check_target(t) for t in targets]
        if len(set(resolved)) != len(resolved):
            raise ValidationError("each model may be picked only once")
        comparison_id = _new_id("c") if len(targets) > 1 else None
        total = 1 if kind == "ask" else len(self._queries)
        return [
            self._new_run(corpus, Target(p, m), kind, total, question=question, comparison_id=comparison_id)
            for p, m in resolved
        ]

    def comparison(self, corpus_id: str, comparison_id: str) -> list[StoryRun]:
        runs = [r for r in self._runs.list_for_corpus(corpus_id) if r.comparison_id == comparison_id]
        if not runs:
            raise NotFound(comparison_id)
        return sorted(runs, key=lambda r: (r.created_at, r.run_id))

    # --- running -------------------------------------------------------------
    def execute(self, run_id: str) -> StoryRun:
        run = self._runs.get(run_id)
        if run is None:
            raise NotFound(run_id)
        corpus = self._ready_corpus(run.corpus_id)
        run.status = RunStatus.RUNNING
        self._runs.save(run)
        self._log_start(run, corpus)
        try:
            if run.kind == "ask":
                self._execute_ask(run, corpus)
            else:
                self._execute_story(run, corpus)
            run.status = RunStatus.DONE
        except Exception as exc:
            log.exception("run %s failed", run_id)
            run.status, run.error = RunStatus.FAILED, str(exc)[:500]
            self._log.log(run.run_id, "run_failed", error=run.error)
        run.finished_at = utcnow()
        self._runs.save(run)
        self._log.log(run.run_id, "run_finished", status=run.status.value)
        return run

    def _log_verification(self, run: StoryRun, result) -> None:
        self._log.log(
            run.run_id,
            "verification",
            report=report_to_dict(result.report),
            judge_prompt=result.judge_messages,
            judge_response=result.judge_response,
        )

    def _execute_story(self, run: StoryRun, corpus: Corpus) -> None:
        retriever, answerer = self._context(run, corpus)
        lock = threading.Lock()
        results: list[tuple[QueryAnswer, list[Chunk]]] = []

        def work(q: QueryTemplate) -> None:
            result = self._answer_one(run, retriever, answerer, q.query_id, q.render(run.person), q.keywords)
            with lock:
                results.append(result)
                run.progress = len(results)
                self._runs.save(run)

        with ThreadPoolExecutor(max_workers=self._config.concurrency) as pool:
            list(pool.map(work, self._queries))  # list() re-raises the first failure

        results.sort(key=lambda r: r[0].query_id)
        run.answers = [a for a, _ in results]
        retrieved = {c.chunk_id: c for _, chunks in results for c in chunks}
        # The weave may only see chunks that an answer actually cited from its own retrieval.
        cited = sorted({cid for a in run.answers for cid in a.cited_ids if cid in a.retrieved_ids})
        story, exchange = StoryWeaver(self._chats.create(run.provider, run.model)).weave(
            run.person, run.answers, [retrieved[c] for c in cited]
        )
        self._log.log(run.run_id, "weave", prompt=exchange.messages, response=exchange.response)

        result = self._verifier(run.provider, run.model).verify(story, retrieved, set(retrieved))
        self._log_verification(run, result)
        run.story, run.verification = story, result.report

    def _execute_ask(self, run: StoryRun, corpus: Corpus) -> None:
        retriever, answerer = self._context(run, corpus)
        answer, chunks = self._answer_one(run, retriever, answerer, 0, run.question or "", "")
        chunk_map = {c.chunk_id: c for c in chunks}
        result = self._verifier(run.provider, run.model).verify(answer.answer or "", chunk_map, set(chunk_map))
        self._log_verification(run, result)
        run.answers, run.story, run.verification, run.progress = [answer], answer.display(), result.report, 1

    def ask(self, corpus_id: str, question: str, provider: str | None = None, model: str | None = None) -> StoryRun:
        """Ask one question and wait for the answer."""
        return self.execute(self.start_ask(corpus_id, question, provider, model).run_id)

    # --- startup housekeeping -------------------------------------------
    def recover_interrupted(self) -> int:
        """A restart kills in-flight threads; mark their records failed instead of 'running' forever."""
        count = 0
        for corpus in self._corpora.list():
            if corpus.status is CorpusStatus.INGESTING:
                self._index.delete_corpus(corpus.corpus_id)
                self._corpora.save(replace(corpus, status=CorpusStatus.FAILED, error="interrupted by a server restart"))
                count += 1
            for run in self._runs.list_for_corpus(corpus.corpus_id):
                if run.status in (RunStatus.QUEUED, RunStatus.RUNNING):
                    run.status, run.error, run.finished_at = (
                        RunStatus.FAILED,
                        "interrupted by a server restart",
                        utcnow(),
                    )
                    self._runs.save(run)
                    count += 1
        return count
