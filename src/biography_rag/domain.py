"""Core domain objects. Plain, immutable data with no I/O, no frameworks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class SourceDocument:
    """One uploaded file, already reduced to clean text."""

    doc_id: str  # short, stable within a corpus: "d01", "d02", ...
    title: str  # usually the original filename
    text: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable slice of a document. ``chunk_id`` is what citations point at."""

    chunk_id: str  # "d01-0042"
    doc_id: str
    ordinal: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float  # fused rank score (higher is better)
    cosine: float | None = None  # vector similarity, when the vector arm found it
    vector_rank: int | None = None
    keyword_rank: int | None = None


class CorpusStatus(StrEnum):
    INGESTING = "ingesting"
    READY = "ready"
    FAILED = "failed"


@dataclass
class Corpus:
    corpus_id: str
    person: str
    embedder_id: str
    status: CorpusStatus = CorpusStatus.INGESTING
    documents: list[str] = field(default_factory=list)  # titles
    chunk_count: int = 0
    error: str | None = None
    created_at: datetime = field(default_factory=utcnow)


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class QueryTemplate:
    query_id: int
    template: str  # contains "{person}"
    keywords: str = ""  # generic retrieval hints, never person-specific facts

    def render(self, person: str) -> str:
        return self.template.format(person=person)


NOT_IN_SOURCES = "Not in the sources."


@dataclass(frozen=True)
class QueryAnswer:
    query_id: int
    question: str
    retrieved_ids: list[str]
    answer: str | None  # None means "not in the sources"
    cited_ids: list[str]

    @property
    def found(self) -> bool:
        return self.answer is not None

    def display(self) -> str:
        return self.answer if self.answer is not None else NOT_IN_SOURCES


class Verdict(StrEnum):
    SUPPORTED = "supported"
    WEAK = "weak"  # cited, but much of the wording/facts are absent from the cited chunks
    UNCITED = "uncited"  # no chunk id at all: candidate training-data leak
    BAD_CITATION = "bad_citation"  # cites an id that was never retrieved
    JUDGE_FLAGGED = "judge_flagged"  # the LLM judge says the cited chunks do not support it


@dataclass(frozen=True)
class SentenceCheck:
    sentence: str
    cited_ids: list[str]
    verdict: Verdict
    support: float  # 0..1 share of content words found in the cited chunks
    missing_terms: list[str] = field(default_factory=list)  # numbers / names absent from chunks
    note: str = ""

    @property
    def flagged(self) -> bool:
        return self.verdict is not Verdict.SUPPORTED


@dataclass(frozen=True)
class VerificationReport:
    checks: list[SentenceCheck]

    @property
    def flagged(self) -> list[SentenceCheck]:
        return [c for c in self.checks if c.flagged]

    @property
    def traced_ratio(self) -> float:
        if not self.checks:
            return 1.0
        return sum(not c.flagged for c in self.checks) / len(self.checks)


@dataclass
class StoryRun:
    run_id: str
    corpus_id: str
    person: str
    provider: str
    model: str
    status: RunStatus = RunStatus.QUEUED
    progress: int = 0  # queries answered so far
    total: int = 0
    answers: list[QueryAnswer] = field(default_factory=list)
    story: str = ""
    verification: VerificationReport | None = None
    error: str | None = None
    kind: str = "story"  # "story" or "ask"
    question: str | None = None  # the question, for kind "ask"
    comparison_id: str | None = None  # runs started together to compare models share this
    created_at: datetime = field(default_factory=utcnow)
    finished_at: datetime | None = None
