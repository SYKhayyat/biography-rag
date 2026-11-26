"""Core domain objects. Plain, immutable data with no I/O, no frameworks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
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
    """A retrievable slice of a document. ``chunk_id`` is what citations point at.

    ``context`` says where the slice comes from (book title and, when one was found, the chapter
    heading above it). It is prefixed to the text for embedding and shown next to the snippet in
    the answer prompt, so the model can tell a son from a grandson and one book from another. It
    is *not* part of ``text``: quotes are still checked against the passage itself."""

    chunk_id: str  # "d01-0042"
    doc_id: str
    ordinal: int
    text: str
    context: str = ""  # e.g. "Familiar Letters — Chapter III: Paris"

    @property
    def embed_input(self) -> str:
        """What the embedder sees: the context line, then the passage."""
        return f"{self.context}\n\n{self.text}" if self.context else self.text


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float  # fused rank score (higher is better)
    cosine: float | None = None  # vector similarity, when the vector arm found it
    vector_rank: int | None = None
    keyword_rank: int | None = None


def merge_windows(selected: Sequence[Chunk], radius: int, texts: Mapping[str, str]) -> list[Chunk]:
    """Snippets to answer from: each selected chunk widened by ``radius`` neighbours on both sides.

    ``texts`` maps a chunk id to its own widened text (supplied by the index, which is the only
    thing that can read neighbours). Selection is swept in reading order per document, so two
    neighbours that were both selected collapse into one window instead of the same paragraph
    appearing twice in the prompt. Chunks that collapse lose their id: the model never saw it, so
    it cannot cite it, and nothing it would have quoted is gone (it is inside the kept window).
    """
    out: list[Chunk] = []
    covered: dict[str, int] = {}  # doc_id -> highest ordinal already inside a kept window
    for chunk in sorted(selected, key=lambda c: (c.doc_id, c.ordinal)):
        if chunk.ordinal <= covered.get(chunk.doc_id, -1):
            continue
        covered[chunk.doc_id] = chunk.ordinal + radius
        widened = texts.get(chunk.chunk_id)
        out.append(replace(chunk, text=widened) if widened else chunk)
    return out


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
    # Who wrote the pages: "auto" (detect per document), "self" (the person: a memoir or
    # autobiography) or "others" (biographies). Detection fills ``first_person_docs``.
    narration: str = "auto"
    first_person_docs: list[str] = field(default_factory=list)  # doc ids narrated by the person as "I"


NARRATIONS = ("auto", "self", "others")


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


class AnswerReason(StrEnum):
    """Why a question got no answer. Only the first four mean the sources lack it; the rest
    are failures that must not be shown as "not in the sources"."""

    NOTHING_RETRIEVED = "nothing_retrieved"
    NOT_IN_SOURCES = "not_in_sources"  # the model said so
    NO_EVIDENCE = "no_evidence"  # no quote survived the verbatim check
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"  # the quotes don't answer the question
    EMPTY_OUTPUT = "empty_output"
    TRUNCATED = "truncated"  # the reply hit the token limit
    ERROR = "error"  # the provider call (or retrieval) failed


REASON_TEXT: dict[str, str] = {
    AnswerReason.NOTHING_RETRIEVED: "nothing relevant was retrieved",
    AnswerReason.NOT_IN_SOURCES: "the model found no answer in the passages",
    AnswerReason.NO_EVIDENCE: "the model's quotes were not found in the passages",
    AnswerReason.EVIDENCE_INSUFFICIENT: "the quoted passages do not answer the question",
    AnswerReason.EMPTY_OUTPUT: "the model returned an empty reply",
    AnswerReason.TRUNCATED: "the model's reply was cut off at the length limit",
    AnswerReason.ERROR: "the model call failed",
}
_FAILURES = {AnswerReason.EMPTY_OUTPUT, AnswerReason.TRUNCATED, AnswerReason.ERROR}


@dataclass(frozen=True)
class QueryAnswer:
    query_id: int
    question: str
    retrieved_ids: list[str]
    answer: str | None  # None means no answer; ``reason`` says why
    cited_ids: list[str]
    reason: str | None = None  # an AnswerReason when ``answer`` is None
    answered_by: str | None = None  # the model that actually replied (a fallback may stand in)

    @property
    def found(self) -> bool:
        return self.answer is not None

    @property
    def failed(self) -> bool:
        """No answer because something broke, not because the sources lack one."""
        return self.answer is None and self.reason in _FAILURES

    def display(self) -> str:
        if self.answer is not None:
            return self.answer
        if self.failed:
            return f"No answer: {REASON_TEXT[self.reason]}."  # type: ignore[index]
        return NOT_IN_SOURCES


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
