"""Core domain objects. Plain, immutable data with no I/O, no frameworks."""

from __future__ import annotations

import re
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


SEAM_BEFORE = "[...before]"

# A passage that opens with one of these was cut away from what they refer to.
_OPENS_WITH_REFERENCE = re.compile(
    r"^(he|she|it|they|his|her|its|their|him|them|this|that|these|those|such|the same|there|then|"
    r"afterward|afterwards|meanwhile|still|yet)\b",
    re.I,
)
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"'\u201d\u2019)\]]*\s+")  # keeps the punctuation


def opens_with_reference(text: str) -> bool:
    """True when the passage starts with a word that points back ('He...', 'This...')."""
    return bool(_OPENS_WITH_REFERENCE.match(text.lstrip()[:40]))


def _trailing_sentences(text: str, limit: int) -> str:
    """The last whole sentences of ``text``, about ``limit`` characters (always at least one).
    Cutting at a character offset leaves fragments like "e sixteen sons and daughters", which read
    as corruption."""
    if limit <= 0:
        return ""
    sentences = [p.strip() for p in _SENTENCE_END.split(text.strip()) if p.strip()]
    if not sentences:
        return ""
    kept: list[str] = []
    for sentence in reversed(sentences):
        if kept and len(sentence) + len(" ".join(kept)) > limit:
            break
        kept.insert(0, sentence)
    return " ".join(kept)


def with_seam(chunk: Chunk, before: str, seam: int) -> Chunk:
    """``chunk`` prefixed with the tail of the passage before it — but only when it needs it.

    A chunk that starts "He was then bound apprentice..." is unreadable and unquotable without the
    sentence before it, which is the whole reason to look at a neighbour. A chunk that starts a
    sentence of its own needs nothing, and measured on the Franklin corpus, adding a seam to every
    passage only lengthened the prompt. Whole sentences, never a character offset.

    The marker is part of the snippet text, so a quote that happens to include it is still found.
    """
    if seam <= 0 or not before.strip() or not opens_with_reference(chunk.text):
        return chunk
    tail = _trailing_sentences(before, seam)
    return replace(chunk, text=f"{SEAM_BEFORE} {tail}\n\n{chunk.text}") if tail else chunk


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
