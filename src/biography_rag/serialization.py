"""JSON round-tripping for domain objects (used by the Postgres store and the API)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from biography_rag.domain import (
    Corpus,
    CorpusStatus,
    QueryAnswer,
    RunStatus,
    SentenceCheck,
    StoryRun,
    Verdict,
    VerificationReport,
)


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def corpus_to_dict(c: Corpus) -> dict[str, Any]:
    return {
        "corpus_id": c.corpus_id,
        "person": c.person,
        "embedder_id": c.embedder_id,
        "status": c.status.value,
        "documents": c.documents,
        "chunk_count": c.chunk_count,
        "error": c.error,
        "created_at": _dt(c.created_at),
    }


def corpus_from_dict(d: dict[str, Any]) -> Corpus:
    return Corpus(
        corpus_id=d["corpus_id"],
        person=d["person"],
        embedder_id=d["embedder_id"],
        status=CorpusStatus(d["status"]),
        documents=list(d.get("documents") or []),
        chunk_count=int(d.get("chunk_count") or 0),
        error=d.get("error"),
        created_at=_parse_dt(d.get("created_at")) or datetime.now().astimezone(),
    )


def answer_to_dict(a: QueryAnswer) -> dict[str, Any]:
    return {
        "query_id": a.query_id,
        "question": a.question,
        "retrieved_ids": a.retrieved_ids,
        "answer": a.answer,
        "cited_ids": a.cited_ids,
        "display": a.display(),
    }


def answer_from_dict(d: dict[str, Any]) -> QueryAnswer:
    return QueryAnswer(d["query_id"], d["question"], list(d["retrieved_ids"]), d.get("answer"), list(d["cited_ids"]))


def check_to_dict(c: SentenceCheck) -> dict[str, Any]:
    return {
        "sentence": c.sentence,
        "cited_ids": c.cited_ids,
        "verdict": c.verdict.value,
        "support": round(c.support, 3),
        "missing_terms": c.missing_terms,
        "note": c.note,
        "flagged": c.flagged,
    }


def check_from_dict(d: dict[str, Any]) -> SentenceCheck:
    return SentenceCheck(
        d["sentence"], list(d["cited_ids"]), Verdict(d["verdict"]), d["support"], list(d["missing_terms"]), d["note"]
    )


def report_to_dict(r: VerificationReport) -> dict[str, Any]:
    return {
        "checks": [check_to_dict(c) for c in r.checks],
        "traced_ratio": round(r.traced_ratio, 3),
        "flagged_count": len(r.flagged),
        "sentence_count": len(r.checks),
    }


def report_from_dict(d: dict[str, Any]) -> VerificationReport:
    return VerificationReport([check_from_dict(c) for c in d["checks"]])


def run_to_dict(r: StoryRun) -> dict[str, Any]:
    return {
        "run_id": r.run_id,
        "corpus_id": r.corpus_id,
        "person": r.person,
        "provider": r.provider,
        "model": r.model,
        "status": r.status.value,
        "progress": r.progress,
        "total": r.total,
        "answers": [answer_to_dict(a) for a in r.answers],
        "story": r.story,
        "verification": report_to_dict(r.verification) if r.verification else None,
        "error": r.error,
        "kind": r.kind,
        "question": r.question,
        "comparison_id": r.comparison_id,
        "created_at": _dt(r.created_at),
        "finished_at": _dt(r.finished_at),
    }


def run_from_dict(d: dict[str, Any]) -> StoryRun:
    return StoryRun(
        run_id=d["run_id"],
        corpus_id=d["corpus_id"],
        person=d["person"],
        provider=d["provider"],
        model=d["model"],
        status=RunStatus(d["status"]),
        progress=d.get("progress", 0),
        total=d.get("total", 0),
        answers=[answer_from_dict(a) for a in d.get("answers", [])],
        story=d.get("story", ""),
        verification=report_from_dict(d["verification"]) if d.get("verification") else None,
        error=d.get("error"),
        kind=d.get("kind", "story"),
        question=d.get("question"),
        comparison_id=d.get("comparison_id"),
        created_at=_parse_dt(d.get("created_at")) or datetime.now().astimezone(),
        finished_at=_parse_dt(d.get("finished_at")),
    )
