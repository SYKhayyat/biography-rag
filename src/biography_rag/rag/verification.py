"""Trace every sentence to the chunks it cites, and flag what looks like it came from the
model's training data instead.

Two layers (the second decorates the first):

* ``LexicalVerifier`` - deterministic. A sentence with no citation, or a citation to a chunk
  that was never retrieved, is flagged outright. Otherwise it measures how many of the
  sentence's content words appear in the cited chunks, and lists any number or proper noun
  that does not appear there at all: the classic signature of a model filling a gap from
  memory ("born on January 17, 1706" when the chunk only says "born in Boston").
* ``JudgedVerifier`` - asks an LLM to audit each sentence against its cited chunks, which
  catches paraphrased hallucinations the word overlap cannot. Citation errors stay flagged
  regardless of what the judge says.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from typing import Protocol

from biography_rag.domain import Chunk, SentenceCheck, Verdict, VerificationReport
from biography_rag.llm.base import ChatModel, Message
from biography_rag.rag.citations import cited_ids, split_sentences, strip_citations
from biography_rag.rag.prompts import judge_messages
from biography_rag.text import STOPWORDS, content_terms, words


@dataclass(frozen=True)
class VerificationResult:
    report: VerificationReport
    judge_messages: list[Message]
    judge_response: str | None


class Verifier(Protocol):
    def verify(self, text: str, chunks: Mapping[str, Chunk], allowed_ids: AbstractSet[str]) -> VerificationResult: ...


_CAPITALISED = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b")
_NUMBER = re.compile(r"\b\d[\d,]*\b")


class LexicalVerifier:
    def __init__(self, min_support: float = 0.5) -> None:
        self.min_support = min_support

    def check(self, sentence: str, chunks: Mapping[str, Chunk], allowed_ids: AbstractSet[str]) -> SentenceCheck:
        ids = cited_ids(sentence)
        if not ids:
            return SentenceCheck(sentence, [], Verdict.UNCITED, 0.0, note="no chunk id cited")
        unknown = [i for i in ids if i not in allowed_ids or i not in chunks]
        if unknown:
            return SentenceCheck(
                sentence, ids, Verdict.BAD_CITATION, 0.0, note=f"cites unretrieved chunk(s): {', '.join(unknown)}"
            )
        source = " ".join(chunks[i].text for i in ids)
        claim = strip_citations(sentence)

        source_terms = set(content_terms(source))
        terms = content_terms(claim)
        support = (sum(t in source_terms for t in terms) / len(terms)) if terms else 1.0

        source_words = set(words(source))
        source_digits = {n.replace(",", "") for n in _NUMBER.findall(source)}
        missing = [n for n in _NUMBER.findall(claim) if n.replace(",", "") not in source_digits]
        missing += [
            w
            for w in _CAPITALISED.findall(claim)
            if w.lower() not in source_words
            and w.lower() not in STOPWORDS
            and w.lower().removesuffix("s") not in source_words
        ]
        missing = list(dict.fromkeys(missing))
        if support < self.min_support or missing:
            note = f"{support:.0%} of content words found in cited chunks"
            if missing:
                note += f"; not in cited chunks: {', '.join(missing)}"
            return SentenceCheck(sentence, ids, Verdict.WEAK, support, missing, note)
        return SentenceCheck(sentence, ids, Verdict.SUPPORTED, support)

    def verify(self, text: str, chunks: Mapping[str, Chunk], allowed_ids: AbstractSet[str]) -> VerificationResult:
        checks = [self.check(s, chunks, allowed_ids) for s in split_sentences(text)]
        return VerificationResult(VerificationReport(checks), [], None)


def _parse_judgement(response: str) -> dict[int, tuple[bool, str]]:
    start, end = response.find("["), response.rfind("]")
    if start == -1 or end <= start:
        return {}
    try:
        items = json.loads(response[start : end + 1])
    except json.JSONDecodeError:
        return {}
    out: dict[int, tuple[bool, str]] = {}
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and isinstance(item.get("i"), int) and isinstance(item.get("supported"), bool):
            out[item["i"]] = (item["supported"], str(item.get("reason", "")))
    return out


class JudgedVerifier:
    def __init__(self, inner: LexicalVerifier, chat: ChatModel) -> None:
        self._inner = inner
        self._chat = chat

    def verify(self, text: str, chunks: Mapping[str, Chunk], allowed_ids: AbstractSet[str]) -> VerificationResult:
        base = self._inner.verify(text, chunks, allowed_ids).report
        if not base.checks:
            return VerificationResult(base, [], None)
        used = sorted({i for c in base.checks for i in c.cited_ids if i in chunks})
        messages = judge_messages([(c.sentence, c.cited_ids) for c in base.checks], [chunks[i] for i in used])
        response = self._chat.complete(messages, max_tokens=4000)
        verdicts = _parse_judgement(response)

        checks: list[SentenceCheck] = []
        for n, check in enumerate(base.checks, 1):
            if n not in verdicts or check.verdict in (Verdict.UNCITED, Verdict.BAD_CITATION):
                checks.append(check)
                continue
            supported, reason = verdicts[n]
            if not supported:
                note = "; ".join(filter(None, [f"judge: {reason or 'not supported'}", check.note]))
                checks.append(replace(check, verdict=Verdict.JUDGE_FLAGGED, note=note))
            elif check.verdict is Verdict.WEAK:
                checks.append(
                    replace(check, verdict=Verdict.SUPPORTED, note=f"judge: supported (lexical: {check.note})")
                )
            else:
                checks.append(check)
        return VerificationResult(VerificationReport(checks), messages, response)


def merge_chunks(groups: Sequence[Sequence[Chunk]]) -> dict[str, Chunk]:
    return {c.chunk_id: c for group in groups for c in group}
