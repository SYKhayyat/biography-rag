"""LLM steps: answer one question from its snippets, then weave answers into a story."""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace

from biography_rag.domain import AnswerReason, Chunk, QueryAnswer
from biography_rag.llm.base import ChatModel, Message, reply_model, reply_truncated
from biography_rag.rag.citations import cited_ids, fill_paragraph_citations
from biography_rag.rag.grounding import gate
from biography_rag.rag.prompts import NOT_IN_SOURCES_TOKEN, answer_messages, evidence_check_messages, weave_messages


@dataclass(frozen=True)
class LlmExchange:
    """What was sent and what came back; ``messages`` is empty when the LLM was not called."""

    messages: list[Message]
    response: str | None
    evidence: tuple[dict, ...] = ()  # the quotes the model gave and whether each was found
    gate_note: str | None = None  # why the evidence gate rejected or trimmed the reply
    check_messages: list[Message] = field(default_factory=list)  # the evidence-sufficiency prompt, if sent
    check_response: str | None = None


# Room for a reasoning model's hidden thinking before the one-word verdict: those tokens
# count against max_tokens, and with a tiny budget the visible reply comes back empty.
CHECK_MAX_TOKENS = 400
_FIRST_WORD = re.compile(r"[A-Za-z]+")


def says_yes(verdict: str) -> bool:
    """The check passes only on an explicit YES as the first word; anything else (NO, empty,
    rambling) fails closed."""
    word = _FIRST_WORD.search(verdict)
    return word is not None and word.group().upper() == "YES"


def _is_not_in_sources(response: str) -> bool:
    text = response.strip().strip(".").strip()
    upper = text.upper().replace(" ", "_")
    return upper.startswith(NOT_IN_SOURCES_TOKEN) or (NOT_IN_SOURCES_TOKEN in upper and not cited_ids(text))


class QueryAnswerer:
    """Answers one question, then gates the reply in two steps:

    1. ``grounding.gate``: each EVIDENCE quote must occur verbatim in the snippet it cites;
    2. the sufficiency check: the model is shown only the question and the verified quotes, and
       must agree they state the answer. A real quote from a loosely related snippet (the
       "his portrait hangs in Boston" quote offered for "whose face is on the $100 bill?")
       passes step 1 but fails step 2.
    """

    def __init__(
        self, chat: ChatModel, *, check_evidence: bool = True, first_person_docs: Collection[str] = ()
    ) -> None:
        self._chat = chat
        self._check_evidence = check_evidence
        self.first_person_docs = frozenset(first_person_docs)  # doc ids the person narrates as "I"

    def answer(
        self, query_id: int, person: str, question: str, chunks: Sequence[Chunk]
    ) -> tuple[QueryAnswer, LlmExchange]:
        ids = [c.chunk_id for c in chunks]

        def none(reason: AnswerReason, by: str | None = None) -> QueryAnswer:
            return QueryAnswer(query_id, question, ids, None, [], reason=reason.value, answered_by=by)

        if not chunks:  # nothing retrieved: never ask the model, it could only guess
            return none(AnswerReason.NOTHING_RETRIEVED), LlmExchange([], None)
        messages = answer_messages(person, question, chunks, self.first_person_docs)
        response = self._chat.complete(messages, max_tokens=1200)
        by = reply_model(response, self._chat.model)
        if not response.strip():
            return none(AnswerReason.EMPTY_OUTPUT, by), LlmExchange(messages, response)
        if _is_not_in_sources(response):
            return none(AnswerReason.NOT_IN_SOURCES, by), LlmExchange(messages, response)
        result = gate(response, {c.chunk_id: c for c in chunks})
        exchange = LlmExchange(messages, response, tuple(result.evidence), result.reason)
        if result.answer is None:
            # A reply cut off mid-way rarely parses; say so rather than blame the evidence.
            reason = AnswerReason.TRUNCATED if reply_truncated(response) else AnswerReason.NO_EVIDENCE
            return none(reason, by), exchange
        found_answer = QueryAnswer(query_id, question, ids, result.answer, result.cited_ids, answered_by=by)
        if not self._check_evidence:
            return found_answer, exchange

        found = [e for e in result.evidence if e["found"]]
        quotes = list(dict.fromkeys(e["quote"] for e in found))
        by_id = {c.chunk_id: c for c in chunks}
        own = {e["quote"] for e in found if by_id[e["chunk_id"]].doc_id in self.first_person_docs}
        check = evidence_check_messages(person, question, quotes, own)
        verdicts = [self._chat.complete(check, max_tokens=CHECK_MAX_TOKENS)]
        if not says_yes(verdicts[0]):
            # One-word verdicts on borderline evidence flip between calls; a second NO is needed
            # to discard an answer whose quotes already passed the verbatim check.
            verdicts.append(self._chat.complete(check, max_tokens=CHECK_MAX_TOKENS))
        exchange = replace(exchange, check_messages=check, check_response="\n---\n".join(verdicts))
        if not says_yes(verdicts[-1]):
            note = "; ".join(filter(None, ["the quoted evidence does not answer the question", result.reason]))
            return none(AnswerReason.EVIDENCE_INSUFFICIENT, by), replace(exchange, gate_note=note)
        return found_answer, exchange


class StoryWeaver:
    def __init__(self, chat: ChatModel) -> None:
        self._chat = chat

    def weave(self, person: str, answers: Sequence[QueryAnswer], cited: Sequence[Chunk]) -> tuple[str, LlmExchange]:
        if not any(a.found for a in answers):
            return "", LlmExchange([], None)
        messages = weave_messages(person, answers, cited)
        raw = self._chat.complete(messages, max_tokens=3500)
        # A sentence cited only by its paragraph's closing citation is checked against that snippet
        # by the verifier, instead of being flagged as uncited; the log keeps the raw reply.
        return fill_paragraph_citations(raw), LlmExchange(messages, raw)
