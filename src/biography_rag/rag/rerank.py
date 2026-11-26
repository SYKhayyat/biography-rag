"""An optional second opinion on the order of the retrieved passages.

Reciprocal rank fusion compares ranks, never values, and the dense and sparse arms are the only
judges. A cheap model reading the question against the candidates is a much better judge of
"does this passage answer this question" than either arm alone, because it can see the question
and the passage at the same time. One call per question reorders what retrieval already found —
it cannot add passages, so it can only help precision, never recall.

Two rules, both about not making things worse:

* the model is asked for a *ranking*, never for a score, so there is nothing to calibrate;
* anything unparseable, out of range or failed is ignored and the fused order stands
  (``fail-open``). A reranker that throws must not take the story down with it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from biography_rag.domain import RetrievedChunk
from biography_rag.llm.base import ChatModel
from biography_rag.rag.prompts import rerank_messages

log = logging.getLogger(__name__)

_NUMBER = re.compile(r"\d+")


@dataclass(frozen=True)
class Ranked:
    """A reranked list plus what happened, for the run log."""

    hits: list[RetrievedChunk]
    note: str | None


class Reranker(Protocol):
    def rerank_logged(self, question: str, candidates: Sequence[RetrievedChunk], k: int) -> Ranked: ...

    def rerank(self, question: str, candidates: Sequence[RetrievedChunk], k: int) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class NoReranker:
    """Keeps the fused order: what runs when reranking is off or the model is unavailable."""

    def rerank(self, question: str, candidates: Sequence[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        return list(candidates[:k])

    def rerank_logged(self, question: str, candidates: Sequence[RetrievedChunk], k: int) -> Ranked:
        return Ranked(self.rerank(question, candidates, k), None)


class ChatReranker:
    def __init__(self, chat: ChatModel, max_tokens: int = 200) -> None:
        self._chat = chat
        self._max_tokens = max_tokens

    def rerank(self, question: str, candidates: Sequence[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        return self.rerank_logged(question, candidates, k).hits

    def rerank_logged(self, question: str, candidates: Sequence[RetrievedChunk], k: int) -> Ranked:
        fallback = list(candidates[:k])
        try:
            reply = self._chat.complete(
                rerank_messages(question, [c.chunk for c in candidates]), max_tokens=self._max_tokens
            )
        except Exception as exc:  # fail open: the fused order is still a reasonable answer
            log.warning("rerank failed, keeping fused order: %s", exc)
            return Ranked(fallback, f"rerank failed ({type(exc).__name__}); kept fused order")
        order = _parse(reply, len(candidates))
        if not order:
            return Ranked(fallback, "rerank reply was not a usable ranking; kept fused order")
        picked = [candidates[i - 1] for i in order[:k]]
        return Ranked(picked, f"rerank order of {len(candidates)}: {order[:k]}")


def _parse(reply: str, count: int) -> list[int]:
    """The 1-based indices the model returned, in order, deduplicated and in range. Empty when
    the reply is not a list of numbers at all."""
    numbers = [int(n) for n in _NUMBER.findall(reply)]
    seen: list[int] = []
    for n in numbers:
        if 1 <= n <= count and n not in seen:
            seen.append(n)
    return seen if seen else []
