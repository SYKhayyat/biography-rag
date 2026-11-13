"""LLM steps: answer one question from its snippets, then weave answers into a story."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from biography_rag.domain import Chunk, QueryAnswer
from biography_rag.llm.base import ChatModel, Message
from biography_rag.rag.citations import cited_ids
from biography_rag.rag.prompts import NOT_IN_SOURCES_TOKEN, answer_messages, weave_messages


@dataclass(frozen=True)
class LlmExchange:
    """What was sent and what came back; ``messages`` is empty when the LLM was not called."""

    messages: list[Message]
    response: str | None


def _is_not_in_sources(response: str) -> bool:
    text = response.strip().strip(".").strip()
    upper = text.upper().replace(" ", "_")
    return not text or upper.startswith(NOT_IN_SOURCES_TOKEN) or (NOT_IN_SOURCES_TOKEN in upper and not cited_ids(text))


class QueryAnswerer:
    def __init__(self, chat: ChatModel) -> None:
        self._chat = chat

    def answer(
        self, query_id: int, person: str, question: str, chunks: Sequence[Chunk]
    ) -> tuple[QueryAnswer, LlmExchange]:
        ids = [c.chunk_id for c in chunks]
        if not chunks:  # nothing retrieved: never ask the model, it could only guess
            return QueryAnswer(query_id, question, ids, None, []), LlmExchange([], None)
        messages = answer_messages(person, question, chunks)
        response = self._chat.complete(messages, max_tokens=700)
        if _is_not_in_sources(response):
            return QueryAnswer(query_id, question, ids, None, []), LlmExchange(messages, response)
        return QueryAnswer(query_id, question, ids, response, cited_ids(response)), LlmExchange(messages, response)


class StoryWeaver:
    def __init__(self, chat: ChatModel) -> None:
        self._chat = chat

    def weave(self, person: str, answers: Sequence[QueryAnswer], cited: Sequence[Chunk]) -> tuple[str, LlmExchange]:
        if not any(a.found for a in answers):
            return "", LlmExchange([], None)
        messages = weave_messages(person, answers, cited)
        story = self._chat.complete(messages, max_tokens=3500)
        return story, LlmExchange(messages, story)
