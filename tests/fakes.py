"""Test doubles: a deterministic bag-of-words embedder and a scriptable chat model."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from biography_rag.embeddings.base import unit
from biography_rag.embeddings.registry import EmbedderRegistry
from biography_rag.llm.base import ChatModel, Message
from biography_rag.text import content_terms


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    @property
    def embedder_id(self) -> str:
        return "fake:hashing"

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for term in content_terms(text):
            v[int(hashlib.md5(term.encode()).hexdigest(), 16) % self.dim] += 1.0
        if not any(v):
            v[0] = 1.0
        return v

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return unit([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def fake_registry(embedder: HashingEmbedder | None = None) -> EmbedderRegistry:
    e = embedder or HashingEmbedder()
    return EmbedderRegistry({"fake": lambda _model: e}, {"fake": True}, "fake")


Responder = Callable[[list[Message]], str]


@dataclass
class ScriptedChat:
    responder: Responder
    provider: str = "fake"
    model: str = "fake-1"
    calls: list[list[Message]] = field(default_factory=list)

    def complete(self, messages: list[Message], *, max_tokens: int = 1200) -> str:
        self.calls.append(messages)
        return self.responder(messages)


@dataclass(frozen=True)
class _Info:
    name: str
    label: str
    model: str


class FakeChatSource:
    def __init__(self, chat: ScriptedChat) -> None:
        self.chat = chat

    def available(self) -> list[_Info]:
        return [_Info(self.chat.provider, "Fake", self.chat.model)]

    @property
    def default_provider(self) -> str:
        return self.chat.provider

    def create(self, provider: str | None = None, model: str | None = None) -> ChatModel:
        if provider not in (None, self.chat.provider):
            raise ValueError(f"unknown provider {provider!r}")
        return self.chat


_SNIPPET = re.compile(r"\[(d\d{2}-\d{4})\]\n(.+?)(?=\n\n\[d\d{2}-\d{4}\]\n|\Z)", re.S)


def extractive_responder(messages: list[Message]) -> str:
    """Behaves like an obedient model: quotes the first sentence of the best snippet that
    shares a word with the question, cites it, or says NOT_IN_SOURCES."""
    system, user = messages[0]["content"], messages[-1]["content"]
    if system.startswith("You audit"):
        n = len(re.findall(r"^\d+\. ", user, re.M))
        return "[" + ",".join(f'{{"i": {i}, "supported": true}}' for i in range(1, n + 1)) + "]"
    if system.startswith("You write a short life story"):
        findings = user.split("The snippets cited above")[0]
        lines = [ln for ln in findings.splitlines() if "[d" in ln]
        return "\n\n".join(lines)
    question = user.split("\n", 1)[0].removeprefix("Question: ")
    q_terms = set(content_terms(question)) - {"person", "franklin", "benjamin"}
    for cid, text in _SNIPPET.findall(user):
        if q_terms & set(content_terms(text)):
            first = re.split(r"(?<=[.!?])\s", text.strip())[0].rstrip(".")
            return f'EVIDENCE: "{first}" [{cid}]\nANSWER: {first} [{cid}].'
    return "NOT_IN_SOURCES"
