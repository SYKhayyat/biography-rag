from __future__ import annotations

from typing import Protocol, TypedDict


class Message(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatModel(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(self, messages: list[Message], *, max_tokens: int = 1200) -> str: ...


class ProviderInfo(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def label(self) -> str: ...

    @property
    def model(self) -> str: ...


class ChatModelSource(Protocol):
    """Where services get chat models from; lets tests swap in a scripted fake."""

    def available(self) -> list[ProviderInfo]: ...

    @property
    def default_provider(self) -> str: ...

    def create(self, provider: str | None = None, model: str | None = None) -> ChatModel: ...
