from __future__ import annotations

from collections.abc import Sequence
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


class ModelInfo(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def label(self) -> str: ...

    @property
    def slow(self) -> bool: ...


class ProviderInfo(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def label(self) -> str: ...

    @property
    def model(self) -> str: ...  # the default model

    @property
    def models(self) -> Sequence[ModelInfo]: ...  # every model users may pick, default first


class ChatModelSource(Protocol):
    """Where services get chat models from; lets tests swap in a scripted fake."""

    def available(self) -> list[ProviderInfo]: ...

    @property
    def default_provider(self) -> str: ...

    def create(self, provider: str | None = None, model: str | None = None) -> ChatModel: ...
