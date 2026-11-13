"""Chooses an embedder by name ("openai", "local") for new corpora, and by stored
``embedder_id`` ("openai:text-embedding-3-small") for querying existing ones."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

from biography_rag.config import Settings
from biography_rag.embeddings.base import Embedder
from biography_rag.embeddings.providers import LocalEmbedder, OpenAIEmbedder

# name -> (is_available, build(model or None))
Builder = Callable[[str | None], Embedder]


class EmbedderRegistry:
    def __init__(self, builders: dict[str, Builder], available: dict[str, bool], default: str) -> None:
        self._builders = builders
        self._available = available
        self._default = default
        self._cache: dict[str, Embedder] = {}

    @classmethod
    def from_settings(cls, settings: Settings) -> EmbedderRegistry:
        def openai_builder(model: str | None) -> Embedder:
            from openai import OpenAI

            return OpenAIEmbedder(OpenAI(api_key=settings.openai_api_key), model or settings.openai_embedding_model)

        def local_builder(model: str | None) -> Embedder:
            return LocalEmbedder(model or settings.local_embedding_model)

        available = {
            "openai": bool(settings.openai_api_key),
            "local": importlib.util.find_spec("fastembed") is not None,
        }
        return cls({"openai": openai_builder, "local": local_builder}, available, settings.embedder)

    def names(self) -> list[str]:
        return [n for n in self._builders if self._available.get(n)]

    @property
    def default(self) -> str:
        names = self.names()
        if self._default in names:
            return self._default
        if names:
            return names[0]
        raise RuntimeError("no embedder is available (set OPENAI_API_KEY or install the 'local' extra)")

    def _get(self, name: str, model: str | None) -> Embedder:
        if name not in self._builders:
            raise ValueError(f"unknown embedder {name!r}; choose from {sorted(self._builders)}")
        if not self._available.get(name):
            raise ValueError(f"embedder {name!r} is not available on this server")
        key = f"{name}:{model or ''}"
        if key not in self._cache:
            self._cache[key] = self._builders[name](model)
        return self._cache[key]

    def by_name(self, name: str | None = None) -> Embedder:
        return self._get(name or self.default, None)

    def by_id(self, embedder_id: str) -> Embedder:
        name, _, model = embedder_id.partition(":")
        return self._get(name, model or None)
