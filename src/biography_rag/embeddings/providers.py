"""Concrete embedders: OpenAI (default) and a local ONNX model (no key, no cost)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from biography_rag.embeddings.base import unit


class OpenAIEmbedder:
    def __init__(self, client: Any, model: str = "text-embedding-3-small", batch_size: int = 96) -> None:
        self._client = client
        self._model = model
        self._batch = batch_size

    @property
    def embedder_id(self) -> str:
        return f"openai:{self._model}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            batch = list(texts[start : start + self._batch])
            response = self._client.embeddings.create(model=self._model, input=batch)
            out.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return unit(out) if out else []

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class LocalEmbedder:
    """fastembed (ONNX runtime) on CPU. Installed with the ``local`` extra; the model is
    downloaded on first use and imported lazily so the default deploy never pays for it."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model
        self._model: Any = None

    @property
    def embedder_id(self) -> str:
        return f"local:{self._model_name}"

    def _get(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover - depends on install
                raise RuntimeError("local embeddings need the extra: uv sync --extra local") from exc
            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return unit([v.tolist() for v in self._get().passage_embed(list(texts))])

    def embed_query(self, text: str) -> list[float]:
        return unit([next(iter(self._get().query_embed([text]))).tolist()])[0]
