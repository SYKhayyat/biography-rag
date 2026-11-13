from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Maps text to unit-length vectors. ``embedder_id`` is stored with each corpus so a
    corpus is always queried with the same model that indexed it."""

    @property
    def embedder_id(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def unit(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).tolist()
