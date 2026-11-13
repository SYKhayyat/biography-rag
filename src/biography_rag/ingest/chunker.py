"""Paragraph- and sentence-aware chunking with overlap."""

from __future__ import annotations

import re
from typing import Protocol

from biography_rag.domain import Chunk, SourceDocument

_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"'”’)]*\s+")


class Chunker(Protocol):
    def chunk(self, document: SourceDocument) -> list[Chunk]: ...


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(paragraph) if s.strip()]


class ParagraphChunker:
    """Packs whole paragraphs into chunks of about ``size`` characters.

    Paragraphs longer than ``size`` are split on sentence boundaries, and a
    single sentence longer than ``size`` is split hard. Each chunk after the
    first starts with the trailing sentences of the previous one (up to
    ``overlap`` characters), so a fact that straddles a boundary survives intact
    in at least one chunk.
    """

    def __init__(self, size: int = 1200, overlap: int = 200) -> None:
        if size <= 0 or overlap < 0 or overlap >= size:
            raise ValueError("need size > 0 and 0 <= overlap < size")
        self.size = size
        self.overlap = overlap

    def _units(self, text: str) -> list[str]:
        """Sentences, each no longer than ``size``, in order."""
        units: list[str] = []
        for paragraph in text.split("\n\n"):
            for sentence in split_sentences(paragraph):
                while len(sentence) > self.size:
                    cut = sentence.rfind(" ", 0, self.size)
                    cut = cut if cut > self.size // 2 else self.size
                    units.append(sentence[:cut].strip())
                    sentence = sentence[cut:].strip()
                if sentence:
                    units.append(sentence)
            if units:
                units[-1] += "\n\n"  # remember the paragraph break
        return units

    def _tail(self, units: list[str]) -> list[str]:
        tail: list[str] = []
        length = 0
        for unit in reversed(units):
            if length + len(unit) > self.overlap:
                break
            tail.insert(0, unit)
            length += len(unit) + 1
        return tail

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        texts: list[str] = []
        current: list[str] = []
        length = 0
        fresh = 0  # units in `current` that are not overlap carried from the previous chunk
        for unit in self._units(document.text):
            if current and length + len(unit) > self.size and fresh:
                texts.append(_join(current))
                current = self._tail(current)
                length = sum(len(u) + 1 for u in current)
                fresh = 0
            current.append(unit)
            length += len(unit) + 1
            fresh += 1
        if fresh:
            texts.append(_join(current))
        return [
            Chunk(chunk_id=f"{document.doc_id}-{i:04d}", doc_id=document.doc_id, ordinal=i, text=t)
            for i, t in enumerate(texts)
        ]


def _join(units: list[str]) -> str:
    out = ""
    for unit in units:
        if out and not out.endswith("\n\n"):
            out += " "
        out += unit
    return out.strip()
