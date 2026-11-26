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

    Each chunk also carries the book title and, while the walk has passed one, the chapter
    heading it sits under (``Chunk.context``). A biography's prose says "he" and "Paris"; the
    heading is often the only place that says which year, which city and which book.
    """

    def __init__(self, size: int = 1200, overlap: int = 200) -> None:
        if size <= 0 or overlap < 0 or overlap >= size:
            raise ValueError("need size > 0 and 0 <= overlap < size")
        self.size = size
        self.overlap = overlap

    def _units(self, text: str, title: str) -> list[tuple[str, str]]:
        """Sentences, each no longer than ``size``, in order, with the heading they fall under."""
        units: list[tuple[str, str]] = []
        heading = ""
        for paragraph in text.split("\n\n"):
            if is_heading(paragraph):
                heading = " ".join(paragraph.split())[:120]
                continue
            for sentence in split_sentences(paragraph):
                while len(sentence) > self.size:
                    cut = sentence.rfind(" ", 0, self.size)
                    cut = cut if cut > self.size // 2 else self.size
                    units.append((sentence[:cut].strip(), heading))
                    sentence = sentence[cut:].strip()
                if sentence:
                    units.append((sentence, heading))
            if units:
                units[-1] = (units[-1][0] + "\n\n", units[-1][1])  # remember the paragraph break
        if not units:  # every paragraph looked like a heading: it is a title page, keep the text
            units = [(sentence, "") for sentence in split_sentences(text) if sentence]
        return units

    def _tail(self, units: list[tuple[str, str]]) -> list[tuple[str, str]]:
        tail: list[tuple[str, str]] = []
        length = 0
        for unit in reversed(units):
            if length + len(unit[0]) > self.overlap:
                break
            tail.insert(0, unit)
            length += len(unit[0]) + 1
        return tail

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        texts: list[tuple[str, str]] = []  # (text, heading of the chunk's first unit)
        current: list[tuple[str, str]] = []
        length = 0
        fresh = 0  # units in `current` that are not overlap carried from the previous chunk
        for unit in self._units(document.text, document.title):
            if current and length + len(unit[0]) > self.size and fresh:
                texts.append((_join([u for u, _ in current]), current[0][1]))
                current = self._tail(current)
                length = sum(len(u) + 1 for u, _ in current)
                fresh = 0
            current.append(unit)
            length += len(unit[0]) + 1
            fresh += 1
        if fresh:
            texts.append((_join([u for u, _ in current]), current[0][1]))
        return [
            Chunk(
                chunk_id=f"{document.doc_id}-{i:04d}",
                doc_id=document.doc_id,
                ordinal=i,
                text=text,
                context=_context(document.title, heading),
            )
            for i, (text, heading) in enumerate(texts)
        ]


_TERMINAL = re.compile(r"[.!?…:;,]\)?\s*$")
_NUMBERED = re.compile(r"^(chapter|part|book|letter|volume|section)\b", re.I)


def is_heading(paragraph: str) -> bool:
    """A short line with no sentence ending that looks like a title: "Chapter IV", "PARIS",
    "Letter the Second". A 12-word rule keeps a short narrative line ("He was born in Boston.")
    out, and the punctuation rule keeps any line that ends a sentence out."""
    text = " ".join(paragraph.split())
    if not text or len(text) > 80 or len(text.split()) > 12:
        return False
    if _TERMINAL.search(text):
        return False
    return bool(_NUMBERED.match(text)) or text.isupper() or text.istitle()


def _context(title: str, heading: str) -> str:
    title = " ".join(title.split())
    if heading and heading.lower() not in title.lower():
        return f"{title} — {heading}" if title else heading
    return title


def _join(units: list[str]) -> str:
    out = ""
    for unit in units:
        if out and not out.endswith("\n\n"):
            out += " "
        out += unit
    return out.strip()
