"""Parsing chunk-id citations and splitting prose into checkable sentences."""

from __future__ import annotations

import re

CHUNK_ID = re.compile(r"\bd\d{2,}-\d{4,}\b")
_BRACKET = re.compile(r"\[([^\[\]]*)\]")
_ABBREVIATIONS = (
    "Mr",
    "Mrs",
    "Ms",
    "Dr",
    "St",
    "Jr",
    "Sr",
    "Rev",
    "Gen",
    "Col",
    "Capt",
    "Gov",
    "Hon",
    "Lt",
    "No",
    "vs",
)
_PLACEHOLDER = "⁣"  # invisible separator: protects abbreviation periods during splitting


def cited_ids(text: str) -> list[str]:
    """Chunk ids that appear inside square brackets, in first-seen order."""
    seen: list[str] = []
    for group in _BRACKET.findall(text):
        for cid in CHUNK_ID.findall(group):
            if cid not in seen:
                seen.append(cid)
    return seen


def keep_citations(text: str, allowed) -> str:
    """Remove chunk ids not in ``allowed`` from every bracket; drop brackets left empty."""

    def fix(m: re.Match) -> str:
        group = m.group(1)
        ids = CHUNK_ID.findall(group)
        if not ids:
            return m.group(0)
        kept = [i for i in ids if i in allowed]
        return f"[{', '.join(kept)}]" if kept else ""

    return re.sub(r"\s+([.,;:!?])", r"\1", _BRACKET.sub(fix, text)).strip()


def cite_forward(paragraph: str) -> list[tuple[str, bool]]:
    """The paragraph's sentences, each with whether it borrowed a citation: an uncited sentence
    takes the ids of the next cited sentence in the paragraph (some models cite once, at the end
    of a paragraph). Uncited sentences with no citation after them stay uncited."""
    out: list[tuple[str, bool]] = []
    pending: list[str] = []
    for s in split_sentences(paragraph):
        ids = cited_ids(s)
        if not ids:
            pending.append(s)
            continue
        tag = f" [{', '.join(ids)}]"
        for p in pending:
            out.append((re.sub(r"([.!?][\"”’']?)?$", lambda m, tag=tag: tag + (m.group(1) or ""), p, count=1), True))
        pending = []
        out.append((s, False))
    out.extend((p, False) for p in pending)
    return out


def fill_paragraph_citations(text: str) -> str:
    """``text`` with every uncited sentence given its paragraph's next citation (see cite_forward)."""
    paragraphs = (cite_forward(p) for p in re.split(r"\n\s*\n", text))
    return "\n\n".join(" ".join(s for s, _ in pairs) for pairs in paragraphs if pairs)


def strip_citations(text: str) -> str:
    return re.sub(r"\s+([.,;:!?])", r"\1", _BRACKET.sub("", text)).strip()


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping each sentence's trailing citation with it.

    Handles both "...Boston [d01-0003]." and "...Boston. [d01-0003]".
    """
    out: list[str] = []
    for paragraph in re.split(r"\n\s*\n|\n(?=#)", text):
        paragraph = paragraph.strip()
        if not paragraph or paragraph.startswith("#"):
            continue
        p = re.sub(r"\s+", " ", paragraph)
        p = re.sub(
            r"([.!?])([\"”’']?)\s*((?:\[[^\[\]]*\]\s*)+)",
            lambda m: f" {m.group(3).strip()}{m.group(1)}{m.group(2)} ",
            p,
        )
        for abbr in _ABBREVIATIONS:
            p = re.sub(rf"\b{abbr}\.", f"{abbr}{_PLACEHOLDER}", p)
        p = re.sub(r"\b([A-Z])\.(?=\s?[A-Z])", rf"\1{_PLACEHOLDER}", p)  # initials: "B. Franklin"
        for sentence in re.split(r"(?<=[.!?])[\"”’']?\s+(?=[\[\"“(A-Z0-9])", p):
            sentence = sentence.replace(_PLACEHOLDER, ".").strip()
            if sentence:
                out.append(sentence)
    return out
