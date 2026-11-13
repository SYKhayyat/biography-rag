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
