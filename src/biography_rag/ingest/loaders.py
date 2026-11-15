"""Turn uploaded files into clean text. One small strategy per file type (open/closed:
support a new format by registering a loader, without touching the others)."""

from __future__ import annotations

import io
import re
from collections.abc import Callable
from pathlib import PurePath

from bs4 import BeautifulSoup
from pypdf import PdfReader

Loader = Callable[[bytes], str]


class UnsupportedFileType(ValueError):
    pass


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")  # pragma: no cover - latin-1 never fails


def load_text(data: bytes) -> str:
    return _decode(data)


def load_html(data: bytes) -> str:
    soup = BeautifulSoup(_decode(data), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    blocks = [el.get_text(" ", strip=True) for el in soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "blockquote"])]
    blocks = [b for b in blocks if b]
    if not blocks:  # unstructured HTML: fall back to all text
        return soup.get_text("\n", strip=True)
    return "\n\n".join(blocks)


def load_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


LOADERS: dict[str, Loader] = {
    ".txt": load_text,
    ".md": load_text,
    ".text": load_text,
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
}


def supported_extensions() -> list[str]:
    return sorted(LOADERS)


_GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", re.I)
_GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.I)


def strip_boilerplate(text: str) -> str:
    """Remove Project Gutenberg licence headers/footers, which are pure noise for retrieval."""
    start = _GUTENBERG_START.search(text)
    if start:
        text = text[start.end() :]
    end = _GUTENBERG_END.search(text)
    if end:
        text = text[: end.start()]
    return text


_ANCHOR_LINK = re.compile(r"\[[^\[\]]*\]\(#[^)]*\)")


def is_link_list(paragraph: str) -> bool:
    """A markdown table of contents: mostly links to in-page anchors. It names every topic
    and states nothing, so like a back-of-book index it would match every question."""
    links = _ANCHOR_LINK.findall(paragraph)
    return len(links) >= 3 and sum(map(len, links)) >= 0.5 * len(paragraph)


def normalize(text: str) -> str:
    """Unwrap hard-wrapped lines inside paragraphs; keep blank lines as paragraph breaks.
    Tables of contents (paragraphs of in-page anchor links) are dropped."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line-break splits
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned = [re.sub(r"\s*\n\s*", " ", p).strip() for p in paragraphs]
    return "\n\n".join(p for p in cleaned if p and not is_link_list(p))


def extract_text(filename: str, data: bytes) -> str:
    ext = PurePath(filename).suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise UnsupportedFileType(
            f"{filename}: unsupported type {ext or '(none)'}; use {', '.join(supported_extensions())}"
        )
    return normalize(strip_boilerplate(loader(data)))
