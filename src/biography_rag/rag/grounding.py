"""The evidence gate: a deterministic check between the model's reply and the answer we keep.

The answer prompt makes the model copy, word for word, the snippet text that states the answer
before it writes anything. This module checks those quotes against the snippets themselves,
so "not in the sources" no longer depends on the model choosing to say so:

* no EVIDENCE line whose quote really occurs in the snippet it names -> not in the sources
  (the model answered from memory and could only point at loosely related text);
* answer sentences with no citation to a retrieved snippet are dropped;
* nothing left after that -> not in the sources.

Every rejection or trim carries a reason, which the run log records next to the raw reply.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from biography_rag.domain import Chunk
from biography_rag.rag.citations import cited_ids, split_sentences
from biography_rag.text import words

_EVIDENCE = re.compile(r"^\s*EVIDENCE:\s*[\"“](.+?)[\"”]\s*\[([^\]]+)\]", re.M | re.I)
_ANSWER = re.compile(r"^\s*ANSWER:\s*", re.M | re.I)
_ELLIPSIS = re.compile(r"\.\s*\.\s*\.|…|\[\s*\.\.\.\s*\]")
MIN_QUOTE_WORDS = 3


@dataclass(frozen=True)
class GateResult:
    answer: str | None  # None: not in the sources
    cited_ids: list[str]
    evidence: list[dict]  # [{"quote", "chunk_id", "found"}]
    reason: str | None  # why the reply was rejected or trimmed; None when kept as is


def _normal(text: str) -> str:
    return " " + " ".join(words(text)) + " "


def check_quote(quote: str, chunk: Chunk) -> bool:
    """True if the quote occurs word for word in the chunk (case, spacing and punctuation aside).

    Models like to shorten long quotes with "...": each piece must then occur, in order, and
    pieces under MIN_QUOTE_WORDS are ignored, but at least one piece must be long enough."""
    text = _normal(chunk.text)
    pieces = [_normal(p) for p in _ELLIPSIS.split(quote)]
    pieces = [p for p in pieces if len(p.split()) >= MIN_QUOTE_WORDS]
    pos = 0
    for piece in pieces:
        pos = text.find(piece, pos)
        if pos < 0:
            return False
        pos += len(piece) - 1  # the pieces share no words; keep the boundary space
    return bool(pieces)


def gate(response: str, chunks: Mapping[str, Chunk]) -> GateResult:
    evidence = []
    for quote, ids in _EVIDENCE.findall(response):
        for cid in (i.strip() for i in ids.split(",")):
            evidence.append(
                {"quote": quote, "chunk_id": cid, "found": cid in chunks and check_quote(quote, chunks[cid])}
            )
    if not any(e["found"] for e in evidence):
        why = "no evidence quote given" if not evidence else "no evidence quote occurs in the snippet it cites"
        return GateResult(None, [], evidence, why)

    match = _ANSWER.search(response)
    body = response[match.end() :] if match else _EVIDENCE.sub("", response)
    sentences = split_sentences(body)
    kept = [s for s in sentences if cited_ids(s) and all(i in chunks for i in cited_ids(s))]
    if not kept:
        return GateResult(None, [], evidence, "no answer sentence cites a retrieved snippet")
    answer = " ".join(kept)
    reason = f"dropped {len(sentences) - len(kept)} uncited sentence(s)" if len(kept) < len(sentences) else None
    return GateResult(answer, cited_ids(answer), evidence, reason)
