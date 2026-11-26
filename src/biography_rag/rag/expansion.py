"""Optional query expansion: a written-out answer (HyDE) and a couple of rephrasings.

``hybrid search`` fails in one specific way: a biography states "he was bound apprentice to his
brother James", and the question says "what was his first job". Nothing in the question is close
to the passage. A model that knows what a biography *sentence* about the topic looks like closes
that gap: the invented passage is embedded and searched, and it lands near the real ones.

The invented text is a **query only**. It never reaches the answer prompt and never counts as
evidence — it is written without the sources, so its facts may be wrong, and the whole point of
the answer stage is that facts come from snippets. It embeds badly-worded too: that costs recall,
which is why it is one extra dense arm among several rather than a replacement.

Everything here fails open. A provider error, an unparseable reply or a refusal means the search
runs on the original phrasing, exactly as before.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from biography_rag.llm.base import ChatModel
from biography_rag.rag.prompts import expansion_messages

log = logging.getLogger(__name__)

_PASSAGE = re.compile(r"^\s*PASSAGE:\s*(.+?)(?=^\s*PHRASINGS:|\Z)", re.M | re.S | re.I)
_PHRASINGS = re.compile(r"^\s*PHRASINGS:\s*(.+)\Z", re.M | re.S | re.I)
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", re.M)
MAX_PASSAGE_CHARS = 900


@dataclass(frozen=True)
class Expansion:
    """What the expander produced; empty fields mean "search the question as written"."""

    hypothetical: str = ""
    phrasings: tuple[str, ...] = ()
    note: str | None = None


NO_EXPANSION = Expansion()


class QueryExpander:
    def __init__(self, chat: ChatModel, max_phrasings: int = 2, max_tokens: int = 500) -> None:
        self._chat = chat
        self._max_phrasings = max_phrasings
        self._max_tokens = max_tokens

    def expand(self, person: str, question: str) -> Expansion:
        try:
            reply = self._chat.complete(expansion_messages(person, question), max_tokens=self._max_tokens)
        except Exception as exc:  # fail open: the plain question is a fine query
            log.warning("query expansion failed: %s", exc)
            return Expansion(note=f"expansion failed ({type(exc).__name__}); searched the question as asked")
        found = parse_expansion(reply, self._max_phrasings)
        if not found.hypothetical and not found.phrasings:
            return Expansion(note="expansion produced nothing usable; searched the question as asked")
        return found


def parse_expansion(reply: str, max_phrasings: int = 2) -> Expansion:
    """Read the PASSAGE / PHRASINGS block. Tolerant of bullets, numbering and a missing half."""
    passage = ""
    match = _PASSAGE.search(reply)
    if match:
        passage = " ".join(match.group(1).split())[:MAX_PASSAGE_CHARS]
    phrasings: list[str] = []
    block = _PHRASINGS.search(reply)
    if block:
        for line in _BULLET.sub("", block.group(1)).splitlines():
            line = " ".join(line.split()).strip(" -–—")
            if len(line.split()) >= 2 and line not in phrasings:
                phrasings.append(line[:200])
    return Expansion(passage, tuple(phrasings[:max_phrasings]))
