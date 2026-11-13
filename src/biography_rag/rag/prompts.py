"""Every prompt the app sends lives here, so they can be read (and reviewed) in one place."""

from __future__ import annotations

from collections.abc import Sequence

from biography_rag.domain import Chunk, QueryAnswer
from biography_rag.llm.base import Message

NOT_IN_SOURCES_TOKEN = "NOT_IN_SOURCES"

ANSWER_SYSTEM = f"""You answer one question about {{person}} using ONLY the source snippets you are given.

Rules:
1. Use only facts stated in the snippets. Never add outside knowledge, even if you are sure it is true. \
No dates, places, names or numbers that are not written in a snippet.
2. End every sentence with the id(s) of the snippet(s) that support it, in square brackets, \
for example [d01-0042] or [d01-0042, d03-0007]. Copy ids exactly.
3. If the snippets do not answer the question, reply with exactly {NOT_IN_SOURCES_TOKEN} and nothing else. \
A snippet that merely mentions the topic without answering it does not count.
4. If the snippets disagree, say so and cite both.
5. Write 2 to 6 plain sentences. Do not mention "snippets", "sources" or "the text"."""

WEAVE_SYSTEM = """You write a short life story of {person} by weaving together research findings.

Rules:
1. Use ONLY facts that appear in the findings. Do not add any fact, date, name, place, number, \
motive or judgement that is not in them, even if you know it.
2. Every sentence must end with the bracketed snippet id(s) supporting it, copied exactly \
from the findings, for example [d01-0042]. A sentence you cannot cite must not be written.
3. Transitions are allowed only if they add no new information ("Later," "Meanwhile,").
4. Skip any topic marked NOT_IN_SOURCES. Do not mention that it is missing.
5. Write flowing paragraphs in roughly chronological order. No title, no headings, no lists."""

JUDGE_SYSTEM = """You audit a story for faithfulness. For each numbered sentence, decide whether \
the snippets it cites fully support every fact in it (names, dates, places, numbers, events). \
Paraphrase is fine; any fact that is absent from the cited snippets is not.

Reply with JSON only: a list of objects {"i": <sentence number>, "supported": true|false, \
"reason": "<short reason if false>"}. Include every sentence."""


def _snippets(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in chunks)


def answer_messages(person: str, question: str, chunks: Sequence[Chunk]) -> list[Message]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM.format(person=person)},
        {"role": "user", "content": f"Question: {question}\n\nSnippets:\n\n{_snippets(chunks)}"},
    ]


def weave_messages(person: str, answers: Sequence[QueryAnswer], cited: Sequence[Chunk]) -> list[Message]:
    findings = "\n\n".join(
        f"Q{a.query_id}. {a.question}\n{a.answer if a.answer is not None else NOT_IN_SOURCES_TOKEN}" for a in answers
    )
    return [
        {"role": "system", "content": WEAVE_SYSTEM.format(person=person)},
        {
            "role": "user",
            "content": f"Findings:\n\n{findings}\n\nThe snippets cited above, for reference:\n\n{_snippets(cited)}",
        },
    ]


def judge_messages(sentences: Sequence[tuple[str, Sequence[str]]], chunks: Sequence[Chunk]) -> list[Message]:
    numbered = "\n".join(f"{i}. {s}  (cites: {', '.join(ids) or 'nothing'})" for i, (s, ids) in enumerate(sentences, 1))
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"Snippets:\n\n{_snippets(chunks)}\n\nSentences:\n{numbered}"},
    ]
