"""Every prompt the app sends lives here, so they can be read (and reviewed) in one place."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from biography_rag.domain import Chunk, QueryAnswer
from biography_rag.llm.base import Message

NOT_IN_SOURCES_TOKEN = "NOT_IN_SOURCES"

ANSWER_SYSTEM = f"""You answer one question about {{person}} using ONLY the source snippets you are given.

Rules:
1. Use only facts stated in the snippets. Never add outside knowledge, even if you are sure it is true. \
No dates, places, names or numbers that are not written in a snippet.
2. First copy the evidence: the exact words from a snippet that directly state the answer. \
Your evidence is checked word for word against the snippet, so copy, do not paraphrase.
3. If no snippet directly states the answer, reply with exactly {NOT_IN_SOURCES_TOKEN} and nothing else. \
A snippet that only mentions the same person, object or topic without stating the answer does not count. \
Some questions deliberately ask about famous facts: if you know the answer but cannot quote it from a \
snippet, the answer is {NOT_IN_SOURCES_TOKEN}.
4. Keep track of who is who: attach each fact to the person the snippet attaches it to \
(a son is not a grandson; one child's death is not another's).
5. If the snippets disagree, say so and cite both.
6. A snippet marked "(first person)" is {{person}}'s own account: there "I", "me" and "my" mean \
{{person}}, except inside quoted speech by someone else. Answer in the third person about {{person}}.

Reply in exactly this format:
EVIDENCE: "<exact words copied from a snippet>" [<snippet id>]
(one EVIDENCE line per passage you rely on, at most 4; each quote one continuous passage of \
at most 40 words, copied exactly, never shortened with "...")
ANSWER: <2 to 6 plain sentences, each ending with the bracketed id(s) supporting it, \
for example [d01-0042] or [d01-0042, d03-0007]>

Do not mention "snippets", "sources" or "the text" in the answer."""

EVIDENCE_CHECK_SYSTEM = """You check whether quoted evidence answers a question about {person}. \
You see only the quotes, not where they came from, except that a quote marked "(first person)" \
comes from {person}'s own account: there "I", "me" and "my" mean {person}, unless the words are \
someone else's quoted speech.

Reply YES only if the quotes themselves state the answer (a paraphrase of what they say is fine). \
Reply NO if they merely mention the same person, object or topic, or if answering would need any fact \
that is not in the quotes, however well known.

Reply with one word: YES or NO."""

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
Paraphrase is fine; any fact that is absent from the cited snippets is not. Check who each fact \
belongs to: a fact the snippet states about one person (a son, a brother, a grandson) but the \
sentence attaches to another is not supported.

Reply with JSON only: a list of objects {"i": <sentence number>, "supported": true|false, \
"reason": "<short reason if false>"}. Include every sentence."""


FIRST_PERSON_MARK = "(first person)"


def _snippets(chunks: Sequence[Chunk], first_person_docs: Collection[str] = ()) -> str:
    return "\n\n".join(
        f"[{c.chunk_id}]{' ' + FIRST_PERSON_MARK if c.doc_id in first_person_docs else ''}\n{c.text}" for c in chunks
    )


def answer_messages(
    person: str, question: str, chunks: Sequence[Chunk], first_person_docs: Collection[str] = ()
) -> list[Message]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM.format(person=person)},
        {"role": "user", "content": f"Question: {question}\n\nSnippets:\n\n{_snippets(chunks, first_person_docs)}"},
    ]


def evidence_check_messages(
    person: str, question: str, quotes: Sequence[str], first_person: Collection[str] = ()
) -> list[Message]:
    """``first_person``: the quotes that come from the person's own (first-person) account."""
    listed = "\n".join(f'- "{q}"' + (f" {FIRST_PERSON_MARK}" if q in first_person else "") for q in quotes)
    return [
        {"role": "system", "content": EVIDENCE_CHECK_SYSTEM.format(person=person)},
        {"role": "user", "content": f"Question: {question}\n\nQuotes:\n{listed}"},
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
