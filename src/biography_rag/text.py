"""Tiny, dependency-free text utilities shared by keyword search and verification."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "one",
        "upon",
        "many",
        "much",
        "may",
        "might",
        "must",
        "shall",
        "us",
        "yet",
        "ever",
        "even",
        "though",
        "although",
        "whose",
        "since",
    ]
)

_WORD = re.compile(r"[A-Za-z][A-Za-z'’]*|\d+")


def words(text: str) -> list[str]:
    return [w.lower().replace("’", "'").removesuffix("'s") for w in _WORD.findall(text)]


_NUMBER = re.compile(r"\b\d+\b")
_LETTERS = re.compile(r"[A-Za-z]+")


def number_density(text: str) -> float:
    """Numbers per word. Back-of-book index pages ("Josiah, emigrates to Boston, 2; his family,
    2, 3") score 0.15-1.2; narrative biography, dates and all, stays under about 0.1."""
    return len(_NUMBER.findall(text)) / max(1, len(_LETTERS.findall(text)))


def stem(word: str) -> str:
    """Crude suffix stripper: enough to match 'invented'/'invention'/'invents'."""
    for suffix in ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ied", "ed", "es", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def content_terms(text: str) -> list[str]:
    return [stem(w) for w in words(text) if w not in STOPWORDS and (len(w) > 2 or w.isdigit())]


_FIRST_PERSON = re.compile(r"\b(I|me|my|myself|mine)\b")
_QUOTED = re.compile(r"[\"“][^\"”]{0,2000}[\"”]")
# First-person pronouns per word, outside quotation marks. Biographies that quote letters at
# length (Bruce's "Franklin, Self-Revealed") stay under 0.016; a dictated memoir is ~0.056.
FIRST_PERSON_MIN = 0.03


def first_person_rate(text: str) -> float:
    """Share of words that are I/me/my/myself/mine, not counting quoted dialogue."""
    narration = _QUOTED.sub(" ", text)
    return len(_FIRST_PERSON.findall(narration)) / max(1, len(_LETTERS.findall(narration)))


def is_first_person(text: str) -> bool:
    """True when the text reads as the subject's own account (memoir, autobiography, diary)."""
    return first_person_rate(text) >= FIRST_PERSON_MIN


def to_first_person(question: str, person: str) -> str:
    """Rephrase a question about ``person`` the way a memoir would put it, for retrieval only:
    "Where was Jane Roe born?" -> "Where was I born?". The memoir says "I" and "my", rarely its
    narrator's name, so a query built around the name finds the few passages where *others*
    mention them. Returns the question unchanged when the name is not in it."""
    name = re.escape(person)
    q = re.sub(rf"{name}[’']s\b", "my", question)
    q = re.sub(rf"\bof\s+{name}\b", "of mine", q)
    q = re.sub(rf"\b(about|for|to|with|from|by|against|on|at)\s+{name}\b", r"\1 me", q, flags=re.I)
    q = re.sub(
        rf"\b(is|was|did|does|has|had)\s+{name}\b", lambda m: f"{'am' if m.group(1) == 'is' else m.group(1)} I", q
    )
    q = re.sub(rf"\b{name}\b", "I", q)
    return q
