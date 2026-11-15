from __future__ import annotations

import pytest

from biography_rag.text import first_person_rate, is_first_person, number_density, to_first_person

# Real chunks from the Franklin corpus (d01-0675, d05-1133, d06-0174, d03-0255).
INDEX_PAGES = [
    "Franklin, James, takes his brother Benjamin Franklin as apprentice, 4; unfriendly relations, 5.",
    "Philadelphia, i., 8, 151\n\nPhiladelphische Zeitung, ii., 74\n\nPitt, Miss, i., 380",
]
NARRATIVE = [
    "On the 17th of April, 1790, about eleven o'clock at night, Dr. Franklin quietly expired. "
    "He had reached an honored and a happy old age.",
    "Upon this Ben paid his garnish, being fully convinced of the folly of not keeping up a good "
    "understanding with those among whom we are destined to live.",
]


@pytest.mark.parametrize("text", INDEX_PAGES)
def test_index_pages_are_dense_with_numbers(text):
    assert number_density(text) > 0.15


@pytest.mark.parametrize("text", NARRATIVE)
def test_narrative_with_dates_is_not(text):
    assert number_density(text) < 0.15


def test_empty_text():
    assert number_density("") == 0.0


MEMOIR = (
    "When I was nine my father sent me to Beis HaTalmud. I hated it, and I ran away to the park. "
    '"You will learn," he told me. My mother was sick, and I sat with her every night.'
)
BIOGRAPHY = (
    "Franklin was born in Boston. He was apprenticed to his brother James, a printer. "
    '"I am resolved," Franklin wrote, "to be frugal." He later moved to Philadelphia.'
)


def test_first_person_detection_ignores_quoted_speech():
    assert is_first_person(MEMOIR)
    assert not is_first_person(BIOGRAPHY)  # the "I" is inside a quotation
    assert first_person_rate("") == 0


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Where and when was Ann Lee born?", "Where and when was I born?"),
        ("What is Ann Lee best known for?", "What am I best known for?"),
        ("Who were Ann Lee's parents?", "Who were my parents?"),
        ("Who were the main influences on Ann Lee?", "Who were the main influences on me?"),
        ("What did Ann Lee study?", "What did I study?"),
    ],
)
def test_to_first_person(question, expected):
    assert to_first_person(question, "Ann Lee") == expected
