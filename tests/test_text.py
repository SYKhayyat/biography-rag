from __future__ import annotations

import pytest

from biography_rag.text import number_density

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
