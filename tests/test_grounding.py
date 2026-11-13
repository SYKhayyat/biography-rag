from __future__ import annotations

import pytest

from biography_rag.domain import Chunk
from biography_rag.rag.grounding import check_quote, gate

C1 = Chunk("d01-0000", "d01", 0, "Benjamin Franklin was born in Boston in 1706, the son of Josiah, a candle-maker.")
C2 = Chunk("d01-0001", "d01", 1, "He flew a kite during a thunderstorm.")
CHUNKS = {c.chunk_id: c for c in (C1, C2)}


class TestCheckQuote:
    @pytest.mark.parametrize(
        "quote",
        [
            "was born in Boston in 1706",
            "WAS BORN in   Boston",
            "the son of Josiah, a candle-maker",
            "Josiah a candle maker",
        ],
    )
    def test_matches_ignoring_case_space_and_punctuation(self, quote):
        assert check_quote(quote, C1)

    @pytest.mark.parametrize("quote", ["born in Philadelphia", "Boston", "Boston in", "son of Josiah Franklin"])
    def test_rejects_paraphrase_altered_and_too_short_quotes(self, quote):
        assert not check_quote(quote, C1)

    def test_word_boundaries(self):
        assert not check_quote("orn in Bosto", C1)


class TestGate:
    def test_keeps_a_grounded_answer(self):
        r = gate('EVIDENCE: "was born in Boston in 1706" [d01-0000]\nANSWER: He was born in Boston [d01-0000].', CHUNKS)
        assert r.answer == "He was born in Boston [d01-0000]." and r.cited_ids == ["d01-0000"] and r.reason is None

    def test_no_evidence_line(self):
        r = gate("He was born in Boston [d01-0000].", CHUNKS)
        assert r.answer is None and r.reason == "no evidence quote given"

    def test_evidence_quoted_from_the_wrong_snippet(self):
        r = gate('EVIDENCE: "was born in Boston" [d01-0001]\nANSWER: Born in Boston [d01-0000].', CHUNKS)
        assert r.answer is None and r.evidence == [
            {"quote": "was born in Boston", "chunk_id": "d01-0001", "found": False}
        ]

    def test_evidence_citing_an_unretrieved_snippet(self):
        r = gate('EVIDENCE: "was born in Boston" [d09-0009]\nANSWER: Born in Boston [d09-0009].', CHUNKS)
        assert r.answer is None

    def test_one_good_quote_is_enough_and_multi_id_lines_are_split(self):
        reply = (
            'EVIDENCE: "flew a kite during a thunderstorm" [d01-0000, d01-0001]\n'
            'EVIDENCE: "invented the telephone" [d01-0000]\n'
            "ANSWER: He flew a kite [d01-0001]."
        )
        r = gate(reply, CHUNKS)
        assert r.answer == "He flew a kite [d01-0001]."
        assert [e["found"] for e in r.evidence] == [False, True, False]

    def test_uncited_and_badly_cited_sentences_are_dropped(self):
        reply = (
            'EVIDENCE: "flew a kite" [d01-0001]\n'
            "ANSWER: He flew a kite [d01-0001]. He also signed the Constitution. He invented bifocals [d07-0001]."
        )
        r = gate(reply, CHUNKS)
        assert r.answer == "He flew a kite [d01-0001]." and r.reason == "dropped 2 uncited sentence(s)"

    def test_nothing_cited_left(self):
        r = gate('EVIDENCE: "flew a kite" [d01-0001]\nANSWER: He flew a kite.', CHUNKS)
        assert r.answer is None and "no answer sentence" in r.reason

    def test_curly_quotes_and_missing_answer_label(self):
        r = gate("EVIDENCE: “flew a kite” [d01-0001]\nHe flew a kite [d01-0001].", CHUNKS)
        assert r.answer == "He flew a kite [d01-0001]."
