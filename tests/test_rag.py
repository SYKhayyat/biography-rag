from __future__ import annotations

import json

import pytest

from biography_rag.domain import Chunk, RunStatus, Verdict
from biography_rag.rag.citations import cited_ids, split_sentences, strip_citations
from biography_rag.rag.generation import QueryAnswerer, StoryWeaver, says_yes
from biography_rag.rag.prompts import answer_messages
from biography_rag.rag.queries import QUERIES
from biography_rag.rag.retriever import HybridRetriever, RetrievalConfig
from biography_rag.rag.verification import JudgedVerifier, LexicalVerifier, _parse_judgement
from biography_rag.store.memory import MemoryChunkIndex
from tests.fakes import HashingEmbedder, ScriptedChat

C1 = Chunk("d01-0000", "d01", 0, "Benjamin Franklin was born in Boston in 1706, the son of Josiah, a candle maker.")
C2 = Chunk("d01-0001", "d01", 1, "He flew a kite during a thunderstorm and proved lightning was electricity.")
CHUNKS = {c.chunk_id: c for c in (C1, C2)}


class TestQueries:
    def test_twenty_unique_templates_with_person_slot(self):
        assert len(QUERIES) == 20
        assert [q.query_id for q in QUERIES] == list(range(1, 21))
        assert all("{person}" in q.template for q in QUERIES)

    def test_render(self):
        assert QUERIES[0].render("Ada Lovelace") == "Where and when was Ada Lovelace born?"

    def test_keyword_hints_never_name_a_person_or_place(self):
        assert not any(w[0].isupper() for q in QUERIES for w in q.keywords.split())


class TestCitations:
    def test_cited_ids_in_brackets_only(self):
        assert cited_ids("Born in Boston [d01-0000, d02-0013]. Not d09-0001.") == ["d01-0000", "d02-0013"]

    def test_strip(self):
        assert strip_citations("Born in Boston [d01-0000].") == "Born in Boston."

    def test_split_keeps_citation_with_its_sentence_either_side_of_period(self):
        text = "He was born in Boston [d01-0000]. He flew a kite. [d01-0001] Mr. Smith laughed [d01-0001]."
        assert split_sentences(text) == [
            "He was born in Boston [d01-0000].",
            "He flew a kite [d01-0001].",
            "Mr. Smith laughed [d01-0001].",
        ]

    def test_split_skips_headings_and_handles_paragraphs(self):
        assert split_sentences("# Title\n\nOne [d01-0000].\n\nTwo [d01-0001].") == [
            "One [d01-0000].",
            "Two [d01-0001].",
        ]


class TestLexicalVerifier:
    v = LexicalVerifier()
    allowed = frozenset(CHUNKS)

    def test_supported(self):
        c = self.v.check("Franklin was born in Boston in 1706 [d01-0000].", CHUNKS, self.allowed)
        assert c.verdict is Verdict.SUPPORTED

    def test_uncited_is_flagged(self):
        assert self.v.check("He was born in Boston.", CHUNKS, self.allowed).verdict is Verdict.UNCITED

    def test_citation_to_unretrieved_chunk_is_flagged(self):
        c = self.v.check("He was born in Boston [d09-0009].", CHUNKS, self.allowed)
        assert c.verdict is Verdict.BAD_CITATION

    def test_training_data_date_is_caught(self):
        # The chunk says 1706 but not the day: "January 17" must come from model memory.
        c = self.v.check("He was born in Boston on January 17, 1706 [d01-0000].", CHUNKS, self.allowed)
        assert c.verdict is Verdict.WEAK
        assert "17" in c.missing_terms and "January" in c.missing_terms

    def test_unfamiliar_name_is_caught(self):
        c = self.v.check("He was born in Boston, like Paul Revere [d01-0000].", CHUNKS, self.allowed)
        assert c.verdict is Verdict.WEAK
        assert {"Paul", "Revere"} <= set(c.missing_terms)

    def test_report(self):
        report = self.v.verify("Born in Boston [d01-0000]. He invented bifocals.", CHUNKS, self.allowed).report
        assert len(report.checks) == 2
        assert report.traced_ratio == 0.5


class TestJudgedVerifier:
    def judge(self, reply: str) -> JudgedVerifier:
        return JudgedVerifier(LexicalVerifier(), ScriptedChat(lambda m: reply))

    def test_judge_can_flag_a_lexically_fine_sentence(self):
        reply = json.dumps([{"i": 1, "supported": False, "reason": "wrong father"}])
        r = self.judge(reply).verify("Born in Boston [d01-0000].", CHUNKS, set(CHUNKS)).report
        assert r.checks[0].verdict is Verdict.JUDGE_FLAGGED
        assert "wrong father" in r.checks[0].note

    def test_judge_can_clear_a_weak_paraphrase_but_not_a_missing_citation(self):
        reply = '[{"i":1,"supported":true},{"i":2,"supported":true}]'
        text = "His dad crafted tapers in the city [d01-0000]. He invented bifocals."
        r = self.judge(reply).verify(text, CHUNKS, set(CHUNKS)).report
        assert r.checks[0].verdict is Verdict.SUPPORTED
        assert r.checks[1].verdict is Verdict.UNCITED

    def test_unparseable_judge_keeps_lexical_result(self):
        r = self.judge("sorry, I cannot").verify("Born in Boston [d01-0000].", CHUNKS, set(CHUNKS))
        assert r.report.checks[0].verdict is Verdict.SUPPORTED
        assert r.judge_response == "sorry, I cannot"

    def test_parse_judgement_tolerates_prose_around_json(self):
        assert _parse_judgement('Here: [{"i": 2, "supported": false, "reason": "x"}] done') == {2: (False, "x")}


class TestRetriever:
    def test_hybrid_returns_relevant_chunk_first(self):
        index = MemoryChunkIndex()
        emb = HashingEmbedder()
        filler = [
            Chunk(f"d02-{i:04d}", "d02", i, f"Unrelated filler about printing, item number {i} of many.")
            for i in range(10)
        ]
        chunks = [C1, C2, *filler]
        index.add("c", chunks, emb.embed_documents([c.text for c in chunks]))
        hits = HybridRetriever(index, emb, RetrievalConfig(top_k=3)).retrieve("c", "When was he born?", "born birth")
        assert hits[0].chunk.chunk_id == "d01-0000"
        assert hits[0].keyword_rank == 1
        assert len(hits) == 3

    def test_min_similarity_filters_dense_arm(self):
        index = MemoryChunkIndex()
        emb = HashingEmbedder()
        index.add("c", [C1], emb.embed_documents([C1.text]))
        hits = HybridRetriever(index, emb, RetrievalConfig(min_similarity=0.99)).retrieve("c", "zebra quantum")
        assert hits == []

    def test_skips_back_of_book_index_pages(self):
        index = MemoryChunkIndex()
        emb = HashingEmbedder()
        entry = Chunk(
            "d09-0900", "d09", 900, "Franklin, Josiah, born in Boston, 2; his family, 2, 3; candle maker, 4, 7."
        )
        index.add("c", [C1, entry], emb.embed_documents([C1.text, entry.text]))
        question = "Where was Josiah born, the candle maker?"
        assert [h.chunk.chunk_id for h in HybridRetriever(index, emb, RetrievalConfig()).retrieve("c", question)] == [
            "d01-0000"
        ]
        keep_all = RetrievalConfig(max_number_density=None)
        assert len(HybridRetriever(index, emb, keep_all).retrieve("c", question)) == 2

    def test_empty_corpus(self):
        assert HybridRetriever(MemoryChunkIndex(), HashingEmbedder(), RetrievalConfig()).retrieve("x", "q") == []


def answer_then(verdict: str):
    """A model that gives ``reply`` to the answer prompt and ``verdict`` to the evidence check."""
    return lambda reply: ScriptedChat(
        lambda m: verdict if m[0]["content"].startswith("You check whether quoted evidence") else reply
    )


@pytest.mark.parametrize(
    ("verdict", "ok"),
    [
        ("YES", True),
        ("Yes.", True),
        ("**YES**", True),
        ("yes, the quote names it", True),
        ("NO", False),
        ("", False),
        ("YESTERDAY", False),
        ("Not sure. YES", False),
    ],
)
def test_says_yes(verdict, ok):
    assert says_yes(verdict) is ok


class TestGeneration:
    def test_evidence_check_leaves_room_for_reasoning_tokens(self):
        budgets = []

        class Recording(ScriptedChat):
            def complete(self, messages, *, max_tokens=1200):
                budgets.append(max_tokens)
                return super().complete(messages, max_tokens=max_tokens)

        reply = 'EVIDENCE: "was born in Boston in 1706" [d01-0000]\nANSWER: Born in Boston [d01-0000].'
        chat = Recording(lambda m: "YES" if m[0]["content"].startswith("You check") else reply)
        answer, _ = QueryAnswerer(chat).answer(1, "X", "Where was he born?", [C1])
        assert answer.found and budgets[1] >= 200

    def test_nothing_retrieved_never_calls_llm(self):
        chat = ScriptedChat(lambda m: pytest.fail("LLM must not be called"))
        answer, exchange = QueryAnswerer(chat).answer(1, "X", "Q?", [])
        assert answer.answer is None and exchange.messages == []

    @pytest.mark.parametrize(
        "reply", ["NOT_IN_SOURCES", "not in sources.", "NOT_IN_SOURCES - the snippets are about printing"]
    )
    def test_not_in_sources_variants(self, reply):
        answer, _ = QueryAnswerer(ScriptedChat(lambda m: reply)).answer(1, "X", "Q?", [C1])
        assert not answer.found
        assert answer.display() == "Not in the sources."

    def test_answer_records_citations(self):
        reply = 'EVIDENCE: "was born in Boston in 1706" [d01-0000]\nANSWER: Born in Boston [d01-0000].'
        answer, ex = QueryAnswerer(answer_then("YES")(reply)).answer(1, "X", "Where was he born?", [C1, C2])
        assert answer.answer == "Born in Boston [d01-0000]."
        assert ex.check_messages[1]["content"].endswith('- "was born in Boston in 1706"') and ex.check_response == "YES"
        assert answer.cited_ids == ["d01-0000"]
        assert ex.evidence[0]["found"] and ex.gate_note is None
        assert answer.retrieved_ids == ["d01-0000", "d01-0001"]
        assert "[d01-0000]" in ex.messages[1]["content"]

    def test_answer_from_memory_with_loose_evidence_is_not_in_sources(self):
        # The canary failure seen in the real run: a famous fact, "supported" by a related snippet.
        reply = (
            'EVIDENCE: "the portrait on the hundred-dollar bill" [d01-0000]\n'
            "ANSWER: His portrait is on the hundred-dollar bill [d01-0000]."
        )
        answer, ex = QueryAnswerer(ScriptedChat(lambda m: reply)).answer(1, "X", "Q?", [C1])
        assert answer.display() == "Not in the sources."
        assert ex.response == reply and "occurs" in ex.gate_note  # the raw reply is still logged

    def test_real_but_irrelevant_quote_is_not_in_sources(self):
        # The canary reply from the first real run, after the quote gate was added: the quote is
        # genuine, it just says nothing about the bill.
        chunk = Chunk("d01-0018", "d01", 18, "BENJAMIN FRANKLIN From the original by Jean Baptiste Greuze.")
        reply = (
            'EVIDENCE: "BENJAMIN FRANKLIN From the original by Jean Baptiste Greuze." [d01-0018]\n'
            "ANSWER: The portrait on the hundred-dollar bill is of Benjamin Franklin. "
            "This is supported by the mention of his portrait [d01-0018]."
        )
        answer, ex = QueryAnswerer(answer_then("NO")(reply)).answer(
            1, "BF", "Whose portrait is on the $100 bill?", [chunk]
        )
        assert answer.display() == "Not in the sources." and answer.cited_ids == []
        assert ex.evidence[0]["found"] and ex.gate_note.startswith("the quoted evidence does not answer")
        assert "Greuze" in ex.check_messages[1]["content"] and "Snippets" not in ex.check_messages[1]["content"]

    def test_evidence_check_can_be_disabled(self):
        reply = 'EVIDENCE: "was born in Boston in 1706" [d01-0000]\nANSWER: Born in Boston [d01-0000].'
        chat = ScriptedChat(lambda m: reply)
        answer, ex = QueryAnswerer(chat, check_evidence=False).answer(1, "X", "Q?", [C1])
        assert answer.found and ex.check_messages == []

    def test_prompt_forbids_outside_knowledge_and_names_person(self):
        system = answer_messages("Ada Lovelace", "Q?", [C1])[0]["content"]
        assert "Ada Lovelace" in system and "outside knowledge" in system and "NOT_IN_SOURCES" in system

    def test_weave_skipped_when_nothing_found(self):
        from biography_rag.domain import QueryAnswer

        chat = ScriptedChat(lambda m: pytest.fail("LLM must not be called"))
        story, ex = StoryWeaver(chat).weave("X", [QueryAnswer(1, "Q", [], None, [])], [])
        assert story == "" and ex.messages == []


class TestStoryService:
    def ready(self, app_parts, bio_file):
        corpus = app_parts["ingest"].create("Benjamin Franklin", [bio_file])
        return app_parts["ingest"].ingest(corpus.corpus_id, [bio_file])

    def test_full_run_answers_twenty_and_logs_everything(self, app_parts, bio_file, chat):
        corpus = self.ready(app_parts, bio_file)
        svc = app_parts["story"]
        run = svc.execute(svc.start(corpus.corpus_id).run_id)

        assert run.status is RunStatus.DONE, run.error
        assert len(run.answers) == 20 and run.progress == 20
        assert run.answers[0].found and "Boston" in run.answers[0].answer
        assert any(not a.found for a in run.answers)  # a 9-paragraph corpus cannot answer everything
        assert run.story and run.verification is not None
        assert run.verification.flagged == []  # the obedient fake only quotes and cites

        events = app_parts["runs"].events(run.run_id)
        types = [e["type"] for e in events]
        assert types[0] == "run_started" and types[-1] == "run_finished"
        queries = [e for e in events if e["type"] == "query"]
        assert len(queries) == 20
        for e in queries:
            assert e["retrieved"] and all("chunk_id" in r for r in e["retrieved"])
            assert e["prompt"][0]["role"] == "system"  # the exact prompt is logged
        assert {"weave", "verification"} <= set(types)
        log_file = app_parts["log_dir"] / f"{run.run_id}.jsonl"
        assert len(log_file.read_text().splitlines()) == len(events)

    def test_canary_question_not_in_corpus(self, app_parts, bio_file):
        corpus = self.ready(app_parts, bio_file)
        run = app_parts["story"].ask(corpus.corpus_id, "Whose portrait is on the hundred-dollar bill?")
        assert run.status is RunStatus.DONE
        assert run.story == "Not in the sources."

    def test_hallucinating_model_is_flagged(self, app_parts, bio_file, chat):
        corpus = self.ready(app_parts, bio_file)
        chat.responder = lambda m: (
            '[{"i": 1, "supported": false, "reason": "date not in snippet"}]'
            if m[0]["content"].startswith("You audit")
            else "YES"
            if m[0]["content"].startswith("You check whether quoted evidence")
            else 'EVIDENCE: "was born in Boston on Milk Street" [d01-0000]\n'
            "ANSWER: He was born on January 17, 1706 in Boston [d01-0000]. He signed the Constitution."
        )
        run = app_parts["story"].ask(corpus.corpus_id, "When was he born?")
        # The gate drops the uncited sentence; the verifier flags the invented day.
        assert run.story == "He was born on January 17, 1706 in Boston [d01-0000]."
        assert [c.verdict for c in run.verification.checks] == [Verdict.JUDGE_FLAGGED]
        query = next(e for e in app_parts["runs"].events(run.run_id) if e["type"] == "query")
        assert "dropped 1 uncited" in query["gate"] and "Constitution" in query["response"]
        assert query["evidence_check"]["response"] == "YES" and query["evidence_check"]["prompt"]

    def test_run_failure_is_recorded(self, app_parts, bio_file, chat):
        corpus = self.ready(app_parts, bio_file)

        def boom(_m):
            raise RuntimeError("provider down")

        chat.responder = boom
        svc = app_parts["story"]
        run = svc.execute(svc.start(corpus.corpus_id).run_id)
        assert run.status is RunStatus.FAILED and "provider down" in run.error
        assert app_parts["runs"].events(run.run_id)[-2]["type"] == "run_failed"

    def test_cannot_run_on_unready_corpus(self, app_parts, bio_file):
        from biography_rag.services import ValidationError

        corpus = app_parts["ingest"].create("X", [bio_file])
        with pytest.raises(ValidationError, match="ingesting"):
            app_parts["story"].start(corpus.corpus_id)

    def test_recover_interrupted(self, app_parts, bio_file):
        corpus = self.ready(app_parts, bio_file)
        run = app_parts["story"].start(corpus.corpus_id)
        app_parts["ingest"].create("Y", [bio_file])
        assert app_parts["story"].recover_interrupted() == 2
        assert app_parts["runs"].get(run.run_id).status is RunStatus.FAILED
