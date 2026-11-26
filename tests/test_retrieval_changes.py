"""The retrieval changes: filters before the limit, mean-over-arms fusion, the per-book cap,
neighbour windows, the optional reranker and query expansion."""

from __future__ import annotations

import pytest

from biography_rag.domain import Chunk
from biography_rag.rag.expansion import Expansion, parse_expansion
from biography_rag.rag.rerank import ChatReranker, NoReranker
from biography_rag.rag.retriever import HybridRetriever, RetrievalConfig, _choose
from biography_rag.store.base import SearchFilters
from biography_rag.store.memory import MemoryChunkIndex
from tests.fakes import HashingEmbedder, ScriptedChat


def build(docs: dict[str, list[str]], emb: HashingEmbedder | None = None) -> tuple[MemoryChunkIndex, HashingEmbedder]:
    """A corpus from {doc_id: [passage, ...]}, chunk ordinal = position in the list."""
    emb = emb or HashingEmbedder()
    index = MemoryChunkIndex()
    chunks = [Chunk(f"{doc}-{i:04d}", doc, i, text) for doc, texts in docs.items() for i, text in enumerate(texts)]
    index.add("c", chunks, emb.embed_documents([c.text for c in chunks]))
    return index, emb


class TestFilters:
    def test_index_pages_are_excluded_before_the_limit(self):
        """A page of numbers matches every question. Excluding it must not eat a slot: ask for 2
        and get 2 narrative passages, not 0."""
        narrative = [f"Francis was born in Boston in 1706, the fifteenth child (passage {w})." for w in "abcd"]
        index_page = "Franklin, Benjamin: 2, 7, 11, 14, 19, 22, 31; Abiah, 4, 9, 12; Josiah, 3, 8, 10, 17, 23."
        index, emb = build({"d01": [index_page, *narrative]})
        refilter = MemoryChunkIndex()
        cfg = RetrievalConfig(top_k=2, candidate_k=3, max_number_density=0.15)
        hits = HybridRetriever(index, emb, cfg).retrieve("c", "Where was Franklin born?")
        assert len(hits) == 2
        assert all("born in Boston" in h.chunk.text for h in hits)
        assert refilter is not None

    def test_cosine_floor_is_applied_before_the_limit(self):
        index, emb = build({"d01": ["A kite, a key and lightning.", "Deborah Read married him."]})
        floor = RetrievalConfig(top_k=2, candidate_k=2, min_similarity=0.99)
        assert HybridRetriever(index, emb, floor).retrieve("c", "Where was he born?") == []

    def test_the_memory_index_honours_filters_itself(self):
        index, emb = build({"d01": ["Kite and lightning, 2, 4, 6, 8, 10, 12, 14, 16, 18.", "A kite in a storm."]})
        filters = SearchFilters(min_similarity=0.0, max_number_density=0.15)
        hits = index.vector_search("c", emb.embed_query("kite lightning"), 2, filters)
        assert [h[0].text for h in hits] == ["A kite in a storm."]


class TestFusion:
    def test_one_strong_arm_is_not_outvoted_by_many_weak_ones(self):
        """The passage that one phrasing ranks first must beat the passage six phrasings rank
        twentieth — the sum-of-RRF behaviour that made multi-part questions drift."""
        passages = {f"p{i:02d}": [f"Passage {i} about {topic}."] for i, topic in enumerate(["kites"] * 6)}
        passages["first"] = ["Goldie and I were married in Jerusalem under a chuppah."]
        index, emb = build({"d01": [t[0] for t in passages.values()]})
        cfg = RetrievalConfig(top_k=2, candidate_k=10, neighbour_radius=0)
        retriever = HybridRetriever(index, emb, cfg)
        hits = retriever.retrieve(
            "c", "Kites about kites.", extra_phrasings=["about kites", "kites passage", "passage about"]
        )
        assert hits  # the weak passages still rank; the point is the ordering below
        top = retriever.retrieve("c", "Where were Goldie and the narrator married?", extra_phrasings=["wedding"])
        assert top[0].chunk.text.startswith("Goldie and I")


class TestDiversity:
    def test_the_cap_spreads_the_quota_over_books(self):
        books = ["one", "two", "three"]
        docs = {
            f"d0{n}": [f"A printer, a kite and lightning, from book {word}, {w}." for w in "abcd"]
            for n, word in enumerate(books, 1)
        }
        index, emb = build(docs)
        cfg = RetrievalConfig(top_k=4, candidate_k=10, max_per_doc=2, neighbour_radius=0)
        hits = HybridRetriever(index, emb, cfg).retrieve("c", "kite and lightning")
        per_book = {}
        for hit in hits:
            per_book[hit.chunk.doc_id] = per_book.get(hit.chunk.doc_id, 0) + 1
        assert max(per_book.values()) == 2

    def test_the_cap_never_starves_a_single_document(self):
        index, emb = build({"d01": [f"Benjamin Franklin, printer, passage {w}." for w in "abcdefgh"]})
        cfg = RetrievalConfig(top_k=4, candidate_k=10, max_per_doc=2, neighbour_radius=0)
        assert len(HybridRetriever(index, emb, cfg).retrieve("c", "Benjamin Franklin printer")) == 4

    def test_choose_takes_the_top_of_the_ranking(self):
        chunks = {f"d01-000{i}": Chunk(f"d01-000{i}", "d01", i, "text") for i in range(10)}
        assert _choose(list(chunks), chunks, 3, 0) == ["d01-0000", "d01-0001", "d01-0002"]


class TestSeams:
    """A hit that opens on a pronoun travels with a seam of what came before it — and only then.
    Measured on the Franklin corpus, a seam on every passage only made the prompt longer."""

    def test_a_passage_that_opens_on_a_pronoun_gets_the_sentence_before_it(self):
        index, emb = build(
            {"d01": ["Josiah had two wives. He was a tallow chandler.", "He bound Benjamin apprentice."]}
        )
        cfg = RetrievalConfig(top_k=1, candidate_k=3, neighbour_radius=1, seam_chars=240)
        hit = HybridRetriever(index, emb, cfg).retrieve("c", "He bound Benjamin apprentice")[0]
        assert (
            hit.chunk.text
            == "[...before] Josiah had two wives. He was a tallow chandler.\n\nHe bound Benjamin apprentice."
        )
        assert hit.chunk.chunk_id == "d01-0001"  # the id still points at the passage itself

    def test_a_passage_that_starts_a_sentence_is_left_alone(self):
        index, emb = build({"d01": ["Josiah was a chandler.", "Benjamin was bound apprentice to James."]})
        cfg = RetrievalConfig(top_k=1, candidate_k=3, neighbour_radius=1, seam_chars=240)
        hit = HybridRetriever(index, emb, cfg).retrieve("c", "Benjamin was bound apprentice to James")[0]
        assert hit.chunk.text == "Benjamin was bound apprentice to James."

    def test_the_seam_is_whole_sentences_and_bounded(self):
        before = "First sentence here. Second sentence here. Third sentence here and it is long."
        index, emb = build({"d01": [before, "He did the thing that matters."]})
        cfg = RetrievalConfig(top_k=1, candidate_k=3, neighbour_radius=1, seam_chars=30)
        hit = HybridRetriever(index, emb, cfg).retrieve("c", "He did the thing that matters")[0]
        assert hit.chunk.text == "[...before] Third sentence here and it is long.\n\nHe did the thing that matters."
        assert "e sentence" not in hit.chunk.text  # never a character-offset fragment

    def test_seams_never_cross_documents(self):
        index, emb = build({"d01": ["Book one first.", "He was in book one."], "d02": ["Book two first sentence."]})
        cfg = RetrievalConfig(top_k=1, candidate_k=5, neighbour_radius=1, seam_chars=240)
        hit = HybridRetriever(index, emb, cfg).retrieve("c", "He was in book one")[0]
        assert "Book two first sentence." not in hit.chunk.text

    def test_radius_zero_sends_the_chunk_alone(self):
        index, emb = build({"d01": ["First passage here.", "He did the second thing."]})
        cfg = RetrievalConfig(top_k=1, candidate_k=2, neighbour_radius=0)
        hit = HybridRetriever(index, emb, cfg).retrieve("c", "He did the second thing")[0]
        assert hit.chunk.text == "He did the second thing."

    def test_seam_chars_zero_sends_the_chunk_alone(self):
        index, emb = build({"d01": ["First passage here.", "He did the second thing."]})
        cfg = RetrievalConfig(top_k=1, candidate_k=2, neighbour_radius=1, seam_chars=0)
        hit = HybridRetriever(index, emb, cfg).retrieve("c", "He did the second thing")[0]
        assert hit.chunk.text == "He did the second thing."


class TestReranker:
    def _candidates(self):
        index, emb = build({"d01": [f"Passage {w} about printing." for w in "abcd"]})
        cfg = RetrievalConfig(top_k=4, candidate_k=4, neighbour_radius=0)
        return HybridRetriever(index, emb, cfg).retrieve("c", "printing")

    def test_no_reranker_keeps_the_fused_order(self):
        candidates = self._candidates()
        assert NoReranker().rerank("printing", candidates, 2) == candidates[:2]

    def test_a_ranking_is_followed(self):
        candidates = self._candidates()
        chat = ScriptedChat(lambda _m: "[4, 3]")
        picked = ChatReranker(chat).rerank("printing", candidates, 2)
        assert [p.chunk.chunk_id for p in picked] == [candidates[3].chunk.chunk_id, candidates[2].chunk.chunk_id]

    def test_a_failed_call_keeps_the_fused_order(self):
        def boom(_messages):
            raise RuntimeError("503")

        candidates = self._candidates()
        ranked = ChatReranker(ScriptedChat(boom)).rerank_logged("printing", candidates, 2)
        assert ranked.hits == candidates[:2]
        assert "rerank failed" in (ranked.note or "")

    def test_an_unusable_reply_keeps_the_fused_order(self):
        candidates = self._candidates()
        ranked = ChatReranker(ScriptedChat(lambda _m: "I cannot help with that")).rerank_logged(
            "printing", candidates, 2
        )
        assert ranked.hits == candidates[:2]
        assert ranked.note

    def test_out_of_range_numbers_are_ignored(self):
        candidates = self._candidates()
        picked = ChatReranker(ScriptedChat(lambda _m: "[9, 2, 2, 1]")).rerank("printing", candidates, 2)
        assert [p.chunk.chunk_id for p in picked] == [candidates[1].chunk.chunk_id, candidates[0].chunk.chunk_id]

    def test_the_retriever_uses_the_reranker_and_says_so(self):
        index, emb = build({"d01": [f"Passage {w} about printing." for w in "abcdef"]})
        cfg = RetrievalConfig(top_k=2, candidate_k=6, rerank_pool=6, neighbour_radius=0)
        chat = ScriptedChat(lambda _m: "[6, 5]")
        retriever = HybridRetriever(index, emb, cfg, ChatReranker(chat))
        hits, note = retriever.retrieve_logged("c", "printing")
        assert [h.chunk.ordinal for h in hits] == [5, 4]
        assert note and note.startswith("rerank order")


class TestExpansion:
    def test_both_halves_are_parsed(self):
        reply = (
            "PASSAGE: He was bound apprentice to his brother James, a printer, and learned the trade.\n"
            "PHRASINGS:\n- bound apprentice to a printer\n- learned the printing trade\n"
        )
        assert parse_expansion(reply) == Expansion(
            "He was bound apprentice to his brother James, a printer, and learned the trade.",
            ("bound apprentice to a printer", "learned the printing trade"),
        )

    def test_a_refusal_yields_nothing(self):
        assert parse_expansion("I'm sorry, I can't do that.") == Expansion()

    def test_the_passage_is_capped(self):
        found = parse_expansion("PASSAGE: " + "word " * 400)
        assert len(found.hypothetical) <= 900

    def test_the_hypothetical_text_is_searched_but_never_answered_from(self):
        """It is a query: an invented passage must not reach the snippets the model answers from."""
        index, emb = build({"d01": ["He was bound apprentice to his brother James."]})
        cfg = RetrievalConfig(top_k=1, candidate_k=2, neighbour_radius=0)
        retriever = HybridRetriever(index, emb, cfg)
        hits = retriever.retrieve("c", "What was his first job?", hypothetical="He was bound apprentice to James.")
        assert hits and hits[0].chunk.text == "He was bound apprentice to his brother James."

    def test_an_expansion_failure_leaves_the_search_unchanged(self):
        from biography_rag.rag.expansion import QueryExpander

        def boom(_messages):
            raise RuntimeError("429")

        found = QueryExpander(ScriptedChat(boom)).expand("Ann Lee", "Who was she?")
        assert found.hypothetical == "" and found.phrasings == ()
        assert "expansion failed" in (found.note or "")


class TestChunkContext:
    def test_a_heading_becomes_the_chunk_context(self):
        from biography_rag.domain import SourceDocument
        from biography_rag.ingest.chunker import ParagraphChunker

        text = "CHAPTER IV\n\nParis\n\n" + "He wrote letters home every week. " * 20
        chunks = ParagraphChunker(size=120, overlap=0).chunk(SourceDocument("d01", "Familiar Letters", text))
        assert chunks[0].context == "Familiar Letters — Paris"
        assert "CHAPTER" not in chunks[0].text  # the heading is context, not prose
        assert chunks[0].embed_input.startswith("Familiar Letters — Paris\n\n")

    def test_a_chunk_with_no_heading_still_names_its_book(self):
        from biography_rag.domain import SourceDocument
        from biography_rag.ingest.chunker import ParagraphChunker

        chunks = ParagraphChunker(size=200, overlap=0).chunk(
            SourceDocument("d01", "franklin.txt", "He was born in Boston. He liked the sea.")
        )
        assert chunks[0].context == "franklin.txt"

    @pytest.mark.parametrize(
        "paragraph,heading",
        [
            ("CHAPTER IV", True),
            ("PARIS", True),
            ("Letter the Second", True),
            ("He was born in Boston.", False),
            ("A sentence that simply runs on and on for a good many words without ever stopping", False),
            ("", False),
        ],
    )
    def test_heading_detection(self, paragraph, heading):
        from biography_rag.ingest.chunker import is_heading

        assert is_heading(paragraph) is heading


class TestEndToEndWithTheOptionalStages:
    def _story(self, tmp_path, **flags):
        from biography_rag.ingest.chunker import ParagraphChunker
        from biography_rag.runlog import JsonlFileSink, RepositorySink, RunLogger
        from biography_rag.services import IngestService, StoryConfig, StoryService
        from biography_rag.store.memory import MemoryChunkIndex, MemoryCorpusRepository, MemoryRunRepository
        from tests.fakes import FakeChatSource, extractive_responder, fake_registry

        corpora, index, runs = MemoryCorpusRepository(), MemoryChunkIndex(), MemoryRunRepository()
        embedders, chat = fake_registry(), ScriptedChat(extractive_responder)
        ingest = IngestService(corpora, index, embedders, ParagraphChunker(size=200, overlap=0))
        story = StoryService(
            corpora,
            index,
            runs,
            embedders,
            FakeChatSource(chat),
            RunLogger([RepositorySink(runs), JsonlFileSink(tmp_path / "logs")]),
            StoryConfig(RetrievalConfig(top_k=2, candidate_k=4, neighbour_radius=1), concurrency=2, **flags),
        )
        return ingest, story, chat

    def _run(self, tmp_path, bio_file, **flags):
        ingest, story, chat = self._story(tmp_path, **flags)
        corpus = ingest.create("Benjamin Franklin", [bio_file])
        ingest.ingest(corpus.corpus_id, [bio_file])
        run = story.execute(story.start(corpus.corpus_id).run_id)
        return run, chat

    def test_a_run_finishes_with_reranking_and_expansion_on(self, tmp_path, bio_file):
        from biography_rag.domain import RunStatus

        run, chat = self._run(tmp_path, bio_file, rerank=True, expand_queries=True)
        assert run.status is RunStatus.DONE, run.error
        systems = [c[0]["content"][:40] for c in chat.calls]
        assert any(s.startswith("You prepare search queries") for s in systems)
        assert any(s.startswith("You rank passages") for s in systems)

    def test_the_invented_passage_never_reaches_the_answer_prompt(self, tmp_path, bio_file):
        """HyDE text is a query. If it leaked into the snippets, the model could quote it as
        evidence for a fact no book states."""
        _, chat = self._run(tmp_path, bio_file, expand_queries=True)
        expansions = [c for c in chat.calls if c[0]["content"].startswith("You prepare search queries")]
        assert expansions
        answers = [c for c in chat.calls if c[0]["content"].startswith("You answer one question")]
        for call in answers:
            assert "A biography would say that this relates to" not in call[-1]["content"]

    def test_the_defaults_ask_for_neither_stage(self, tmp_path, bio_file):
        _, chat = self._run(tmp_path, bio_file)
        systems = [c[0]["content"][:40] for c in chat.calls]
        assert not any(s.startswith("You prepare search queries") for s in systems)
        assert not any(s.startswith("You rank passages") for s in systems)
