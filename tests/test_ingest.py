from __future__ import annotations

from dataclasses import replace

import pytest

from biography_rag.domain import CorpusStatus, SourceDocument
from biography_rag.ingest.chunker import ParagraphChunker
from biography_rag.ingest.loaders import UnsupportedFileType, extract_text, load_html, normalize, strip_boilerplate
from biography_rag.services import UploadedFile, ValidationError


class TestLoaders:
    def test_normalize_unwraps_lines_and_keeps_paragraphs(self):
        assert normalize("one\ntwo\n\n\nthree  four\r\n") == "one two\n\nthree four"

    def test_normalize_rejoins_hyphenated_line_breaks(self):
        assert normalize("electri-\ncity") == "electricity"

    def test_strips_gutenberg_licence(self):
        text = "licence\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nbody\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nmore licence"
        assert strip_boilerplate(text).strip() == "body"

    def test_html_keeps_paragraph_text_and_drops_scripts(self):
        html = b"<html><script>evil()</script><nav>menu</nav><p>First para.</p><p>Second.</p></html>"
        assert load_html(html) == "First para.\n\nSecond."

    def test_unknown_extension_rejected(self):
        with pytest.raises(UnsupportedFileType):
            extract_text("photo.jpg", b"\xff")

    def test_cp1252_text_decodes(self):
        assert extract_text("a.txt", "caf\xe9".encode("cp1252")) == "café"


class TestChunker:
    def doc(self, text: str) -> SourceDocument:
        return SourceDocument("d01", "t", text)

    def test_ids_are_stable_and_ordered(self):
        chunks = ParagraphChunker(size=40, overlap=0).chunk(
            self.doc("Alpha one. Beta two.\n\nGamma three. Delta four.")
        )
        assert [c.chunk_id for c in chunks] == [f"d01-{i:04d}" for i in range(len(chunks))]
        assert all(len(c.text) <= 40 for c in chunks)

    def test_no_text_is_lost(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(50))
        chunks = ParagraphChunker(size=120, overlap=0).chunk(self.doc(text))
        assert " ".join(c.text for c in chunks) == text

    def test_overlap_repeats_trailing_sentence(self):
        text = "Aaaa aaaa aaaa. Bbbb bbbb bbbb. Cccc cccc cccc. Dddd dddd dddd."
        chunks = ParagraphChunker(size=35, overlap=17).chunk(self.doc(text))
        assert len(chunks) >= 2
        last_sentence_of_first = chunks[0].text.split(". ")[-1].rstrip(".")
        assert chunks[1].text.startswith(last_sentence_of_first)

    def test_overlong_sentence_is_hard_split(self):
        chunks = ParagraphChunker(size=50, overlap=0).chunk(self.doc("word " * 60))
        assert all(len(c.text) <= 50 for c in chunks)

    @pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
    def test_rejects_bad_config(self, size, overlap):
        with pytest.raises(ValueError):
            ParagraphChunker(size=size, overlap=overlap)


class TestIngestService:
    def test_ingest_makes_corpus_ready(self, app_parts, bio_file):
        svc = app_parts["ingest"]
        corpus = svc.create("Benjamin  Franklin", [bio_file])
        assert corpus.person == "Benjamin Franklin"
        assert corpus.status is CorpusStatus.INGESTING
        done = svc.ingest(corpus.corpus_id, [bio_file])
        assert done.status is CorpusStatus.READY
        assert done.chunk_count > 3
        assert app_parts["corpora"].get(corpus.corpus_id).status is CorpusStatus.READY

    @pytest.mark.parametrize(
        "person,files,message",
        [
            ("", [UploadedFile("a.txt", b"x")], "name"),
            ("X", [], "at least one"),
            ("X", [UploadedFile("a.exe", b"x")], "unsupported"),
        ],
    )
    def test_create_validates(self, app_parts, person, files, message):
        with pytest.raises(ValidationError, match=message):
            app_parts["ingest"].create(person, files)

    def test_unknown_embedder_is_validation_error(self, app_parts, bio_file):
        with pytest.raises(ValidationError, match="unknown embedder"):
            app_parts["ingest"].create("X", [bio_file], embedder="nope")

    def test_empty_files_are_rejected_without_a_corpus(self, app_parts):
        with pytest.raises(ValidationError, match="empty"):
            app_parts["ingest"].create("X", [UploadedFile("blank.txt", b"   \n\n ")])
        assert app_parts["ingest"]._corpora.list() == []

    def test_files_without_extractable_text_fail_cleanly(self, app_parts):
        f = UploadedFile("blank.html", b"<html><script>var x = 1;</script></html>")
        corpus = app_parts["ingest"].create("X", [f])
        done = app_parts["ingest"].ingest(corpus.corpus_id, [f])
        assert done.status is CorpusStatus.FAILED
        assert "no text" in done.error

    @pytest.mark.parametrize(
        "name,data",
        [
            ("bin.txt", bytes(range(256)) * 12),  # random-looking bytes with NULs
            ("pic.html", b"\x89PNG\r\n\x1a\n" + b"\x01\x02\x03\x04" * 200),
            ("fake.pdf", b"just text, not a pdf"),
        ],
    )
    def test_binary_files_are_rejected(self, app_parts, name, data):
        ok = UploadedFile("bio.txt", b"Benjamin Franklin was born in Boston.")
        with pytest.raises(ValidationError, match="not a text file"):
            app_parts["ingest"].create("X", [ok, UploadedFile(name, data)])
        assert app_parts["ingest"]._corpora.list() == []

    def test_failed_corpora_are_purged_after_their_ttl(self, app_parts):
        from datetime import timedelta

        from biography_rag.domain import utcnow

        ingest = app_parts["ingest"]
        corpora = ingest._corpora
        f = UploadedFile("blank.html", b"<html></html>")
        old = ingest.ingest(ingest.create("Old", [f]).corpus_id, [f])
        corpora.save(replace(old, created_at=utcnow() - timedelta(hours=30)))
        recent = ingest.ingest(ingest.create("New", [f]).corpus_id, [f])
        assert ingest.purge_failed(24) == 1
        assert [c.corpus_id for c in corpora.list()] == [recent.corpus_id]
        assert ingest.purge_failed(0) == 0


def test_table_of_contents_link_lists_are_dropped():
    toc = "[Chapter 1](#ch1) [Chapter 2](#ch2) [Chapter 3](#ch3) [Chapter 4](#ch4)"
    text = normalize(f"# My Life\n\n{toc}\n\nI was born in a small town.")
    assert "Chapter" not in text and "My Life" in text and "small town" in text


def test_narration_detected_per_document(app_parts):
    memoir = UploadedFile("memoir.txt", b"I was born in Vilna. My father taught me to read, and I loved it. " * 20)
    bio = UploadedFile("bio.txt", b"Ann Lee was born in Vilna. Her father taught her to read. " * 20)
    ingest = app_parts["ingest"]
    auto = ingest.create("Ann Lee", [memoir, bio])
    assert ingest.ingest(auto.corpus_id, [memoir, bio]).first_person_docs == ["d01"]
    forced = ingest.create("Ann Lee", [memoir, bio], narration="others")
    assert ingest.ingest(forced.corpus_id, [memoir, bio]).first_person_docs == []
    with pytest.raises(ValidationError):
        ingest.create("Ann Lee", [memoir], narration="ghost")
