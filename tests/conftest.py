from __future__ import annotations

from pathlib import Path

import pytest

from biography_rag.ingest.chunker import ParagraphChunker
from biography_rag.rag.retriever import RetrievalConfig
from biography_rag.runlog import JsonlFileSink, RepositorySink, RunLogger
from biography_rag.services import IngestService, StoryConfig, StoryService, UploadedFile
from biography_rag.store.memory import MemoryChunkIndex, MemoryCorpusRepository, MemoryRunRepository
from tests.fakes import FakeChatSource, ScriptedChat, extractive_responder, fake_registry

BIO = """Benjamin Franklin was born in Boston on Milk Street in 1706.

His father, Josiah Franklin, was a tallow chandler and soap boiler who had emigrated from England.

As a boy he loved the sea and was a strong swimmer who led other boys in mischief.

He had many brothers and sisters; James, an older brother, was a printer.

He attended the Boston grammar school for less than a year and then a school for writing and arithmetic.

At ten he was taken from school to help his father cut wicks for candles, his first work.

He married Deborah Read in 1730 by common-law marriage.

His son Francis died of smallpox at the age of four, which grieved him deeply.

He flew a kite in a thunderstorm to show that lightning is electricity, and invented the lightning rod.
"""


@pytest.fixture
def bio_file() -> UploadedFile:
    return UploadedFile("franklin.txt", BIO.encode())


@pytest.fixture
def chat() -> ScriptedChat:
    return ScriptedChat(extractive_responder)


@pytest.fixture
def app_parts(tmp_path: Path, chat: ScriptedChat):
    corpora, index, runs = MemoryCorpusRepository(), MemoryChunkIndex(), MemoryRunRepository()
    embedders = fake_registry()
    logger = RunLogger([RepositorySink(runs), JsonlFileSink(tmp_path / "logs")])
    ingest = IngestService(corpora, index, embedders, ParagraphChunker(size=200, overlap=0))
    story = StoryService(
        corpora,
        index,
        runs,
        embedders,
        FakeChatSource(chat),
        logger,
        StoryConfig(RetrievalConfig(top_k=3, candidate_k=6), concurrency=2, verify_with_llm=True),
    )
    return {
        "corpora": corpora,
        "index": index,
        "runs": runs,
        "ingest": ingest,
        "story": story,
        "log_dir": tmp_path / "logs",
    }
