from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from biography_rag.config import Settings
from biography_rag.jobs import InlineJobRunner
from biography_rag.store.memory import MemoryChunkIndex, MemoryCorpusRepository, MemoryRunRepository
from biography_rag.web.app import create_app
from biography_rag.web.container import build_container
from biography_rag.web.ratelimit import SlidingWindowLimiter
from tests.conftest import BIO
from tests.fakes import FakeChatSource, fake_registry


def make_client(tmp_path, chat, **overrides) -> TestClient:
    settings = Settings(
        _env_file=None,
        log_dir=str(tmp_path / "logs"),
        chunk_size=200,
        chunk_overlap=0,
        admin_token="s3cret",
        **overrides,
    )
    container = build_container(
        settings,
        embedders=fake_registry(),
        chats=FakeChatSource(chat),
        jobs=InlineJobRunner(),
        stores=(MemoryCorpusRepository(), MemoryChunkIndex(), MemoryRunRepository()),
    )
    return TestClient(create_app(container))


@pytest.fixture
def client(tmp_path, chat):
    with make_client(tmp_path, chat) as c:
        yield c


def upload(client, person="Benjamin Franklin", files=None):
    files = files or [("files", ("franklin.txt", BIO.encode(), "text/plain"))]
    return client.post("/api/corpora", data={"person": person}, files=files)


def test_upload_ingests_and_lists(client):
    res = upload(client)
    assert res.status_code == 202
    corpus_id = res.json()["corpus_id"]
    detail = client.get(f"/api/corpora/{corpus_id}").json()
    assert detail["status"] == "ready" and detail["chunk_count"] > 3
    assert [c["corpus_id"] for c in client.get("/api/corpora").json()] == [corpus_id]
    assert "Benjamin Franklin" in client.get("/").text


def test_upload_rejects_bad_input(client):
    assert upload(client, person="  ").status_code == 422
    bad = [("files", ("x.exe", b"MZ", "application/octet-stream"))]
    assert "unsupported" in upload(client, files=bad).json()["detail"]


def test_upload_size_limit(tmp_path, chat):
    with make_client(tmp_path, chat, max_upload_mb=1) as c:
        big = [("files", ("big.txt", b"a" * (1024 * 1024 + 10), "text/plain"))]
        assert upload(c, files=big).status_code == 413


def test_story_run_end_to_end(client):
    corpus_id = upload(client).json()["corpus_id"]
    res = client.post(f"/api/corpora/{corpus_id}/runs", json={})
    assert res.status_code == 202
    run_id = res.json()["run_id"]

    run = client.get(f"/api/runs/{run_id}").json()
    assert run["status"] == "done" and len(run["answers"]) == 20
    assert run["verification"]["traced_ratio"] == 1.0
    assert client.get(f"/runs/{run_id}").status_code == 200
    assert run_id in client.get(f"/corpora/{corpus_id}").text

    log = client.get(f"/api/runs/{run_id}/log")
    assert log.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in log.text.splitlines()]
    assert sum(e["type"] == "query" for e in events) == 20


def test_chunk_lookup_for_citation_chips(client):
    corpus_id = upload(client).json()["corpus_id"]
    chunk = client.get(f"/api/corpora/{corpus_id}/chunks/d01-0000").json()
    assert "Boston" in chunk["text"]
    assert client.get(f"/api/corpora/{corpus_id}/chunks/d99-0000").status_code == 404


def test_canary_ask(client):
    corpus_id = upload(client).json()["corpus_id"]
    run = client.post(f"/api/corpora/{corpus_id}/ask", json={"question": "Whose face is on the hundred-dollar bill?"})
    assert run.status_code == 200
    assert run.json()["story"] == "Not in the sources."


def test_public_api_cannot_choose_arbitrary_model(client):
    corpus_id = upload(client).json()["corpus_id"]
    res = client.post(f"/api/corpora/{corpus_id}/runs", json={"provider": "fake", "model": "o1-pro"})
    assert res.status_code == 202
    assert res.json()["model"] == "fake-1"  # the extra field is ignored; models come from server config


def test_unknown_provider_and_missing_things(client):
    corpus_id = upload(client).json()["corpus_id"]
    assert client.post(f"/api/corpora/{corpus_id}/runs", json={"provider": "nope"}).status_code == 422
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/corpora/nope").status_code == 404
    assert client.post("/api/corpora/nope/ask", json={"question": "Q?"}).status_code == 404


def test_delete_requires_admin_token(client):
    corpus_id = upload(client).json()["corpus_id"]
    assert client.delete(f"/api/corpora/{corpus_id}").status_code == 403
    assert client.delete(f"/api/corpora/{corpus_id}", headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert client.delete(f"/api/corpora/{corpus_id}", headers={"X-Admin-Token": "s3cret"}).status_code == 204
    assert client.get(f"/api/corpora/{corpus_id}").status_code == 404


def test_rate_limit_on_paid_endpoints(tmp_path, chat):
    with make_client(tmp_path, chat, runs_per_hour_per_client=2) as c:
        corpus_id = upload(c).json()["corpus_id"]  # 1
        assert c.post(f"/api/corpora/{corpus_id}/ask", json={"question": "Q?"}).status_code == 200  # 2
        res = c.post(f"/api/corpora/{corpus_id}/ask", json={"question": "Q?"})
        assert res.status_code == 429 and int(res.headers["Retry-After"]) > 0
        assert c.get("/api/health").status_code == 200  # reads are never limited


def test_options_lists_providers_and_embedders(client):
    opts = client.get("/api/options").json()
    assert opts["providers"][0]["name"] == "fake"
    assert opts["default_embedder"] in opts["embedders"]


class TestSlidingWindowLimiter:
    def test_window_slides(self):
        now = [0.0]
        limiter = SlidingWindowLimiter(2, window_seconds=10, clock=lambda: now[0])
        assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
        assert limiter.allow("b")  # per client
        now[0] = 10.0
        assert limiter.allow("a")

    def test_zero_disables(self):
        limiter = SlidingWindowLimiter(0)
        assert all(limiter.allow("a") for _ in range(100))
