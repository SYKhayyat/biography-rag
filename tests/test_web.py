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
from tests.fakes import FakeChatSource, ScriptedChat, extractive_responder, fake_registry


def make_client(tmp_path, chat, *more_chats, **overrides) -> TestClient:
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
        chats=FakeChatSource(chat, *more_chats),
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


def test_public_api_accepts_only_listed_models(client):
    corpus_id = upload(client).json()["corpus_id"]
    res = client.post(f"/api/corpora/{corpus_id}/runs", json={"provider": "fake", "model": "o1-pro"})
    assert res.status_code == 422 and "not offered" in res.json()["detail"]
    res = client.post(f"/api/corpora/{corpus_id}/runs", json={"provider": "fake", "model": "fake-1"})
    assert res.status_code == 202 and res.json()["model"] == "fake-1"


@pytest.fixture
def two_model_client(tmp_path, chat):
    other = ScriptedChat(extractive_responder, provider="other", model="other-9")
    with make_client(tmp_path, chat, other) as c:
        yield c


def test_compare_story_across_models(two_model_client):
    client = two_model_client
    corpus_id = upload(client).json()["corpus_id"]
    targets = [{"provider": "fake"}, {"provider": "other", "model": "other-9"}]
    res = client.post(f"/api/corpora/{corpus_id}/compare", json={"kind": "story", "targets": targets})
    assert res.status_code == 202
    cmp_id = res.json()["comparison_id"]
    assert cmp_id and len(res.json()["run_ids"]) == 2

    data = client.get(f"/api/corpora/{corpus_id}/comparisons/{cmp_id}").json()
    assert data["kind"] == "story"
    assert [(r["provider"], r["model"], r["status"]) for r in data["runs"]] == [
        ("fake", "fake-1", "done"),
        ("other", "other-9", "done"),
    ]
    assert all(r["comparison_id"] == cmp_id and r["story"] for r in data["runs"])
    page = client.get(f"/corpora/{corpus_id}/compare/{cmp_id}")
    assert page.status_code == 200 and "other-9" in page.text


def test_compare_one_question(two_model_client):
    client = two_model_client
    corpus_id = upload(client).json()["corpus_id"]
    body = {
        "kind": "ask",
        "question": "Whose face is on the $100 bill?",
        "targets": [{"provider": "fake"}, {"provider": "other"}],
    }
    cmp_id = client.post(f"/api/corpora/{corpus_id}/compare", json=body).json()["comparison_id"]
    runs = client.get(f"/api/corpora/{corpus_id}/comparisons/{cmp_id}").json()["runs"]
    assert {r["story"] for r in runs} == {"Not in the sources."}
    assert {r["question"] for r in runs} == {"Whose face is on the $100 bill?"}


def test_single_target_compare_is_a_plain_run(two_model_client):
    client = two_model_client
    corpus_id = upload(client).json()["corpus_id"]
    res = client.post(f"/api/corpora/{corpus_id}/compare", json={"kind": "story", "targets": [{"provider": "other"}]})
    assert res.status_code == 202 and res.json()["comparison_id"] is None
    assert client.get(f"/api/runs/{res.json()['run_ids'][0]}").json()["status"] == "done"


@pytest.mark.parametrize(
    "body",
    [
        {"kind": "story", "targets": []},
        {"kind": "story", "targets": [{"provider": "fake"}, {"provider": "fake", "model": "fake-1"}]},
        {"kind": "story", "targets": [{"provider": "fake"}, {"provider": "other", "model": "secret-model"}]},
        {"kind": "ask", "targets": [{"provider": "fake"}]},
        {"kind": "poem", "targets": [{"provider": "fake"}]},
    ],
    ids=["no-models", "duplicate", "unlisted-model", "ask-without-question", "bad-kind"],
)
def test_compare_rejects_bad_requests_and_queues_nothing(two_model_client, body):
    client = two_model_client
    corpus_id = upload(client).json()["corpus_id"]
    assert client.post(f"/api/corpora/{corpus_id}/compare", json=body).status_code == 422
    assert client.get(f"/api/corpora/{corpus_id}").json()["runs"] == []


def test_compare_caps_targets(tmp_path, chat):
    more = [ScriptedChat(extractive_responder, provider=f"p{i}", model="m") for i in range(3)]
    with make_client(tmp_path, chat, *more, max_compare_targets=3) as client:
        corpus_id = upload(client).json()["corpus_id"]
        targets = [{"provider": p} for p in ("fake", "p0", "p1", "p2")]
        res = client.post(f"/api/corpora/{corpus_id}/compare", json={"kind": "story", "targets": targets})
        assert res.status_code == 422 and "1 to 3" in res.json()["detail"]


def test_compare_costs_one_rate_limit_hit_per_model(tmp_path, chat):
    other = ScriptedChat(extractive_responder, provider="other", model="other-9")
    with make_client(tmp_path, chat, other, runs_per_hour_per_client=3) as client:
        corpus_id = upload(client).json()["corpus_id"]  # hit 1
        body = {"kind": "story", "targets": [{"provider": "fake"}, {"provider": "other"}]}
        assert client.post(f"/api/corpora/{corpus_id}/compare", json=body).status_code == 202  # hits 2-3
        assert client.post(f"/api/corpora/{corpus_id}/compare", json=body).status_code == 429


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
    assert opts["providers"][0]["models"] == [{"id": "fake-1", "label": "fake-1", "slow": False}]
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

    def test_cost_is_all_or_nothing(self):
        limiter = SlidingWindowLimiter(3, window_seconds=10, clock=lambda: 0.0)
        assert limiter.allow("a", 2)
        assert not limiter.allow("a", 2)  # would exceed: nothing recorded
        assert limiter.allow("a", 1)
