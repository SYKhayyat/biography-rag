from __future__ import annotations

import base64

import pytest

from biography_rag.web.auth import password_from
from tests.test_web import make_client

PASSWORD = "open-sesame"  # test-only; the real one lives in the host's environment


def basic(user: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


@pytest.fixture
def locked(tmp_path, chat):
    with make_client(tmp_path, chat, site_password=PASSWORD) as c:
        yield c


@pytest.mark.parametrize("path", ["/", "/api/options", "/static/app.js", "/api/corpora"])
def test_everything_needs_the_password(locked, path):
    res = locked.get(path)
    assert res.status_code == 401
    assert res.headers["www-authenticate"].startswith('Basic realm="Biography RAG"')


def test_right_password_any_username(locked):
    assert locked.get("/api/options", headers=basic("anyone", PASSWORD)).status_code == 200
    assert locked.get("/", headers=basic("", PASSWORD)).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        basic("x", PASSWORD.upper()),
        basic("x", PASSWORD + " "),
        {"Authorization": f"Bearer {PASSWORD}"},
        {"Authorization": "Basic !!"},
    ],
)
def test_wrong_credentials_are_refused(locked, headers):
    assert locked.get("/api/options", headers=headers).status_code == 401


def test_health_stays_open(locked):
    assert locked.get("/api/health").status_code == 200


def test_no_password_means_open(tmp_path, chat):
    with make_client(tmp_path, chat) as c:
        assert c.get("/api/options").status_code == 200


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Basic " + base64.b64encode(b"u:p:w").decode(), "p:w"),
        ("basic " + base64.b64encode(b"u:").decode(), ""),
        ("Basic " + base64.b64encode(b"nocolon").decode(), None),
        ("Digest abc", None),
        ("Basic", None),
    ],
)
def test_password_from(header, expected):
    assert password_from(header) == expected
