"""One shared password in front of the whole site (HTTP Basic, any username).

It keeps the free model quotas from being spent by strangers; it is not per-user auth.
Health checks stay open so the host can tell the service is up."""

from __future__ import annotations

import base64
import binascii
import hmac

from starlette.types import ASGIApp, Receive, Scope, Send

OPEN_PATHS = frozenset({"/api/health"})
REALM = "Biography RAG"


def password_from(header: str) -> str | None:
    """The password in an `Authorization: Basic ...` header, or None if it is malformed."""
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic" or not encoded:
        return None
    try:
        decoded = base64.b64decode(encoded.strip(), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None
    _user, sep, password = decoded.partition(":")
    return password if sep else None


class SitePasswordMiddleware:
    def __init__(self, app: ASGIApp, password: str) -> None:
        self.app = app
        self._password = password.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in OPEN_PATHS or self._allowed(scope):
            await self.app(scope, receive, send)
            return
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"www-authenticate", f'Basic realm="{REALM}", charset="UTF-8"'.encode()),
                    (b"content-type", b"text/plain; charset=utf-8"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b"password required"})

    def _allowed(self, scope: Scope) -> bool:
        for name, value in scope["headers"]:
            if name == b"authorization":
                given = password_from(value.decode("latin-1"))
                return given is not None and hmac.compare_digest(given.encode(), self._password)
        return False
