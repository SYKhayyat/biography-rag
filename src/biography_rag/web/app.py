"""HTTP layer: thin FastAPI routes over the services. No business logic lives here."""

from __future__ import annotations

import hmac
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from biography_rag.rag.queries import QUERIES
from biography_rag.serialization import corpus_to_dict, run_to_dict
from biography_rag.services import NotFound, UploadedFile, ValidationError
from biography_rag.web.container import Container
from biography_rag.web.ratelimit import SlidingWindowLimiter

HERE = Path(__file__).parent


class RunRequest(BaseModel):
    provider: str | None = Field(default=None, max_length=40)


class AskRequest(RunRequest):
    question: str = Field(min_length=1, max_length=500)


def create_app(container: Container) -> FastAPI:
    settings = container.settings
    limiter = SlidingWindowLimiter(settings.runs_per_hour_per_client)
    templates = Jinja2Templates(directory=HERE / "templates")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        container.story.recover_interrupted()
        yield
        container.close()

    app = FastAPI(title="Biography RAG", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.exception_handler(ValidationError)
    async def _invalid(_r: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(NotFound)
    async def _missing(_r: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse({"detail": f"not found: {exc}"}, status_code=404)

    def rate_limited(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(
                429, "rate limit reached, try again later", headers={"Retry-After": str(limiter.retry_after(client))}
            )

    def require_admin(x_admin_token: Annotated[str, Header()] = "") -> None:
        if not settings.admin_token or not hmac.compare_digest(x_admin_token, settings.admin_token):
            raise HTTPException(403, "admin token required")

    def get_corpus(corpus_id: str):
        corpus = container.corpora.get(corpus_id)
        if corpus is None:
            raise NotFound(corpus_id)
        return corpus

    def get_run(run_id: str):
        run = container.runs.get(run_id)
        if run is None:
            raise NotFound(run_id)
        return run

    def options() -> dict[str, Any]:
        providers = [{"name": p.name, "label": p.label, "model": p.model} for p in container.chats.available()]
        try:
            default_provider = container.chats.default_provider
        except RuntimeError:
            default_provider = None
        names = container.embedders.names()
        return {
            "providers": providers,
            "default_provider": default_provider,
            "embedders": names,
            "default_embedder": container.embedders.default if names else None,
            "max_upload_mb": settings.max_upload_mb,
            "max_files": settings.max_files,
        }

    # --- pages -----------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index_page(request: Request) -> HTMLResponse:
        corpora = [corpus_to_dict(c) for c in container.corpora.list()]
        return templates.TemplateResponse(request, "index.html", {"corpora": corpora, "opts": options()})

    @app.get("/corpora/{corpus_id}", response_class=HTMLResponse)
    def corpus_page(request: Request, corpus_id: str) -> HTMLResponse:
        corpus = get_corpus(corpus_id)
        runs = [run_to_dict(r) for r in container.runs.list_for_corpus(corpus_id)]
        queries = [q.render(corpus.person) for q in QUERIES]
        ctx = {"corpus": corpus_to_dict(corpus), "runs": runs, "opts": options(), "queries": queries}
        return templates.TemplateResponse(request, "corpus.html", ctx)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(request: Request, run_id: str) -> HTMLResponse:
        return templates.TemplateResponse(request, "run.html", {"run": run_to_dict(get_run(run_id))})

    # --- API ---------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/options")
    def api_options() -> dict[str, Any]:
        return options()

    @app.get("/api/corpora")
    def list_corpora() -> list[dict[str, Any]]:
        return [corpus_to_dict(c) for c in container.corpora.list()]

    @app.post("/api/corpora", status_code=202, dependencies=[Depends(rate_limited)])
    async def create_corpus(
        person: Annotated[str, Form()],
        files: Annotated[list[UploadFile], File()],
        embedder: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        if len(files) > settings.max_files:
            raise ValidationError(f"at most {settings.max_files} files per corpus")
        budget = settings.max_upload_mb * 1024 * 1024
        uploads: list[UploadedFile] = []
        for f in files:
            data = await f.read(budget + 1)
            budget -= len(data)
            if budget < 0:
                raise HTTPException(413, f"uploads exceed {settings.max_upload_mb} MB in total")
            uploads.append(UploadedFile(Path(f.filename or "upload.txt").name, data))
        corpus = container.ingest.create(person, uploads, embedder or None)
        container.jobs.submit(container.ingest.ingest, corpus.corpus_id, uploads)
        return corpus_to_dict(container.corpora.get(corpus.corpus_id) or corpus)

    @app.get("/api/corpora/{corpus_id}")
    def corpus_detail(corpus_id: str) -> dict[str, Any]:
        data = corpus_to_dict(get_corpus(corpus_id))
        data["runs"] = [
            {k: v for k, v in run_to_dict(r).items() if k not in ("answers", "verification")}
            for r in container.runs.list_for_corpus(corpus_id)
        ]
        return data

    @app.delete("/api/corpora/{corpus_id}", status_code=204, dependencies=[Depends(require_admin)])
    def delete_corpus(corpus_id: str) -> Response:
        container.ingest.delete(corpus_id)
        return Response(status_code=204)

    @app.get("/api/corpora/{corpus_id}/chunks/{chunk_id}")
    def chunk(corpus_id: str, chunk_id: str) -> dict[str, Any]:
        found = container.index.get_many(get_corpus(corpus_id).corpus_id, [chunk_id])
        if chunk_id not in found:
            raise NotFound(chunk_id)
        c = found[chunk_id]
        return {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "ordinal": c.ordinal, "text": c.text}

    @app.post("/api/corpora/{corpus_id}/runs", status_code=202, dependencies=[Depends(rate_limited)])
    def start_run(corpus_id: str, body: RunRequest | None = None) -> dict[str, Any]:
        run = container.story.start(corpus_id, (body or RunRequest()).provider)
        container.jobs.submit(container.story.execute, run.run_id)
        return run_to_dict(container.runs.get(run.run_id) or run)

    @app.post("/api/corpora/{corpus_id}/ask", dependencies=[Depends(rate_limited)])
    def ask(corpus_id: str, body: AskRequest) -> dict[str, Any]:
        return run_to_dict(container.story.ask(corpus_id, body.question, body.provider))

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        return run_to_dict(get_run(run_id))

    @app.get("/api/runs/{run_id}/log")
    def run_log(run_id: str) -> Response:
        get_run(run_id)
        body = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in container.runs.events(run_id))
        headers = {"Content-Disposition": f'inline; filename="{run_id}.jsonl"'}
        return Response(body, media_type="application/x-ndjson", headers=headers)

    return app


def main() -> None:  # pragma: no cover - process entry point
    import logging
    import os

    import uvicorn

    from biography_rag.config import get_settings
    from biography_rag.web.container import build_container

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    uvicorn.run(
        create_app(build_container(settings)),
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        proxy_headers=True,
        forwarded_allow_ips="*",
        workers=1,  # background jobs live in this process
    )
