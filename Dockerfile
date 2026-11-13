FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first so code changes don't bust the layer. No `local` extra:
# fastembed's ONNX model doesn't fit Render's 512 MB free instance, so production
# embeds with OpenAI.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs && chown -R appuser:appuser /app
USER appuser

# Two glibc arenas keep the thread pool's memory from fragmenting on 512 MB.
ENV PATH="/app/.venv/bin:$PATH" \
    LOG_DIR=/app/logs \
    MALLOC_ARENA_MAX=2

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', 8000))"

# One worker on purpose: background jobs run in this process's thread pool.
CMD ["biography-rag"]
