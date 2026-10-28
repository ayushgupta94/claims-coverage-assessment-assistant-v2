# Two stages: dependencies are resolved with uv from the lockfile in a builder
# image, and only the virtualenv + source are copied into a slim runtime image
# that runs as a non-root user.
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra tracing --no-install-project
COPY src/ ./src/
RUN uv sync --frozen --no-dev --extra tracing

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
COPY --chown=app:app data/ ./data/
COPY --chown=app:app scripts/ ./scripts/
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PORT=8000
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health/live')"

# Render injects PORT. uvicorn handles SIGTERM gracefully on shutdown.
CMD ["sh", "-c", "uvicorn app.main:create_app --factory --host 0.0.0.0 --port ${PORT}"]
