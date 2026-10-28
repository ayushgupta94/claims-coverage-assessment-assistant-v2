"""Structured JSON logging with request correlation.

Every log line is one JSON object on stdout, so Render / Azure / any container
log collector can index it without a parser. Three conventions:

- Context: pass structured fields as `extra={"ctx_<name>": value}`; they are
  emitted as top-level keys (`ctx_tool` -> `"tool"`).
- Correlation: `request_id` and `claim_id` live in contextvars and are added to
  every line automatically, so one request's lines can be grepped together --
  including the MCP tool calls it triggers (the agent forwards X-Request-ID).
- Privacy: claim descriptions are never logged (callers log IDs and sizes),
  and emails / phone numbers are redacted as a last line of defence.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
claim_id_var: ContextVar[str | None] = ContextVar("claim_id", default=None)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# 10-digit numbers, optionally with a country code ("+91-9876543210", "98765 43210").
# Dates ("2024-06-01") and amounts don't match.
_PHONE = re.compile(r"(?<![\d-])(?:\+\d{1,3}[\s-]?)?\d{5}[\s-]?\d{5}(?![\d-])")


def redact(value: object) -> object:
    if isinstance(value, str):
        return _PHONE.sub("[redacted-phone]", _EMAIL.sub("[redacted-email]", value))
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        if request_id := request_id_var.get():
            payload["request_id"] = request_id
        if claim_id := claim_id_var.get():
            payload["claim_id"] = claim_id
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                payload[key[4:]] = redact(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()  # idempotent under uvicorn --reload and in tests

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    # Our own middleware logs every request; uvicorn's access log would duplicate it.
    logging.getLogger("uvicorn.access").disabled = True
    for noisy in ("pymongo", "motor", "httpx", "httpcore", "openai", "mcp"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


_request_logger = get_logger("app.request")


async def request_context_middleware(request, call_next):
    """Assigns a request_id (or reuses an incoming X-Request-ID), logs one
    line per request with status and latency, and echoes the id back."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request_id_var.set(request_id)
    claim_id_var.set(None)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        _request_logger.exception("request_failed", extra={"ctx_method": request.method, "ctx_path": request.url.path})
        raise
    _request_logger.info(
        "request_completed",
        extra={
            "ctx_method": request.method,
            "ctx_path": request.url.path,
            "ctx_status": response.status_code,
            "ctx_latency_ms": round((time.perf_counter() - started) * 1000),
        },
    )
    response.headers["X-Request-ID"] = request_id
    return response
