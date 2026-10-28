"""Typed application exceptions.

Using specific exception types (rather than raising bare HTTPException from
deep inside the service/tool layers) keeps business logic framework-agnostic
— the tools and orchestrator don't need to know they're running inside
FastAPI — while still mapping cleanly to HTTP responses at the edge.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all application-raised errors."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AuthenticationError(AppError):
    status_code = 401
    error_code = "unauthorized"


class NotFoundError(AppError):
    status_code = 404
    error_code = "not_found"


class PolicyNotFoundError(NotFoundError):
    error_code = "policy_not_found"


class ClaimNotFoundError(NotFoundError):
    error_code = "claim_not_found"


class ClaimValidationError(AppError):
    status_code = 422
    error_code = "claim_validation_error"


class ConflictError(AppError):
    status_code = 409
    error_code = "conflict"


class RunLimitExceededError(AppError):
    status_code = 429
    error_code = "daily_run_limit_reached"


class LLMProviderError(AppError):
    """Raised when the configured LLM provider fails or returns an
    unusable response after retries."""

    status_code = 502
    error_code = "llm_provider_error"


class ToolExecutionError(AppError):
    status_code = 500
    error_code = "tool_execution_error"

    def __init__(self, tool_name: str, message: str, *, details: dict | None = None) -> None:
        super().__init__(message, details=details)
        self.tool_name = tool_name


def register_exception_handlers(app) -> None:
    """Registers the AppError -> JSON response mapping on a FastAPI app.

    Factored out so both the real app (main.py) and tests building a
    minimal FastAPI app for a single router register identical error
    handling -- error responses shouldn't differ between environments.
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse

    from app.core.logging import get_logger

    logger = get_logger(__name__)

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"error_code": "internal_error", "message": "Unexpected server error.", "details": {}},
        )

    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "request_rejected",
            extra={"ctx_error_code": exc.error_code, "ctx_status": exc.status_code, "ctx_detail": exc.message},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error_code": exc.error_code, "message": exc.message, "details": exc.details},
        )
