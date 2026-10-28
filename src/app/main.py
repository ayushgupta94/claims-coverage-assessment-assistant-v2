"""Application factory: one process, one container, two surfaces (REST + MCP).

    uvicorn app.main:create_app --factory --port 8000

Builds the object graph explicitly in one place:
    Mongo -> repositories -> embedder -> hybrid retriever -> MCP server (/mcp)
    LLM -> orchestrator (MCP client) -> LangGraph workflow (+ checkpointer) -> service
Anything can be injected (tests pass fakes / in-memory versions).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.agent.llm_client import OpenAIChatClient
from app.agent.orchestrator import ClaimAssessmentOrchestrator, ToolExecutor
from app.api.routes import claims, health, reviews
from app.api.security import AccessCodeMiddleware
from app.config import Settings, get_settings
from app.core import tracing
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger, request_context_middleware, request_id_var
from app.db.mongo_client import MongoDatabase
from app.db.repositories.claim_repository import ClaimRepository
from app.db.repositories.policy_repository import PolicyRepository
from app.db.repositories.usage_repository import UsageRepository
from app.mcp_server.server import build_mcp_server
from app.retrieval.embeddings import Embedder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.services.claim_assessment_service import ClaimAssessmentService
from app.workflow.graph import build_claim_workflow

STATIC_DIR = Path(__file__).resolve().parent / "static"

logger = get_logger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    mongo_database=None,
    llm=None,
    embedder=None,
    checkpointer=None,
    tool_executor_factory=None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    mongo_database = mongo_database or MongoDatabase(settings)
    policy_repository = PolicyRepository(mongo_database.db)
    claim_repository = ClaimRepository(mongo_database.db)
    retriever = HybridRetriever(policy_repository, embedder or Embedder(settings), settings)
    mcp_server = build_mcp_server(
        policy_repository=policy_repository, claim_repository=claim_repository, retriever=retriever, settings=settings
    )

    def new_tool_executor() -> ToolExecutor:
        # The agent authenticates to its own /mcp like any client, and forwards
        # the request id so tool-side logs correlate with the request.
        codes = settings.access_code_list
        headers = {"X-Access-Code": codes[0] if codes else "", "X-Request-ID": request_id_var.get() or ""}
        return ToolExecutor(settings.mcp_server_url, headers)

    orchestrator = ClaimAssessmentOrchestrator(
        llm or OpenAIChatClient(settings), tool_executor_factory or new_tool_executor
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await policy_repository.ensure_indexes()
        await claim_repository.ensure_indexes()
        workflow = build_claim_workflow(
            orchestrator=orchestrator,
            claim_repository=claim_repository,
            settings=settings,
            checkpointer=checkpointer or _mongo_checkpointer(settings),
        )
        app.state.mongo_database = mongo_database
        app.state.claim_service = ClaimAssessmentService(
            workflow=workflow,
            claim_repository=claim_repository,
            policy_repository=policy_repository,
            usage_repository=UsageRepository(mongo_database.db),
            settings=settings,
        )
        logger.info("startup_complete", extra={"ctx_environment": settings.environment})
        async with mcp_server.session_manager.run():
            yield
        tracing.flush()
        mongo_database.close()
        logger.info("shutdown_complete")

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Assesses insurance claims with a tool-calling agent over MCP, hybrid RAG on policy wording, "
            "a code-enforced decision policy and human-in-the-loop review."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.middleware("http")(request_context_middleware)
    register_exception_handlers(app)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.include_router(health.router)
    app.include_router(claims.router)
    app.include_router(reviews.router)
    app.mount("/mcp", AccessCodeMiddleware(mcp_server.streamable_http_app(), settings))
    return app


def _mongo_checkpointer(settings: Settings):
    """Workflow state for paused (human-review) claims lives in MongoDB, so it
    survives restarts and Render's sleep. Uses a sync pymongo client, which
    the saver wraps in a thread for async calls."""
    from langgraph.checkpoint.mongodb import MongoDBSaver
    from pymongo import MongoClient

    return MongoDBSaver(
        MongoClient(settings.mongo_uri),
        db_name=settings.mongo_db_name,
        checkpoint_collection_name="workflow_checkpoints",
        writes_collection_name="workflow_checkpoint_writes",
    )
