"""The LangGraph workflow, driven through the service: background assessment
with live progress, auto-finalize, pause for human review, resume (including
after a simulated restart), failures, and validation."""

import asyncio
from datetime import UTC, datetime

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from mongomock_motor import AsyncMongoMockClient

from app.agent.orchestrator import ClaimAssessmentOrchestrator
from app.config import Settings
from app.core.exceptions import ConflictError, PolicyNotFoundError, RunLimitExceededError, ToolExecutionError
from app.db.repositories.claim_repository import ClaimRepository
from app.db.repositories.policy_repository import PolicyRepository
from app.db.repositories.usage_repository import UsageRepository
from app.domain.models import ClaimStatus, CoverageOutcome, IssuedPolicy, ReviewRequest
from app.services.claim_assessment_service import ClaimAssessmentService
from app.workflow.graph import build_claim_workflow
from tests.fakes import CLEAN_TOOL_RESULTS, FakeLLM, FakeToolExecutor, final_answer, make_claim, tool_call

SETTINGS = Settings(_env_file=None, openai_api_key="unused", max_runs_per_day=100)


@pytest.fixture
async def env():
    db = AsyncMongoMockClient()["test_workflow"]
    policies, claims = PolicyRepository(db), ClaimRepository(db)
    await claims.ensure_indexes()
    await policies.upsert_issued_policy(
        IssuedPolicy(
            policy_id="POL-1",
            customer_id="CUST-1",
            product_version_id="AUTO-GOLD-V1",
            sum_insured=1_000_000,
            inception_date=datetime(2023, 1, 1, tzinfo=UTC),
        )
    )
    return {"db": db, "policies": policies, "claims": claims, "checkpointer": InMemorySaver()}


def make_service(env, tool_results=CLEAN_TOOL_RESULTS, settings=SETTINGS, tools=None) -> ClaimAssessmentService:
    script = [
        tool_call("retrieve_policy_clauses", {"query": "collision", "policy_id": "POL-1"}),
        tool_call("check_coverage_rules", {"claim_id": "CLAIM-1"}),
        tool_call("score_fraud_risk", {"claim_id": "CLAIM-1"}),
        final_answer(),
    ]
    orchestrator = ClaimAssessmentOrchestrator(FakeLLM(script), tools or (lambda: FakeToolExecutor(tool_results)))
    workflow = build_claim_workflow(
        orchestrator=orchestrator, claim_repository=env["claims"], settings=settings, checkpointer=env["checkpointer"]
    )
    return ClaimAssessmentService(
        workflow=workflow,
        claim_repository=env["claims"],
        policy_repository=env["policies"],
        usage_repository=UsageRepository(env["db"]),
        settings=settings,
    )


async def submit_and_wait(service: ClaimAssessmentService):
    await service.submit(make_claim())
    await service.drain()
    return await service.get("CLAIM-1")


class SlowToolExecutor(FakeToolExecutor):
    """Tools that take a little time, like real network calls."""

    async def execute(self, name: str, arguments: dict) -> dict:
        await asyncio.sleep(0.05)
        return await super().execute(name, arguments)


async def test_clean_claim_is_auto_approved(env):
    record = await submit_and_wait(make_service(env))

    assert record.status == ClaimStatus.APPROVED
    assert record.decision.requires_human_review is False
    assert [h.action for h in record.decision_history] == ["assessed", "auto_approved"]
    assert [s.tool for s in record.steps] == ["retrieve_policy_clauses", "check_coverage_rules", "score_fraud_risk"]
    assert record.usage.llm_calls == 5


RISKY = {**CLEAN_TOOL_RESULTS, "score_fraud_risk": {"risk_level": "medium", "risk_score": 0.35, "signals": ["x"]}}


async def test_risky_claim_pauses_for_review_and_resumes_after_restart(env):
    record = await submit_and_wait(make_service(env, RISKY))
    assert record.status == ClaimStatus.UNDER_REVIEW
    assert [h.action for h in record.decision_history] == ["assessed", "escalated"]

    # A brand-new service + graph (as after a restart) resumes from the checkpoint.
    review = ReviewRequest(action="approve", reason="Receipts verified", reviewer="Asha")
    record = await make_service(env, RISKY).review("CLAIM-1", review)

    assert record.status == ClaimStatus.APPROVED
    assert record.decision_history[-1].action == "approved"
    assert record.decision_history[-1].actor == "Asha"


async def test_reviewer_override_changes_the_outcome(env):
    service = make_service(env, RISKY)
    await submit_and_wait(service)
    review = ReviewRequest(action="override", outcome="not_covered", reason="Staged accident", reviewer="Ravi")
    record = await service.review("CLAIM-1", review)

    assert record.status == ClaimStatus.REJECTED
    assert record.decision.coverage_outcome == CoverageOutcome.NOT_COVERED
    assert record.decision_history[-1].action == "overridden"


async def test_review_is_only_allowed_while_under_review(env):
    service = make_service(env)
    await submit_and_wait(service)
    with pytest.raises(ConflictError):
        await service.review("CLAIM-1", ReviewRequest(action="approve", reason="ok ok", reviewer="A"))


async def test_submission_validation(env):
    service = make_service(env)
    with pytest.raises(PolicyNotFoundError):
        await service.submit(make_claim(policy_id="NOPE"))
    await service.submit(make_claim())
    with pytest.raises(ConflictError):
        await service.submit(make_claim())


async def test_daily_cap_blocks_new_runs(env):
    capped = Settings(_env_file=None, openai_api_key="unused", max_runs_per_day=0)
    with pytest.raises(RunLimitExceededError):
        await make_service(env, settings=capped).submit(make_claim())


async def test_submit_returns_at_once_and_steps_are_visible_while_running(env):
    service = make_service(env, tools=lambda: SlowToolExecutor(CLEAN_TOOL_RESULTS))

    record = await service.submit(make_claim())
    assert record.status == ClaimStatus.SUBMITTED
    assert record.decision is None

    for _ in range(100):  # poll like a client would
        record = await service.get("CLAIM-1")
        if record.steps:
            break
        await asyncio.sleep(0.01)
    assert record.status == ClaimStatus.SUBMITTED
    assert record.steps[0].tool == "retrieve_policy_clauses"

    await service.drain()
    record = await service.get("CLAIM-1")
    assert record.status == ClaimStatus.APPROVED
    assert len(record.steps) == 3  # the live steps were replaced, not duplicated


async def test_a_failing_assessment_marks_the_claim_failed(env):
    class BrokenToolExecutor(FakeToolExecutor):
        async def execute(self, name: str, arguments: dict) -> dict:
            raise ToolExecutionError(name, "tool backend down")

    record = await submit_and_wait(make_service(env, tools=lambda: BrokenToolExecutor({})))

    assert record.status == ClaimStatus.FAILED
    assert record.decision_history[-1].action == "failed"
    assert "could not be completed" in record.decision_history[-1].reason


async def test_an_assessment_cut_off_by_shutdown_is_resumed_on_the_next_start(env):
    service = make_service(env, tools=lambda: SlowToolExecutor(CLEAN_TOOL_RESULTS))
    await service.submit(make_claim())
    await asyncio.sleep(0.02)
    await service.drain(timeout=0)  # shutdown: cancel what is still running

    assert (await service.get("CLAIM-1")).status == ClaimStatus.SUBMITTED

    restarted = make_service(env)  # same database and checkpoints, new process
    assert await restarted.resume_interrupted() == 1
    await restarted.drain()

    record = await restarted.get("CLAIM-1")
    assert record.status == ClaimStatus.APPROVED
    assert len(record.steps) == 3
    assert [h.action for h in record.decision_history] == ["requeued", "assessed", "auto_approved"]
