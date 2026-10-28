"""Claim use cases: submit a claim for assessment, and review an escalated one.

Kept free of FastAPI so the same use cases can be driven from HTTP, a CLI or
the eval runner. Each claim is one workflow thread (thread_id = claim_id).

Submission is asynchronous: `submit` validates, stores the claim as
`submitted` and starts the assessment as a background task, so the caller gets
an answer in milliseconds and polls the claim for the result. A run that
raises marks the claim `failed`; a run cut off by a shutdown leaves it
`submitted`, and `resume_interrupted` runs it again on the next startup.
"""

from __future__ import annotations

import asyncio

from langgraph.types import Command

from app.config import Settings
from app.core.exceptions import (
    AppError,
    ClaimNotFoundError,
    ClaimValidationError,
    ConflictError,
    PolicyNotFoundError,
    RunLimitExceededError,
)
from app.core.logging import claim_id_var, get_logger
from app.core.tracing import observe
from app.db.repositories.claim_repository import ClaimRepository
from app.db.repositories.policy_repository import PolicyRepository
from app.db.repositories.usage_repository import UsageRepository
from app.domain.models import ClaimRecord, ClaimRequest, ClaimStatus, HistoryEntry, ReviewRequest

logger = get_logger(__name__)


class ClaimAssessmentService:
    def __init__(
        self,
        *,
        workflow,
        claim_repository: ClaimRepository,
        policy_repository: PolicyRepository,
        usage_repository: UsageRepository,
        settings: Settings,
    ) -> None:
        self._workflow = workflow
        self._claims = claim_repository
        self._policies = policy_repository
        self._usage = usage_repository
        self._settings = settings
        self._tasks: set[asyncio.Task] = set()  # strong refs, so running tasks aren't garbage-collected

    async def submit(self, claim: ClaimRequest) -> ClaimRecord:
        claim_id_var.set(claim.claim_id)
        policy = await self._policies.get_issued_policy(claim.policy_id)
        if policy is None:
            raise PolicyNotFoundError(f"No issued policy found for policy_id={claim.policy_id}")
        if claim.customer_id != policy.customer_id:
            raise ClaimValidationError(f"customer_id {claim.customer_id!r} does not own policy {claim.policy_id}")
        if await self._claims.get_claim(claim.claim_id) is not None:
            raise ConflictError(f"Claim {claim.claim_id} already exists.")
        if not await self._usage.try_consume_run(self._settings.max_runs_per_day):
            raise RunLimitExceededError("The demo's daily assessment limit has been reached. Try again tomorrow.")

        await self._claims.create_claim(claim)
        logger.info(
            "claim_submitted",
            extra={"ctx_claim_type": claim.claim_type, "ctx_description_chars": len(claim.description)},
        )
        self._start_assessment(claim)
        return await self._claims.get_record(claim.claim_id)

    async def resume_interrupted(self) -> int:
        """Re-runs claims a restart left `submitted` (their daily-cap run was
        already counted). Returns how many were resumed."""
        records = await self._claims.list_records(status=ClaimStatus.SUBMITTED, limit=1000)
        for record in records:
            claim = record.claim
            entry = HistoryEntry(actor="system", action="requeued", reason="Assessment was interrupted by a restart.")
            await self._claims.requeue(claim.claim_id, entry)
            self._start_assessment(claim)
        if records:
            logger.info("assessments_resumed", extra={"ctx_count": len(records)})
        return len(records)

    async def drain(self, timeout: float | None = None) -> None:
        """Waits for running assessments. Any still running after `timeout` are
        cancelled; they stay `submitted` and are resumed on the next startup."""
        if not self._tasks:
            return
        _, pending = await asyncio.wait(set(self._tasks), timeout=timeout)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    def _start_assessment(self, claim: ClaimRequest) -> None:
        task = asyncio.create_task(self._assess(claim), name=f"assess-{claim.claim_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _assess(self, claim: ClaimRequest) -> None:
        # The task runs in a copy of the submitting request's context, so its
        # logs keep that request_id.
        claim_id_var.set(claim.claim_id)
        try:
            with observe("claim_assessment", as_type="agent", input={"claim_id": claim.claim_id}):
                await self._workflow.ainvoke({"claim": claim.model_dump(mode="json")}, _thread(claim.claim_id))
        except Exception as exc:
            logger.exception("assessment_failed")
            reason = exc.message if isinstance(exc, AppError) else "Unexpected error during assessment."
            await self._claims.mark_failed(claim.claim_id, HistoryEntry(actor="system", action="failed", reason=reason))

    async def review(self, claim_id: str, review: ReviewRequest) -> ClaimRecord:
        claim_id_var.set(claim_id)
        record = await self._claims.get_record(claim_id)
        if record is None:
            raise ClaimNotFoundError(f"No claim found for claim_id={claim_id}")
        if record.status != ClaimStatus.UNDER_REVIEW:
            raise ConflictError(f"Claim {claim_id} is {record.status.value}, not under_review.")
        if review.action == "override" and review.outcome is None:
            raise ClaimValidationError("An override needs an outcome.")

        await self._workflow.ainvoke(Command(resume=review.model_dump(mode="json")), _thread(claim_id))
        return await self._claims.get_record(claim_id)

    async def get(self, claim_id: str) -> ClaimRecord:
        record = await self._claims.get_record(claim_id)
        if record is None:
            raise ClaimNotFoundError(f"No claim found for claim_id={claim_id}")
        return record

    async def list(self, status: ClaimStatus | None) -> list[ClaimRecord]:
        return await self._claims.list_records(status=status)


def _thread(claim_id: str) -> dict:
    return {"configurable": {"thread_id": claim_id}}
