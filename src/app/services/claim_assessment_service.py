"""Claim use cases: submit a claim for assessment, and review an escalated one.

Kept free of FastAPI so the same use cases can be driven from HTTP, a CLI or
the eval runner. Each claim is one workflow thread (thread_id = claim_id).
"""

from __future__ import annotations

from langgraph.types import Command

from app.config import Settings
from app.core.exceptions import (
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
from app.domain.models import ClaimRecord, ClaimRequest, ClaimStatus, ReviewRequest

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
        with observe("claim_assessment", as_type="agent", input={"claim_id": claim.claim_id}):
            await self._workflow.ainvoke({"claim": claim.model_dump(mode="json")}, _thread(claim.claim_id))
        return await self._claims.get_record(claim.claim_id)

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
