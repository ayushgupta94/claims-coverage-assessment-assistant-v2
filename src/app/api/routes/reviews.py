"""Human review: the queue of escalated claims, and the reviewer's decision."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_claim_service
from app.api.security import verify_access_code
from app.domain.models import ClaimRecord, ClaimStatus, ReviewRequest
from app.services.claim_assessment_service import ClaimAssessmentService

router = APIRouter(tags=["reviews"], dependencies=[Depends(verify_access_code)])


@router.get("/reviews", response_model=list[ClaimRecord], summary="Claims waiting for human review.")
async def review_queue(service: ClaimAssessmentService = Depends(get_claim_service)) -> list[ClaimRecord]:
    return await service.list(ClaimStatus.UNDER_REVIEW)


@router.post(
    "/claims/{claim_id}/review",
    response_model=ClaimRecord,
    summary="Approve the agent's outcome or override it; resumes the paused workflow.",
)
async def review_claim(
    claim_id: str, review: ReviewRequest, service: ClaimAssessmentService = Depends(get_claim_service)
) -> ClaimRecord:
    return await service.review(claim_id, review)
