"""Claims API: submit a claim for assessment, read claims."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.api.deps import get_claim_service
from app.api.security import verify_access_code
from app.domain.models import ClaimRecord, ClaimRequest, ClaimStatus
from app.services.claim_assessment_service import ClaimAssessmentService

router = APIRouter(prefix="/claims", tags=["claims"], dependencies=[Depends(verify_access_code)])


@router.post(
    "",
    response_model=ClaimRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a claim; the agent assesses it and it is auto-finalized or escalated to review.",
)
async def submit_claim(
    claim: ClaimRequest, service: ClaimAssessmentService = Depends(get_claim_service)
) -> ClaimRecord:
    return await service.submit(claim)


@router.get("", response_model=list[ClaimRecord], summary="List claims, newest first (e.g. ?status=under_review).")
async def list_claims(
    status: ClaimStatus | None = None, service: ClaimAssessmentService = Depends(get_claim_service)
) -> list[ClaimRecord]:
    return await service.list(status)


@router.get("/{claim_id}", response_model=ClaimRecord, summary="A claim with its decision, tool steps and audit trail.")
async def get_claim(claim_id: str, service: ClaimAssessmentService = Depends(get_claim_service)) -> ClaimRecord:
    return await service.get(claim_id)
