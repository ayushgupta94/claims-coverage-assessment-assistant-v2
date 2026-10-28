"""Claims API: submit a claim for assessment (202 + poll), read claims."""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Depends, Response, status

from app.api.deps import get_claim_service
from app.api.security import verify_access_code
from app.domain.models import ClaimRecord, ClaimRequest, ClaimStatus
from app.services.claim_assessment_service import ClaimAssessmentService

router = APIRouter(prefix="/claims", tags=["claims"], dependencies=[Depends(verify_access_code)])


@router.post(
    "",
    response_model=ClaimRecord,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a claim. It is assessed in the background: poll the Location URL until status leaves 'submitted'.",
)
async def submit_claim(
    claim: ClaimRequest, response: Response, service: ClaimAssessmentService = Depends(get_claim_service)
) -> ClaimRecord:
    record = await service.submit(claim)
    response.headers["Location"] = f"/claims/{quote(claim.claim_id, safe='')}"
    return record


@router.get("", response_model=list[ClaimRecord], summary="List claims, newest first (e.g. ?status=under_review).")
async def list_claims(
    status: ClaimStatus | None = None, service: ClaimAssessmentService = Depends(get_claim_service)
) -> list[ClaimRecord]:
    return await service.list(status)


@router.get("/{claim_id}", response_model=ClaimRecord, summary="A claim with its decision, tool steps and audit trail.")
async def get_claim(claim_id: str, service: ClaimAssessmentService = Depends(get_claim_service)) -> ClaimRecord:
    return await service.get(claim_id)
