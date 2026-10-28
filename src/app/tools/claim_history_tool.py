"""Tool: lookup_claim_history

Returns prior claims filed against the same policy as the given claim --
"history" is a query over the same `claims` collection, not a separate
stored entity (see db/repositories/claim_repository.py's docstring).

Takes only `claim_id`, mirroring the other tools: the tool resolves the
claim's own policy_id itself and excludes the claim's own claim_id from
the results -- without that exclusion, a claim would appear in its own
history, since it's already persisted (status=submitted) before this tool
can be called.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.exceptions import ClaimValidationError
from app.db.repositories.claim_repository import ClaimRepository
from app.domain.models import ClaimRequest


class ClaimHistoryLookupInput(BaseModel):
    """Look up prior claims filed on the same policy as a submitted claim (the claim itself is excluded)."""

    claim_id: str = Field(description="The claim_id of the claim under assessment")
    lookback_days: int | None = Field(default=None, description="Only claims filed in the last N days")


async def lookup_claim_history(claim_repository: ClaimRepository, input: ClaimHistoryLookupInput) -> list[ClaimRequest]:
    claim = await claim_repository.get_claim(input.claim_id)
    if claim is None:
        raise ClaimValidationError(f"No claim found for claim_id={input.claim_id}")

    return await claim_repository.get_history_for_policy(
        claim.policy_id, exclude_claim_id=claim.claim_id, lookback_days=input.lookback_days
    )
