"""Persistence for claims -- one collection, one entity across its lifecycle.

A claim document holds the submitted claim fields at the top level (so
history queries can filter on policy_id / filed_at), plus its lifecycle
state: status, the decision, the agent's tool steps, token usage, and an
append-only `decision_history` audit trail.

    submitted -> under_review -> approved / rejected   (escalated to a human)
    submitted -> approved / rejected                   (auto-finalized)

"Claim history" is a query over this same collection for other claims on
the same policy, excluding the claim being assessed (it is persisted before
the agent runs and would otherwise count in its own fraud-frequency signal).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

from app.core.exceptions import ConflictError
from app.domain.models import (
    ClaimRecord,
    ClaimRequest,
    ClaimStatus,
    CoverageDecision,
    HistoryEntry,
    TokenUsage,
    ToolStep,
)


class ClaimRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._claims = db["claims"]

    async def ensure_indexes(self) -> None:
        await self._claims.create_index("claim_id", unique=True)
        await self._claims.create_index("policy_id")
        await self._claims.create_index("status")

    async def create_claim(self, claim: ClaimRequest, status: ClaimStatus = ClaimStatus.SUBMITTED) -> None:
        document = {**claim.model_dump(), "status": status.value, "decision": None, "steps": [], "decision_history": []}
        try:
            await self._claims.insert_one(document)
        except DuplicateKeyError as exc:
            raise ConflictError(f"Claim {claim.claim_id} already exists.") from exc

    async def save_assessment(
        self,
        claim_id: str,
        *,
        decision: CoverageDecision,
        steps: list[ToolStep],
        usage: TokenUsage,
        status: ClaimStatus,
        history: list[HistoryEntry],
    ) -> None:
        await self._claims.update_one(
            {"claim_id": claim_id},
            {
                "$set": {
                    "decision": decision.model_dump(mode="json"),
                    "steps": [step.model_dump() for step in steps],
                    "usage": usage.model_dump(),
                    "status": status.value,
                },
                "$push": {"decision_history": {"$each": [entry.model_dump(mode="json") for entry in history]}},
            },
        )

    async def record_review(
        self, claim_id: str, *, decision: CoverageDecision, status: ClaimStatus, entry: HistoryEntry
    ) -> None:
        await self._claims.update_one(
            {"claim_id": claim_id},
            {
                "$set": {"decision": decision.model_dump(mode="json"), "status": status.value},
                "$push": {"decision_history": entry.model_dump(mode="json")},
            },
        )

    async def get_claim(self, claim_id: str) -> ClaimRequest | None:
        doc = await self._claims.find_one({"claim_id": claim_id})
        return ClaimRequest.model_validate(doc) if doc else None

    async def get_record(self, claim_id: str) -> ClaimRecord | None:
        doc = await self._claims.find_one({"claim_id": claim_id})
        return _to_record(doc) if doc else None

    async def list_records(self, *, status: ClaimStatus | None = None, limit: int = 50) -> list[ClaimRecord]:
        query = {"status": status.value} if status else {}
        cursor = self._claims.find(query).sort("filed_at", -1).limit(limit)
        return [_to_record(doc) async for doc in cursor]

    async def get_history_for_policy(
        self,
        policy_id: str,
        *,
        exclude_claim_id: str | None = None,
        lookback_days: int | None = None,
    ) -> list[ClaimRequest]:
        query: dict = {"policy_id": policy_id}
        if exclude_claim_id is not None:
            query["claim_id"] = {"$ne": exclude_claim_id}
        if lookback_days is not None:
            query["filed_at"] = {"$gte": datetime.now(UTC) - timedelta(days=lookback_days)}

        cursor = self._claims.find(query).sort("filed_at", -1)
        return [ClaimRequest.model_validate(doc) async for doc in cursor]


def _to_record(doc: dict) -> ClaimRecord:
    return ClaimRecord(
        claim=ClaimRequest.model_validate(doc),
        status=ClaimStatus(doc["status"]),
        decision=doc.get("decision"),
        steps=doc.get("steps", []),
        usage=doc.get("usage") or TokenUsage(),
        decision_history=doc.get("decision_history", []),
    )
