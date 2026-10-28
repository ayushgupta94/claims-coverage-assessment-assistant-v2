"""Tool: retrieve_policy_clauses

Retrieves the policy clauses most relevant to a query, scoped to the claim's
own policy wording. Takes `policy_id` and resolves policy_id -> IssuedPolicy
-> product_version_id server-side, so the LLM never needs to know the
product/version structure ("resolve identifiers server-side", same pattern
as the other tools).

Returns matched clauses (citable by clause_id) plus each match's parent
section once, as wider context.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.exceptions import PolicyNotFoundError
from app.db.repositories.policy_repository import PolicyRepository
from app.domain.models import RetrievedClause, RetrievedSection
from app.retrieval.hybrid_retriever import HybridRetriever


class PolicyClauseRetrievalInput(BaseModel):
    """Retrieve the policy clauses most relevant to a query, scoped to the claim's own policy wording."""

    query: str = Field(description="What to look for, e.g. 'theft of vehicle, police report requirement'")
    policy_id: str = Field(description="The policy_id of the claim under assessment")


class PolicyClauseRetrievalOutput(BaseModel):
    clauses: list[RetrievedClause]
    sections: list[RetrievedSection]


async def retrieve_policy_clauses(
    policy_repository: PolicyRepository, retriever: HybridRetriever, input: PolicyClauseRetrievalInput
) -> PolicyClauseRetrievalOutput:
    issued_policy = await policy_repository.get_issued_policy(input.policy_id)
    if issued_policy is None:
        raise PolicyNotFoundError(f"No issued policy found for policy_id={input.policy_id}")

    result = await retriever.retrieve(input.query, product_version_id=issued_policy.product_version_id)
    return PolicyClauseRetrievalOutput(
        clauses=[
            RetrievedClause(
                clause_id=match.clause.clause_id,
                title=match.clause.title,
                page=match.clause.page,
                text=match.clause.text,
                section_id=match.clause.section_id,
                score=match.score,
                vector_score=match.vector_score,
            )
            for match in result.matches
        ],
        sections=[RetrievedSection(section_id=s.section_id, title=s.title, text=s.text) for s in result.sections],
    )
