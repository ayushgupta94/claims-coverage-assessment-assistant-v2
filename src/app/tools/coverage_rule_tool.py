"""Tool: coverage_rule_check

A deterministic, auditable rule engine -- deliberately *not* an LLM call.
Coverage eligibility needs to be traceable to explicit business rules, not
model inference: an insurer needs to be able to point to *why* a claim was
rejected. The agent calls this tool and treats its output as ground truth.

Rule evaluation is split by where the data actually lives:
  - covered claim types (an allow-list of insured perils), exclusions and
    waiting_period_days come from PolicyProductVersion (shared, product-level rules)
  - sum_insured + inception_date come from IssuedPolicy
    (customer-specific values)
  - claim_type + amount + incident_date come from the Claim itself

Takes only `claim_id` -- the tool looks up the claim, its issued policy,
and that policy's product version itself, rather than requiring the LLM to
reproduce that data in its tool call arguments.
"""

from __future__ import annotations

from datetime import timedelta

from pydantic import BaseModel, Field

from app.core.exceptions import ClaimValidationError, PolicyNotFoundError
from app.db.repositories.claim_repository import ClaimRepository
from app.db.repositories.policy_repository import PolicyRepository
from app.domain.models import CoverageRuleResult


class CoverageRuleCheckInput(BaseModel):
    """Check a claim against its policy's insured perils, exclusions, waiting period and sum insured (deterministic)."""

    claim_id: str = Field(description="The claim_id of the claim under assessment")


async def check_coverage_rules(
    claim_repository: ClaimRepository,
    policy_repository: PolicyRepository,
    input: CoverageRuleCheckInput,
) -> CoverageRuleResult:
    claim = await claim_repository.get_claim(input.claim_id)
    if claim is None:
        raise ClaimValidationError(f"No claim found for claim_id={input.claim_id}")

    issued_policy = await policy_repository.get_issued_policy(claim.policy_id)
    if issued_policy is None:
        raise PolicyNotFoundError(f"No issued policy found for policy_id={claim.policy_id}")

    product_version = await policy_repository.get_product_version(issued_policy.product_version_id)
    if product_version is None:
        raise PolicyNotFoundError(f"No product version found for product_version_id={issued_policy.product_version_id}")

    violated_rules: list[str] = []

    # Claim types are customer-entered: compare them normalised, so "Racing " can't slip past an exclusion.
    claim_type = claim.claim_type.strip().lower()
    if claim_type in product_version.excluded_claim_types:
        violated_rules.append(f"claim_type '{claim_type}' is explicitly excluded by the policy")
    elif claim_type not in product_version.covered_claim_types:
        # Allow-list, failing closed: a peril the wording doesn't insure (e.g. a car
        # collision on a home policy) is not covered, and neither is anything on a
        # product whose insured perils were never configured.
        covers = ", ".join(product_version.covered_claim_types) or "none configured"
        violated_rules.append(
            f"claim_type '{claim_type}' is not an insured peril under {product_version.product_id} (covers: {covers})"
        )

    if product_version.waiting_period_days > 0:
        waiting_period_end = issued_policy.inception_date + timedelta(days=product_version.waiting_period_days)
        if claim.incident_date < waiting_period_end:
            violated_rules.append(
                f"incident occurred during the {product_version.waiting_period_days}-day waiting period"
            )

    if claim.amount > issued_policy.sum_insured:
        violated_rules.append(f"claim amount {claim.amount} exceeds policy sum insured {issued_policy.sum_insured}")

    is_covered = len(violated_rules) == 0
    notes = "All coverage rules satisfied." if is_covered else "; ".join(violated_rules)

    return CoverageRuleResult(is_covered=is_covered, violated_rules=violated_rules, notes=notes)
