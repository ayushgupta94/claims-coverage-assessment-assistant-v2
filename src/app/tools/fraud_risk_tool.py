"""Tool: fraud_risk_scoring

A transparent, weighted heuristic scorer -- explicitly a POC-level
implementation, not a trained ML fraud model. Each signal contributes a
documented weight to a 0-1 risk score, and every triggered signal is named
in the output. Swapping in a trained classifier later wouldn't require
changing this tool's interface (claim_id in, FraudRiskResult out).

Takes only `claim_id`, mirroring coverage_rule_tool: the tool looks up the
claim, its issued policy (for inception_date), and its claim history
itself rather than trusting the LLM to reproduce them. Product-version
data isn't needed here -- fraud signals are about timing/amount/frequency,
not policy wording or exclusions.
"""

from __future__ import annotations

from datetime import timedelta

from pydantic import BaseModel, Field

from app.config import Settings
from app.core.exceptions import ClaimValidationError, PolicyNotFoundError
from app.db.repositories.claim_repository import ClaimRepository
from app.db.repositories.policy_repository import PolicyRepository
from app.domain.models import FraudRiskLevel, FraudRiskResult

# Signal weights sum to 1.0 -- documented here rather than buried as magic
# numbers so the scoring logic is auditable at a glance.
_WEIGHT_HIGH_AMOUNT = 0.35
_WEIGHT_EARLY_CLAIM = 0.30
_WEIGHT_FREQUENCY = 0.35


class FraudRiskScoringInput(BaseModel):
    """Score fraud risk for a submitted claim from amount, timing and claim-frequency signals (deterministic)."""

    claim_id: str = Field(description="The claim_id of the claim under assessment")


async def score_fraud_risk(
    settings: Settings,
    claim_repository: ClaimRepository,
    policy_repository: PolicyRepository,
    input: FraudRiskScoringInput,
) -> FraudRiskResult:
    claim = await claim_repository.get_claim(input.claim_id)
    if claim is None:
        raise ClaimValidationError(f"No claim found for claim_id={input.claim_id}")

    issued_policy = await policy_repository.get_issued_policy(claim.policy_id)
    if issued_policy is None:
        raise PolicyNotFoundError(f"No issued policy found for policy_id={claim.policy_id}")

    # Lookback is computed relative to the claim's own filed_at, not
    # wall-clock "now" -- a fixed real-world date makes wall-clock lookback
    # silently exclude older seed/test data (a real bug caught by tests).
    # exclude_claim_id is required here: this claim was already persisted
    # (status=submitted) before this tool runs, so without excluding it,
    # a claim would count itself in its own frequency signal.
    history = await claim_repository.get_history_for_policy(claim.policy_id, exclude_claim_id=claim.claim_id)
    cutoff = claim.filed_at - timedelta(days=settings.fraud_frequency_lookback_days)
    history = [h for h in history if h.filed_at >= cutoff]

    score = 0.0
    signals: list[str] = []

    if claim.amount >= settings.fraud_high_amount_threshold:
        score += _WEIGHT_HIGH_AMOUNT
        signals.append(
            f"claim amount {claim.amount} exceeds high-value threshold {settings.fraud_high_amount_threshold}"
        )

    days_since_inception = (claim.incident_date - issued_policy.inception_date).days
    if 0 <= days_since_inception <= settings.fraud_early_claim_days_threshold:
        score += _WEIGHT_EARLY_CLAIM
        signals.append(
            f"incident occurred {days_since_inception} days after policy inception "
            f"(threshold: {settings.fraud_early_claim_days_threshold})"
        )

    if len(history) >= settings.fraud_frequency_claim_count_threshold:
        score += _WEIGHT_FREQUENCY
        signals.append(
            f"{len(history)} claims filed on this policy in the last "
            f"{settings.fraud_frequency_lookback_days} days "
            f"(threshold: {settings.fraud_frequency_claim_count_threshold})"
        )

    score = round(min(score, 1.0), 4)

    if score >= 0.6:
        level = FraudRiskLevel.HIGH
    elif score >= 0.3:
        level = FraudRiskLevel.MEDIUM
    else:
        level = FraudRiskLevel.LOW

    return FraudRiskResult(risk_level=level, risk_score=score, signals=signals)
