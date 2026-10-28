"""Decision policy: turns an AgentRun into the final CoverageDecision.

The LLM proposes; this code decides. Pure functions, no I/O, fully unit-tested.

1. Outcome. The deterministic coverage rules win: if the LLM's outcome
   contradicts check_coverage_rules, the rules' outcome is used and the
   claim is escalated. "partially_covered" counts as agreeing with "covered".
2. Citations. Only clauses actually retrieved in this run are kept.
3. Confidence. Computed from evidence, not taken from the model:
       0.40 * LLM agrees with the rules
     + 0.25 * top retrieval similarity (how well the policy wording matched)
     + 0.20 * share of the LLM's citations that are valid
     + 0.15 * (1 - fraud risk score)
   The model's self-reported confidence is only logged (it is not calibrated).
4. Human review. Required when any reason applies; the reasons are returned
   so a reviewer sees *why* a claim was escalated.
"""

from __future__ import annotations

from app.agent.guardrails import INJECTION_SUSPECTED, TOOL_ARGUMENT_VIOLATION, validate_citations
from app.domain.models import (
    AgentRun,
    Citation,
    CoverageDecision,
    CoverageOutcome,
    FraudRiskLevel,
    ReviewReason,
)

WEIGHT_AGREEMENT = 0.40
WEIGHT_RETRIEVAL = 0.25
WEIGHT_CITATIONS = 0.20
WEIGHT_FRAUD = 0.15


def confidence_components(
    *, agrees_with_rules: bool, top_retrieval_score: float, citation_validity: float, fraud_score: float
) -> dict[str, float]:
    """Each evidence term's weighted contribution; together they make the confidence."""
    return {
        "rules_agreement": WEIGHT_AGREEMENT * (1.0 if agrees_with_rules else 0.0),
        "retrieval_match": WEIGHT_RETRIEVAL * top_retrieval_score,
        "citation_validity": WEIGHT_CITATIONS * citation_validity,
        "low_fraud_risk": WEIGHT_FRAUD * (1.0 - fraud_score),
    }


def compute_confidence(
    *, agrees_with_rules: bool, top_retrieval_score: float, citation_validity: float, fraud_score: float
) -> float:
    score = sum(
        confidence_components(
            agrees_with_rules=agrees_with_rules,
            top_retrieval_score=top_retrieval_score,
            citation_validity=citation_validity,
            fraud_score=fraud_score,
        ).values()
    )
    return round(min(max(score, 0.0), 1.0), 3)


def review_reasons(
    *,
    outcome: CoverageOutcome,
    fraud_level: FraudRiskLevel,
    confidence: float,
    confidence_threshold: float,
    llm_contradicts_rules: bool,
    guardrail_flags: list[str],
    used_fallback: bool,
) -> list[ReviewReason]:
    checks = [
        (outcome == CoverageOutcome.NOT_COVERED, ReviewReason.NOT_COVERED),
        (fraud_level != FraudRiskLevel.LOW, ReviewReason.FRAUD_RISK),
        (confidence < confidence_threshold, ReviewReason.LOW_CONFIDENCE),
        (llm_contradicts_rules, ReviewReason.LLM_CONTRADICTS_RULES),
        (INJECTION_SUSPECTED in guardrail_flags, ReviewReason.INJECTION_SUSPECTED),
        (TOOL_ARGUMENT_VIOLATION in guardrail_flags, ReviewReason.TOOL_ARGUMENT_VIOLATION),
        (used_fallback, ReviewReason.FALLBACK_USED),
    ]
    return [reason for applies, reason in checks if applies]


def decide(
    run: AgentRun, *, claim_id: str, confidence_threshold: float, model: str, prompt_version: str
) -> CoverageDecision:
    rules_outcome = CoverageOutcome.COVERED if run.coverage.is_covered else CoverageOutcome.NOT_COVERED

    if run.answer is not None:
        llm_says_not_covered = run.answer.coverage_outcome == CoverageOutcome.NOT_COVERED
        agrees = llm_says_not_covered == (not run.coverage.is_covered)
        outcome = run.answer.coverage_outcome if agrees else rules_outcome
        cited = run.answer.supporting_clause_ids
        reasoning = run.answer.reasoning
    else:
        agrees = False  # no model opinion to corroborate the rules
        outcome = rules_outcome
        cited = []
        reasoning = (
            "Assembled from the deterministic checks because no valid agent answer was available. "
            f"Coverage rules: {run.coverage.notes}"
        )

    valid_ids, dropped_ids = validate_citations(cited, set(run.retrieved))
    citation_validity = len(valid_ids) / len(cited) if cited else 0.0
    vector_scores = [clause.vector_score for clause in run.retrieved.values() if clause.vector_score is not None]

    evidence = dict(
        agrees_with_rules=agrees,
        top_retrieval_score=max(vector_scores, default=0.0),
        citation_validity=citation_validity,
        fraud_score=run.fraud.risk_score,
    )
    confidence = compute_confidence(**evidence)
    reasons = review_reasons(
        outcome=outcome,
        fraud_level=run.fraud.risk_level,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
        llm_contradicts_rules=run.answer is not None and not agrees,
        guardrail_flags=run.guardrail_flags,
        used_fallback=run.used_fallback,
    )

    return CoverageDecision(
        claim_id=claim_id,
        coverage_outcome=outcome,
        citations=[
            Citation(
                clause_id=clause_id,
                title=run.retrieved[clause_id].title,
                page=run.retrieved[clause_id].page,
                text=run.retrieved[clause_id].text,
            )
            for clause_id in valid_ids
        ],
        dropped_citations=dropped_ids,
        confidence_score=confidence,
        confidence_breakdown={name: round(value, 3) for name, value in confidence_components(**evidence).items()},
        llm_outcome=run.answer.coverage_outcome if run.answer is not None else None,
        fraud_risk=run.fraud,
        coverage_rules=run.coverage,
        requires_human_review=bool(reasons),
        review_reasons=reasons,
        reasoning=reasoning,
        model=model,
        prompt_version=prompt_version,
    )
