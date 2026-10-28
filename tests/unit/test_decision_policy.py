"""Decision policy: outcome, citations, confidence and review routing are code."""

from app.decision.policy import compute_confidence, decide
from app.domain.models import (
    AgentAnswer,
    AgentRun,
    CoverageOutcome,
    CoverageRuleResult,
    FraudRiskResult,
    RetrievedClause,
    ReviewReason,
)

CLAUSE = RetrievedClause(
    clause_id="AUTO-GOLD-V1-2.1",
    title="2.1 Collision Damage",
    page=2,
    text="Collision damage is covered.",
    section_id="AUTO-GOLD-V1-S2",
    score=0.03,
    vector_score=0.9,
)


def make_run(
    *,
    outcome="covered",
    is_covered=True,
    fraud=("low", 0.0),
    cited=("AUTO-GOLD-V1-2.1",),
    answer=True,
    flags=(),
    vector_score=0.9,
) -> AgentRun:
    return AgentRun(
        answer=AgentAnswer(
            claim_id="CLAIM-1",
            coverage_outcome=CoverageOutcome(outcome),
            supporting_clause_ids=list(cited),
            reasoning="Because of clause 2.1.",
            self_reported_confidence=0.99,
        )
        if answer
        else None,
        coverage=CoverageRuleResult(is_covered=is_covered, violated_rules=[] if is_covered else ["excluded"]),
        fraud=FraudRiskResult(risk_level=fraud[0], risk_score=fraud[1]),
        retrieved={CLAUSE.clause_id: CLAUSE.model_copy(update={"vector_score": vector_score})},
        guardrail_flags=list(flags),
        used_fallback=not answer,
    )


def decide_for(run: AgentRun):
    return decide(run, claim_id="CLAIM-1", confidence_threshold=0.7, model="m", prompt_version="v")


def test_clean_covered_claim_is_auto_approved_with_high_confidence():
    decision = decide_for(make_run())
    assert decision.coverage_outcome == CoverageOutcome.COVERED
    assert decision.requires_human_review is False
    assert decision.confidence_score == compute_confidence(
        agrees_with_rules=True, top_retrieval_score=0.9, citation_validity=1.0, fraud_score=0.0
    )
    assert [c.clause_id for c in decision.citations] == ["AUTO-GOLD-V1-2.1"]


def test_rules_win_when_llm_contradicts_them():
    decision = decide_for(make_run(outcome="covered", is_covered=False))
    assert decision.coverage_outcome == CoverageOutcome.NOT_COVERED
    assert ReviewReason.LLM_CONTRADICTS_RULES in decision.review_reasons
    assert ReviewReason.NOT_COVERED in decision.review_reasons


def test_partially_covered_counts_as_agreeing_with_covered_rules():
    decision = decide_for(make_run(outcome="partially_covered", is_covered=True))
    assert decision.coverage_outcome == CoverageOutcome.PARTIALLY_COVERED
    assert ReviewReason.LLM_CONTRADICTS_RULES not in decision.review_reasons


def test_invented_citations_are_dropped_and_lower_confidence():
    honest = decide_for(make_run())
    invented = decide_for(make_run(cited=("AUTO-GOLD-V1-2.1", "MADE-UP-9.9")))
    assert [c.clause_id for c in invented.citations] == ["AUTO-GOLD-V1-2.1"]
    assert invented.dropped_citations == ["MADE-UP-9.9"]
    assert invented.confidence_score < honest.confidence_score


def test_fraud_risk_escalates_even_when_covered():
    decision = decide_for(make_run(fraud=("medium", 0.35)))
    assert decision.coverage_outcome == CoverageOutcome.COVERED
    assert decision.review_reasons == [ReviewReason.FRAUD_RISK]


def test_guardrail_flags_escalate():
    decision = decide_for(make_run(flags=("injection_suspected", "tool_argument_violation")))
    assert ReviewReason.INJECTION_SUSPECTED in decision.review_reasons
    assert ReviewReason.TOOL_ARGUMENT_VIOLATION in decision.review_reasons


def test_fallback_uses_rules_outcome_and_always_escalates():
    decision = decide_for(make_run(answer=False))
    assert decision.coverage_outcome == CoverageOutcome.COVERED
    assert ReviewReason.FALLBACK_USED in decision.review_reasons
    assert ReviewReason.LOW_CONFIDENCE in decision.review_reasons
    assert decision.citations == []


def test_weak_evidence_alone_escalates_as_low_confidence():
    # No citations and a weak retrieval match: 0.40 + 0.25*0.5 + 0 + 0.15 = 0.675 < 0.7
    decision = decide_for(make_run(cited=(), vector_score=0.5))
    assert decision.confidence_score == 0.675
    assert decision.review_reasons == [ReviewReason.LOW_CONFIDENCE]


def test_confidence_is_bounded():
    assert (
        compute_confidence(agrees_with_rules=True, top_retrieval_score=1.0, citation_validity=1.0, fraud_score=0.0)
        == 1.0
    )
    assert (
        compute_confidence(agrees_with_rules=False, top_retrieval_score=0.0, citation_validity=0.0, fraud_score=1.0)
        == 0.0
    )
