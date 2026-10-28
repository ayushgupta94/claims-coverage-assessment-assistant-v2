r"""The claim workflow as a LangGraph state graph.

    START -> assess -> apply_policy -> route --(auto)--> END
                                           \--(needs review)--> human_review -> END
                                                                 (pauses here)

Why LangGraph: human review can take hours, so the workflow must pause,
persist its state and resume later -- possibly in another process after the
server slept or redeployed. `interrupt()` pauses the graph; the checkpointer
(MongoDB in production) stores the state under thread_id = claim_id; a
`Command(resume=...)` with the reviewer's decision continues it.

Nodes only pass plain dicts through the state so checkpoints serialise cleanly.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.agent.orchestrator import ClaimAssessmentOrchestrator
from app.agent.prompts import PROMPT_VERSION
from app.config import Settings
from app.core.logging import get_logger
from app.db.repositories.claim_repository import ClaimRepository
from app.decision.policy import decide
from app.domain.models import (
    AgentRun,
    ClaimRequest,
    ClaimStatus,
    CoverageDecision,
    CoverageOutcome,
    HistoryEntry,
    ReviewRequest,
)

logger = get_logger(__name__)


class ClaimState(TypedDict, total=False):
    claim: dict
    run: dict
    decision: dict
    status: str


def final_status(outcome: CoverageOutcome) -> ClaimStatus:
    return ClaimStatus.REJECTED if outcome == CoverageOutcome.NOT_COVERED else ClaimStatus.APPROVED


def build_claim_workflow(
    *,
    orchestrator: ClaimAssessmentOrchestrator,
    claim_repository: ClaimRepository,
    settings: Settings,
    checkpointer,
):
    async def assess(state: ClaimState) -> ClaimState:
        run = await orchestrator.run(ClaimRequest.model_validate(state["claim"]))
        return {"run": run.model_dump(mode="json")}

    async def apply_policy(state: ClaimState) -> ClaimState:
        run = AgentRun.model_validate(state["run"])
        decision = decide(
            run,
            claim_id=state["claim"]["claim_id"],
            confidence_threshold=settings.review_confidence_threshold,
            model=settings.openai_model,
            prompt_version=PROMPT_VERSION,
        )
        logger.info(
            "decision",
            extra={
                "ctx_outcome": decision.coverage_outcome.value,
                "ctx_confidence": decision.confidence_score,
                "ctx_fraud_level": decision.fraud_risk.risk_level.value,
                "ctx_review_reasons": [reason.value for reason in decision.review_reasons],
                "ctx_citations": [citation.clause_id for citation in decision.citations],
                "ctx_cost_usd": run.usage.cost_usd,
            },
        )
        return {"decision": decision.model_dump(mode="json")}

    async def route(state: ClaimState) -> ClaimState:
        decision = CoverageDecision.model_validate(state["decision"])
        run = AgentRun.model_validate(state["run"])
        history = [
            HistoryEntry(
                actor="agent",
                action="assessed",
                outcome=decision.coverage_outcome,
                reason=decision.reasoning,
                model=decision.model,
                prompt_version=decision.prompt_version,
            )
        ]
        if decision.requires_human_review:
            status = ClaimStatus.UNDER_REVIEW
            reasons = ", ".join(reason.value for reason in decision.review_reasons)
            history.append(HistoryEntry(actor="system", action="escalated", reason=reasons))
        else:
            status = final_status(decision.coverage_outcome)
            action = "auto_approved" if status == ClaimStatus.APPROVED else "auto_rejected"
            history.append(HistoryEntry(actor="system", action=action, outcome=decision.coverage_outcome))

        await claim_repository.save_assessment(
            decision.claim_id, decision=decision, steps=run.steps, usage=run.usage, status=status, history=history
        )
        return {"status": status.value}

    async def human_review(state: ClaimState) -> ClaimState:
        # Pauses the graph until the API resumes it with the reviewer's input.
        review = ReviewRequest.model_validate(
            interrupt({"claim_id": state["claim"]["claim_id"], "decision": state["decision"]})
        )
        decision = CoverageDecision.model_validate(state["decision"])
        if review.action == "override" and review.outcome is not None:
            decision.coverage_outcome = review.outcome
        status = final_status(decision.coverage_outcome)

        entry = HistoryEntry(
            actor=review.reviewer,
            action="overridden" if review.action == "override" else "approved",
            outcome=decision.coverage_outcome,
            reason=review.reason,
        )
        await claim_repository.record_review(decision.claim_id, decision=decision, status=status, entry=entry)
        logger.info("human_review", extra={"ctx_action": review.action, "ctx_outcome": decision.coverage_outcome.value})
        return {"decision": decision.model_dump(mode="json"), "status": status.value}

    def after_route(state: ClaimState) -> str:
        return "human_review" if state["status"] == ClaimStatus.UNDER_REVIEW.value else END

    graph = StateGraph(ClaimState)
    graph.add_node("assess", assess)
    graph.add_node("apply_policy", apply_policy)
    graph.add_node("route", route)
    graph.add_node("human_review", human_review)
    graph.add_edge(START, "assess")
    graph.add_edge("assess", "apply_policy")
    graph.add_edge("apply_policy", "route")
    graph.add_conditional_edges("route", after_route, ["human_review", END])
    graph.add_edge("human_review", END)
    return graph.compile(checkpointer=checkpointer)
