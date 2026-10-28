"""Agent loop: tool execution, safety nets, guardrails and fallback paths."""

from app.agent.llm_client import LLMResponse
from app.agent.orchestrator import ClaimAssessmentOrchestrator
from app.core.exceptions import LLMProviderError
from app.domain.models import CoverageOutcome
from tests.fakes import CLEAN_TOOL_RESULTS, FakeLLM, FakeToolExecutor, final_answer, make_claim, tool_call


def make_orchestrator(llm, tools: FakeToolExecutor) -> ClaimAssessmentOrchestrator:
    return ClaimAssessmentOrchestrator(llm, lambda: tools)


async def test_runs_tool_calls_then_returns_structured_answer():
    claim = make_claim()
    llm = FakeLLM(
        [
            tool_call("retrieve_policy_clauses", {"query": "collision", "policy_id": "POL-1"}),
            tool_call("check_coverage_rules", {"claim_id": "CLAIM-1"}),
            tool_call("score_fraud_risk", {"claim_id": "CLAIM-1"}),
            final_answer(),
        ]
    )
    tools = FakeToolExecutor(CLEAN_TOOL_RESULTS)

    run = await make_orchestrator(llm, tools).run(claim)

    assert run.answer.coverage_outcome == CoverageOutcome.COVERED
    assert run.used_fallback is False
    assert [name for name, _ in tools.calls] == ["retrieve_policy_clauses", "check_coverage_rules", "score_fraud_risk"]
    assert list(run.retrieved) == ["AUTO-GOLD-V1-2.1"]
    assert all(step.ok for step in run.steps)
    assert run.usage.llm_calls == 5  # 4 agent calls + 1 injection classifier


async def test_deterministic_checks_are_fetched_even_if_the_model_skips_them():
    tools = FakeToolExecutor(CLEAN_TOOL_RESULTS)
    run = await make_orchestrator(FakeLLM([final_answer()]), tools).run(make_claim())

    assert {name for name, _ in tools.calls} == {"check_coverage_rules", "score_fraud_risk"}
    assert run.coverage.is_covered is True


async def test_tool_arguments_for_another_claim_are_blocked():
    llm = FakeLLM([tool_call("check_coverage_rules", {"claim_id": "SOMEONE-ELSE"}), final_answer()])
    tools = FakeToolExecutor(CLEAN_TOOL_RESULTS)

    run = await make_orchestrator(llm, tools).run(make_claim())

    assert ("check_coverage_rules", {"claim_id": "SOMEONE-ELSE"}) not in tools.calls
    assert "tool_argument_violation" in run.guardrail_flags
    assert run.steps[0].ok is False


async def test_injection_classifier_flag_is_recorded():
    run = await make_orchestrator(
        FakeLLM([final_answer()], injection_verdict=True), FakeToolExecutor(CLEAN_TOOL_RESULTS)
    ).run(make_claim())
    assert "injection_suspected" in run.guardrail_flags


async def test_refusal_or_missing_answer_uses_fallback():
    llm = FakeLLM([LLMResponse(content=None, refusal="I can't help with that")])
    run = await make_orchestrator(llm, FakeToolExecutor(CLEAN_TOOL_RESULTS)).run(make_claim())
    assert run.used_fallback is True
    assert run.answer is None


async def test_answer_for_a_different_claim_is_discarded():
    run = await make_orchestrator(FakeLLM([final_answer(claim_id="OTHER")]), FakeToolExecutor(CLEAN_TOOL_RESULTS)).run(
        make_claim()
    )
    assert run.used_fallback is True


async def test_llm_outage_mid_run_falls_back_with_deterministic_results():
    class FailingLLM(FakeLLM):
        async def generate(self, messages, *, tools=None, response_format=None, purpose="agent"):
            if purpose == "agent":
                raise LLMProviderError("timeout")
            return await super().generate(messages, purpose=purpose)

    run = await make_orchestrator(FailingLLM([]), FakeToolExecutor(CLEAN_TOOL_RESULTS)).run(make_claim())
    assert run.used_fallback is True
    assert run.fraud.risk_level == "low"


async def test_max_iterations_stops_the_loop():
    endless = [tool_call("lookup_claim_history", {"claim_id": "CLAIM-1"}) for _ in range(10)]
    llm = FakeLLM(endless)
    run = await make_orchestrator(llm, FakeToolExecutor(CLEAN_TOOL_RESULTS)).run(make_claim())
    assert llm.agent_calls == 6
    assert run.used_fallback is True
