"""Agent loop: tool execution, concurrency, safety nets, guardrails and fallback paths."""

import asyncio
import json

import pytest

from app.agent.llm_client import LLMResponse
from app.agent.orchestrator import ClaimAssessmentOrchestrator
from app.core.exceptions import LLMProviderError, ToolExecutionError
from app.domain.models import CoverageOutcome
from tests.fakes import (
    CLEAN_TOOL_RESULTS,
    FakeLLM,
    FakeToolExecutor,
    final_answer,
    make_claim,
    tool_call,
    tool_calls,
)


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


# --- concurrency -----------------------------------------------------------------


class BarrierToolExecutor(FakeToolExecutor):
    """Each call waits until `parties` calls have started, so a turn only
    completes if its calls run concurrently (run serially, it times out).
    Earlier calls then finish later, to check results still keep call order."""

    def __init__(self, results_by_tool: dict[str, dict], parties: int) -> None:
        super().__init__(results_by_tool)
        self._parties = parties
        self._all_started = asyncio.Event()

    async def execute(self, name: str, arguments: dict) -> dict:
        position = len(self.calls)
        self.calls.append((name, arguments))
        if len(self.calls) >= self._parties:
            self._all_started.set()
        await asyncio.wait_for(self._all_started.wait(), timeout=1)
        await asyncio.sleep(0.01 * (self._parties - position))
        return self._results[name]


async def test_tool_calls_of_one_turn_run_concurrently_and_keep_call_order():
    names = ["check_coverage_rules", "score_fraud_risk", "retrieve_policy_clauses"]
    arguments = {"claim_id": "CLAIM-1"}
    llm = FakeLLM([tool_calls(*[(name, arguments) for name in names]), final_answer()])
    tools = BarrierToolExecutor(CLEAN_TOOL_RESULTS, parties=len(names))

    run = await make_orchestrator(llm, tools).run(make_claim())

    tool_messages = [m for m in llm.last_messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_messages] == [f"call-{name}" for name in names]
    assert [json.loads(m.content) for m in tool_messages] == [CLEAN_TOOL_RESULTS[name] for name in names]
    assert all(step.ok for step in run.steps)
    # Every call started before the first one finished.
    first_finish = min(step.started_ms + step.latency_ms for step in run.steps)
    assert all(step.started_ms <= first_finish for step in run.steps)
    assert run.used_fallback is False


class ClassifierWaitsForAgentLLM(FakeLLM):
    """The classifier only answers once the agent loop has made its first LLM
    call, so the run only completes if the two run concurrently. With
    unblock_classifier=False it hangs until cancelled (or a 1s safety timeout)."""

    def __init__(self, script, *, unblock_classifier: bool = True, **kwargs) -> None:
        super().__init__(script, **kwargs)
        self._unblock_classifier = unblock_classifier
        self.agent_started = asyncio.Event()
        self.classifier_cancelled = False

    async def generate(self, messages, *, tools=None, response_format=None, purpose="agent"):
        if purpose == "injection_classifier":
            try:
                await asyncio.wait_for(self.agent_started.wait(), timeout=1)
            except asyncio.CancelledError:
                self.classifier_cancelled = True
                raise
        elif self._unblock_classifier:
            self.agent_started.set()
        return await super().generate(messages, tools=tools, response_format=response_format, purpose=purpose)


async def test_injection_classifier_runs_alongside_the_agent_loop():
    llm = ClassifierWaitsForAgentLLM([final_answer()], injection_verdict=True)
    run = await make_orchestrator(llm, FakeToolExecutor(CLEAN_TOOL_RESULTS)).run(make_claim())

    assert run.guardrail_flags[0] == "injection_suspected"
    assert run.usage.llm_calls == 2  # agent + classifier


async def test_classifier_is_cancelled_when_the_agent_loop_fails():
    class BrokenTools(FakeToolExecutor):
        async def execute(self, name: str, arguments: dict) -> dict:
            await asyncio.sleep(0.01)  # like real I/O: lets the classifier start first
            raise ToolExecutionError(name, "down")

    llm = ClassifierWaitsForAgentLLM([final_answer()], unblock_classifier=False)
    with pytest.raises(ToolExecutionError):
        await make_orchestrator(llm, BrokenTools({})).run(make_claim())
    await asyncio.sleep(0)  # let the cancellation reach the task
    assert llm.classifier_cancelled is True
