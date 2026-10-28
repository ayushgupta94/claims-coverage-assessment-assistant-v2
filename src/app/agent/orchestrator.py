"""The agent: one LLM in a tool-calling (ReAct) loop over MCP tools.

    [ LLM decides -> call tool(s) concurrently -> observe results ]* -> structured answer
    (the injection classifier screens the input at the same time)

- The LLM chooses which tools to call and when to stop (max _MAX_ITERATIONS).
  The tool calls of one turn are independent, so they run concurrently.
- Tools are called through a real MCP client (ToolExecutor) against this
  app's own /mcp endpoint -- the same protocol an external client uses.
- The coverage-rules and fraud results are always fetched (safety net), even
  if the model skipped them, because the decision policy needs them.
- The run never fails just because the model misbehaved: a refusal, a bad
  answer, an LLM outage or hitting the iteration limit sets `used_fallback`,
  and the decision policy then routes the claim to a human.

The orchestrator returns an AgentRun record; it does not make the final
decision. That is code: see decision/policy.py.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

import httpx

from app.agent.guardrails import INJECTION_SUSPECTED, TOOL_ARGUMENT_VIOLATION, check_tool_arguments, screen_claim_text
from app.agent.llm_client import ChatMessage, OpenAIChatClient, ToolCall
from app.agent.prompts import SYSTEM_PROMPT, TOOL_SCHEMAS, build_user_message
from app.core.exceptions import LLMProviderError, ToolExecutionError
from app.core.logging import get_logger
from app.core.tracing import observe, trace_carrier
from app.domain.models import (
    AgentAnswer,
    AgentRun,
    ClaimRequest,
    CoverageRuleResult,
    FraudRiskResult,
    RetrievedClause,
    TokenUsage,
    ToolStep,
)

if TYPE_CHECKING:
    from mcp.client.session import ClientSession

logger = get_logger(__name__)

_MAX_ITERATIONS = 6

StepCallback = Callable[[ToolStep], Awaitable[None]]


class ToolExecutor:
    """Calls tools through a real MCP client session. One session (one HTTP
    connection + initialize handshake) per agent run, reused for every tool
    call in that run: use as `async with executor:`. Concurrent calls share
    the session; MCP matches responses to requests by id."""

    def __init__(self, mcp_server_url: str, headers: dict[str, str]) -> None:
        self._mcp_server_url = mcp_server_url
        self._headers = headers
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> ToolExecutor:
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        exit_stack = AsyncExitStack()
        try:
            http_client = await exit_stack.enter_async_context(
                httpx.AsyncClient(headers=self._headers, timeout=httpx.Timeout(30, read=300), follow_redirects=True)
            )
            read, write, _ = await exit_stack.enter_async_context(
                streamable_http_client(self._mcp_server_url, http_client=http_client)
            )
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except Exception as exc:
            await exit_stack.aclose()
            raise ToolExecutionError("mcp", f"Could not open MCP session: {exc}") from exc
        self._exit_stack, self._session = exit_stack, session
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._exit_stack = self._session = None

    async def execute(self, name: str, arguments: dict) -> dict:
        if self._session is None:
            raise RuntimeError("ToolExecutor.execute() called outside 'async with'")
        try:
            # Carry the trace context so server-side spans (e.g. retrieval)
            # nest under this tool call rather than starting their own trace.
            result = await self._session.call_tool(name, arguments, meta=trace_carrier() or None)
        except Exception as exc:
            raise ToolExecutionError(name, f"MCP call failed: {exc}") from exc
        if result.isError or not result.content:
            raise ToolExecutionError(name, result.content[0].text if result.content else "empty MCP result")
        try:
            return json.loads(result.content[0].text)
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(name, f"non-JSON MCP result: {exc}") from exc


class ClaimAssessmentOrchestrator:
    def __init__(self, llm: OpenAIChatClient, tool_executor_factory: Callable[[], ToolExecutor]) -> None:
        self._llm = llm
        self._new_executor = tool_executor_factory

    async def run(self, claim: ClaimRequest, *, on_step: StepCallback | None = None) -> AgentRun:
        """`on_step` is awaited with each tool step as it finishes (progress reporting)."""
        run = _RunState(claim, on_step)
        # The injection classifier only adds a flag (it never stops the run),
        # so it runs alongside the agent loop instead of in front of it.
        screening = asyncio.create_task(screen_claim_text(self._llm, claim))
        try:
            answer = await self._investigate(run)
        except BaseException:
            screening.cancel()
            raise
        screen = await screening
        run.usage = screen.usage.add(run.usage)
        if screen.injection_suspected:
            run.flags.insert(0, INJECTION_SUSPECTED)

        if answer is not None and answer.claim_id != claim.claim_id:
            logger.warning("agent_answer_for_wrong_claim", extra={"ctx_answer_claim_id": answer.claim_id})
            answer = None
        if answer is not None:
            logger.info(
                "agent_answer",
                extra={
                    "ctx_llm_outcome": answer.coverage_outcome.value,
                    "ctx_self_reported_confidence": answer.self_reported_confidence,
                },
            )

        return AgentRun(
            answer=answer,
            coverage=CoverageRuleResult.model_validate(run.results["check_coverage_rules"]),
            fraud=FraudRiskResult.model_validate(run.results["score_fraud_risk"]),
            retrieved=run.retrieved,
            steps=run.steps,
            guardrail_flags=run.flags,
            used_fallback=answer is None,
            usage=run.usage,
        )

    async def _investigate(self, run: _RunState) -> AgentAnswer | None:
        """The ReAct loop. Returns the model's answer (None if it gave no
        valid one) and guarantees the two deterministic results are in run.results."""
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=build_user_message(run.claim)),
        ]
        answer: AgentAnswer | None = None

        async with self._new_executor() as tools:
            try:
                for _ in range(_MAX_ITERATIONS):
                    response = await self._llm.generate(messages, tools=TOOL_SCHEMAS, response_format=AgentAnswer)
                    run.usage = run.usage.add(response.usage)
                    if not response.tool_calls:
                        answer = response.parsed if isinstance(response.parsed, AgentAnswer) else None
                        if answer is None:
                            logger.warning("agent_no_valid_answer", extra={"ctx_refusal": response.refusal})
                        break
                    messages.append(ChatMessage(role="assistant", tool_calls=response.tool_calls))
                    # The calls of one turn don't depend on each other, so they run
                    # concurrently; gather keeps call order, so each result lines up
                    # with its tool_call_id.
                    results = await asyncio.gather(*(self._call_tool(tools, call, run) for call in response.tool_calls))
                    for call, result in zip(response.tool_calls, results, strict=True):
                        messages.append(ChatMessage(role="tool", tool_call_id=call.id, content=json.dumps(result)))
                else:
                    logger.warning("agent_max_iterations_reached", extra={"ctx_max": _MAX_ITERATIONS})
            except LLMProviderError:
                logger.warning("agent_llm_unavailable_using_fallback")

            # Safety net: the decision policy always needs these two deterministic results.
            await self._ensure_result(tools, "check_coverage_rules", run)
            await self._ensure_result(tools, "score_fraud_risk", run)
        return answer

    async def _call_tool(self, tools: ToolExecutor, call: ToolCall, run: _RunState) -> dict:
        """Executes one tool call and records it. Errors are returned to the
        model as data (so it can recover) rather than failing the run."""
        if violation := check_tool_arguments(call.name, call.arguments, run.claim):
            logger.warning("guardrail_tool_argument_violation", extra={"ctx_tool": call.name})
            if TOOL_ARGUMENT_VIOLATION not in run.flags:
                run.flags.append(TOOL_ARGUMENT_VIOLATION)
            await run.add_step(
                ToolStep(
                    tool=call.name,
                    arguments=call.arguments,
                    ok=False,
                    latency_ms=0,
                    started_ms=run.elapsed_ms(),
                    error=violation,
                )
            )
            return {"error": violation}

        started_ms = run.elapsed_ms()
        started = time.perf_counter()
        with observe(call.name, as_type="tool", input=call.arguments) as obs:
            try:
                result = await tools.execute(call.name, call.arguments)
                error = None
            except ToolExecutionError as exc:
                result, error = {"error": exc.message}, exc.message
            obs.update(output=result)
        latency_ms = round((time.perf_counter() - started) * 1000)

        await run.add_step(
            ToolStep(
                tool=call.name,
                arguments=call.arguments,
                ok=error is None,
                latency_ms=latency_ms,
                started_ms=started_ms,
                error=error,
            )
        )
        logger.info("tool_call", extra={"ctx_tool": call.name, "ctx_ok": error is None, "ctx_latency_ms": latency_ms})
        if error is None:
            run.results[call.name] = result
            if call.name == "retrieve_policy_clauses":
                for clause in result.get("clauses", []):
                    run.retrieved[clause["clause_id"]] = RetrievedClause.model_validate(clause)
        return result

    async def _ensure_result(self, tools: ToolExecutor, name: str, run: _RunState) -> dict:
        if name not in run.results:
            await self._call_tool(
                tools, ToolCall(id=f"safety-net-{name}", name=name, arguments={"claim_id": run.claim.claim_id}), run
            )
        if name not in run.results:
            raise ToolExecutionError(name, "required deterministic check could not be completed")
        return run.results[name]


class _RunState:
    """Mutable scratchpad for one run (kept out of the loop for readability)."""

    def __init__(self, claim: ClaimRequest, on_step: StepCallback | None = None) -> None:
        self.claim = claim
        self.usage = TokenUsage()
        self.steps: list[ToolStep] = []
        self.results: dict[str, dict] = {}
        self.retrieved: dict[str, RetrievedClause] = {}
        self.flags: list[str] = []
        self._on_step = on_step
        self._started = time.perf_counter()

    def elapsed_ms(self) -> int:
        return round((time.perf_counter() - self._started) * 1000)

    async def add_step(self, step: ToolStep) -> None:
        self.steps.append(step)
        if self._on_step is not None:
            await self._on_step(step)
