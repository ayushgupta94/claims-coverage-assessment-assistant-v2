"""Test doubles shared by the test suite. None of these ship with the app."""

from datetime import UTC, datetime

from app.agent.guardrails import InjectionVerdict
from app.agent.llm_client import LLMResponse, ToolCall
from app.domain.models import AgentAnswer, ClaimRequest, CoverageOutcome, TokenUsage

USAGE = TokenUsage(llm_calls=1, prompt_tokens=100, completion_tokens=20, cost_usd=0.00003)


class FakeLLM:
    """Replays a script of agent responses. The injection classifier call is
    answered separately (by `injection_verdict`) so scripts stay readable."""

    model = "fake-model"

    def __init__(self, script: list[LLMResponse], *, injection_verdict: bool = False) -> None:
        self._script = list(script)
        self._injection_verdict = injection_verdict
        self.agent_calls = 0

    async def generate(self, messages, *, tools=None, response_format=None, purpose="agent") -> LLMResponse:
        if purpose == "injection_classifier":
            verdict = InjectionVerdict(is_injection=self._injection_verdict, reason="scripted")
            return LLMResponse(parsed=verdict, usage=USAGE)
        self.agent_calls += 1
        return self._script.pop(0)


class FakeToolExecutor:
    """Mirrors ToolExecutor's `async with` + execute() interface with canned results."""

    def __init__(self, results_by_tool: dict[str, dict]) -> None:
        self._results = results_by_tool
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self) -> "FakeToolExecutor":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def execute(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        return self._results[name]


class FakeEmbedder:
    """Deterministic 3-dim vectors from keywords -- no API calls."""

    def __init__(self) -> None:
        self.embedded_texts: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts += texts
        return [[float("theft" in t.lower()), float("collision" in t.lower()), 1.0] for t in texts]


def make_claim(**overrides) -> ClaimRequest:
    defaults = dict(
        claim_id="CLAIM-1",
        policy_id="POL-1",
        customer_id="CUST-1",
        claim_type="collision",
        description="Collided with another vehicle at a junction",
        amount=20_000.0,
        incident_date=datetime(2024, 6, 1, tzinfo=UTC),
        filed_at=datetime(2024, 6, 2, tzinfo=UTC),
    )
    defaults.update(overrides)
    return ClaimRequest(**defaults)


def tool_call(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=f"call-{name}", name=name, arguments=arguments)], usage=USAGE)


def final_answer(claim_id: str = "CLAIM-1", outcome: str = "covered", clause_ids=("AUTO-GOLD-V1-2.1",)) -> LLMResponse:
    answer = AgentAnswer(
        claim_id=claim_id,
        coverage_outcome=CoverageOutcome(outcome),
        supporting_clause_ids=list(clause_ids),
        reasoning="Collision damage is covered under AUTO-GOLD-V1-2.1 and all rules passed.",
        self_reported_confidence=0.9,
    )
    return LLMResponse(content=answer.model_dump_json(), parsed=answer, usage=USAGE)


CLEAN_TOOL_RESULTS = {
    "retrieve_policy_clauses": {
        "clauses": [
            {
                "clause_id": "AUTO-GOLD-V1-2.1",
                "title": "2.1 Collision Damage",
                "page": 2,
                "text": "The insurer will pay the reasonable cost of repair...",
                "section_id": "AUTO-GOLD-V1-S2",
                "score": 0.032,
                "vector_score": 0.92,
            }
        ],
        "sections": [],
    },
    "lookup_claim_history": {"history": []},
    "check_coverage_rules": {"is_covered": True, "violated_rules": [], "notes": "All coverage rules satisfied."},
    "score_fraud_risk": {"risk_level": "low", "risk_score": 0.0, "signals": []},
}
