"""End-to-end test of the real app on a real port.

Runs create_app() under uvicorn so everything is real -- routes, access-code
auth, the MCP server and the agent's MCP client talking to it over loopback
HTTP, the LangGraph workflow and the decision policy -- except three things
that need external services: the LLM (scripted), embeddings (fake) and the
Atlas-only $vectorSearch/$search stages (replaced with an in-memory search).
"""

import asyncio
import threading
import time
from datetime import UTC, datetime

import httpx
import pytest
import uvicorn
from langgraph.checkpoint.memory import InMemorySaver
from mongomock_motor import AsyncMongoMockClient

from app.config import Settings
from app.db.repositories.policy_repository import PolicyRepository
from app.domain.models import IssuedPolicy, PolicyClause, PolicyProductVersion, PolicySection
from app.main import create_app
from tests.fakes import FakeEmbedder, FakeLLM, final_answer, tool_call

PORT = 8931
BASE_URL = f"http://127.0.0.1:{PORT}"
CODE = {"X-Access-Code": "test-code"}
CLAUSE = PolicyClause(
    clause_id="AUTO-GOLD-V1-2.1",
    product_version_id="AUTO-GOLD-V1",
    section_id="AUTO-GOLD-V1-S2",
    title="2.1 Collision Damage",
    text="The insurer will pay the reasonable cost of repair following a collision.",
    page=2,
    chunk_hash="h",
)


class Mongo:
    def __init__(self) -> None:
        self.db = AsyncMongoMockClient()["test_e2e"]

    async def ping(self) -> bool:
        return True

    def close(self) -> None:
        pass


async def _in_memory_vector_search(self, query_vector, *, product_version_id, limit):
    return [(clause, 0.9) for clause in await self.get_clauses(product_version_id)][:limit]


async def _no_text_search(self, query, *, product_version_id, limit):
    return []


async def _seed(db) -> None:
    policies = PolicyRepository(db)
    await policies.upsert_product_version(
        PolicyProductVersion(product_version_id="AUTO-GOLD-V1", product_id="AUTO-GOLD", policy_type="auto")
    )
    await policies.upsert_issued_policy(
        IssuedPolicy(
            policy_id="POL-1",
            customer_id="CUST-1",
            product_version_id="AUTO-GOLD-V1",
            sum_insured=1_000_000,
            inception_date=datetime(2023, 1, 1, tzinfo=UTC),
        )
    )
    await policies.upsert_clause(CLAUSE)
    await policies.replace_sections(
        "AUTO-GOLD-V1",
        [
            PolicySection(
                section_id="AUTO-GOLD-V1-S2", product_version_id="AUTO-GOLD-V1", title="2. Cover", text="..", page=2
            )
        ],
    )


def agent_script(claim_id: str, llm_outcome: str) -> list:
    return [
        tool_call("retrieve_policy_clauses", {"query": "collision damage", "policy_id": "POL-1"}),
        tool_call("check_coverage_rules", {"claim_id": claim_id}),
        tool_call("score_fraud_risk", {"claim_id": claim_id}),
        final_answer(claim_id=claim_id, outcome=llm_outcome),
    ]


@pytest.fixture(scope="module")
def server():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(PolicyRepository, "vector_search", _in_memory_vector_search)
    monkeypatch.setattr(PolicyRepository, "text_search", _no_text_search)

    mongo = Mongo()
    asyncio.run(_seed(mongo.db))
    # CLAIM-OK: the LLM agrees with the rules (auto-approved). CLAIM-DISPUTED: the LLM says
    # not_covered although every rule passes -> the rules win and a human must review.
    llm = FakeLLM(agent_script("CLAIM-OK", "covered") + agent_script("CLAIM-DISPUTED", "not_covered"))
    settings = Settings(_env_file=None, openai_api_key="unused", access_codes="test-code", api_port=PORT)
    app = create_app(settings, mongo_database=mongo, llm=llm, embedder=FakeEmbedder(), checkpointer=InMemorySaver())

    uv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    for _ in range(50):
        try:
            if httpx.get(f"{BASE_URL}/health/live").status_code == 200:
                break
        except httpx.TransportError:
            time.sleep(0.1)
    yield
    uv.should_exit = True
    thread.join(timeout=5)
    monkeypatch.undo()


def claim_payload(claim_id: str) -> dict:
    return {
        "claim_id": claim_id,
        "policy_id": "POL-1",
        "customer_id": "CUST-1",
        "claim_type": "collision",
        "description": "Collided with another vehicle at a junction",
        "amount": 15000,
        "incident_date": "2024-06-01T00:00:00Z",
        "filed_at": "2024-06-02T00:00:00Z",
    }


def test_requests_without_access_code_are_rejected(server):
    assert httpx.post(f"{BASE_URL}/claims", json=claim_payload("X")).status_code == 401
    assert httpx.get(f"{BASE_URL}/reviews").status_code == 401
    assert httpx.post(f"{BASE_URL}/mcp/", json={}).status_code == 401
    assert httpx.get(f"{BASE_URL}/health/live").status_code == 200


def test_claim_is_assessed_end_to_end_through_real_mcp(server):
    response = httpx.post(f"{BASE_URL}/claims", json=claim_payload("CLAIM-OK"), headers=CODE, timeout=30)

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "approved"
    assert body["decision"]["coverage_outcome"] == "covered"
    assert [c["clause_id"] for c in body["decision"]["citations"]] == ["AUTO-GOLD-V1-2.1"]
    assert [s["tool"] for s in body["steps"]] == ["retrieve_policy_clauses", "check_coverage_rules", "score_fraud_risk"]
    assert all(s["ok"] for s in body["steps"])  # tools really ran via MCP against mongomock
    assert response.headers["X-Request-ID"]


def test_disputed_claim_goes_to_the_review_queue_and_reviewer_resumes_it(server):
    body = httpx.post(f"{BASE_URL}/claims", json=claim_payload("CLAIM-DISPUTED"), headers=CODE, timeout=30).json()
    assert body["status"] == "under_review"
    assert body["decision"]["coverage_outcome"] == "covered"  # rules win over the LLM
    assert "llm_contradicts_rules" in body["decision"]["review_reasons"]

    queue = httpx.get(f"{BASE_URL}/reviews", headers=CODE).json()
    assert [c["claim"]["claim_id"] for c in queue] == ["CLAIM-DISPUTED"]

    review = {"action": "override", "outcome": "partially_covered", "reason": "Only bumper repair", "reviewer": "QA"}
    reviewed = httpx.post(f"{BASE_URL}/claims/CLAIM-DISPUTED/review", json=review, headers=CODE, timeout=30).json()
    assert reviewed["status"] == "approved"
    assert reviewed["decision"]["coverage_outcome"] == "partially_covered"
    assert [h["action"] for h in reviewed["decision_history"]] == ["assessed", "escalated", "overridden"]


def test_error_responses(server):
    unknown = claim_payload("CLAIM-404") | {"policy_id": "NOPE"}
    assert httpx.post(f"{BASE_URL}/claims", json=unknown, headers=CODE).json()["error_code"] == "policy_not_found"
    assert httpx.post(f"{BASE_URL}/claims", json={"claim_id": "X"}, headers=CODE).status_code == 422
    assert httpx.get(f"{BASE_URL}/claims/NOPE", headers=CODE).status_code == 404
