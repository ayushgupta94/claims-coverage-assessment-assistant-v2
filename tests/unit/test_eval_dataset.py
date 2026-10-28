"""Guards the golden dataset: every label must be consistent with the
deterministic layer (coverage rules + fraud score) computed from the seed data.

"Disguised exclusion" cases are the exception by design: their claim_type
passes the rules, but the policy wording excludes the loss -- only the LLM
reading the wording (and then a human) can catch them.
"""

import json
from pathlib import Path

import pytest
from mongomock_motor import AsyncMongoMockClient

from app.config import Settings
from app.db.repositories.claim_repository import ClaimRepository
from app.db.repositories.policy_repository import PolicyRepository
from app.domain.models import ClaimRequest, ClaimStatus, IssuedPolicy, PolicyProductVersion
from app.tools.coverage_rule_tool import CoverageRuleCheckInput, check_coverage_rules
from app.tools.fraud_risk_tool import FraudRiskScoringInput, score_fraud_risk

ROOT = Path(__file__).resolve().parents[2]
CASES = [json.loads(line) for line in (ROOT / "evals" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]


def load(name: str) -> list[dict]:
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
async def test_label_matches_deterministic_rules(case):
    db = AsyncMongoMockClient()[f"eval_{case['id']}"]
    policies, claims = PolicyRepository(db), ClaimRepository(db)
    await claims.ensure_indexes()
    for raw in load("product_versions.json"):
        await policies.upsert_product_version(PolicyProductVersion.model_validate(raw))
    for raw in load("seed_issued_policies.json"):
        await policies.upsert_issued_policy(IssuedPolicy.model_validate(raw))
    for raw in load("seed_claims.json"):
        await claims.create_claim(ClaimRequest.model_validate(raw), status=ClaimStatus(raw["status"]))
    claim = ClaimRequest.model_validate({**case["claim"], "claim_id": case["id"]})
    await claims.create_claim(claim)

    coverage = await check_coverage_rules(claims, policies, CoverageRuleCheckInput(claim_id=claim.claim_id))
    fraud = await score_fraud_risk(
        Settings(_env_file=None, openai_api_key="unused"),
        claims,
        policies,
        FraudRiskScoringInput(claim_id=claim.claim_id),
    )

    if "disguised_exclusion" in case["tags"]:
        assert coverage.is_covered and case["expected_outcome"] == "not_covered" and case["expected_review"]
        return
    rules_outcome = "covered" if coverage.is_covered else "not_covered"
    must_review = rules_outcome == "not_covered" or fraud.risk_level != "low" or "injection" in case["tags"]
    assert rules_outcome == case["expected_outcome"], coverage.notes
    assert must_review == case["expected_review"], fraud.signals
