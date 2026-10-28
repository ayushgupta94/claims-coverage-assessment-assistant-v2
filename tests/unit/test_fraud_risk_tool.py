from datetime import UTC, datetime, timedelta

import pytest
from mongomock_motor import AsyncMongoMockClient

from app.config import Settings
from app.db.repositories.claim_repository import ClaimRepository
from app.db.repositories.policy_repository import PolicyRepository
from app.domain.models import ClaimRequest, FraudRiskLevel, IssuedPolicy, PolicyProductVersion
from app.tools.fraud_risk_tool import FraudRiskScoringInput, score_fraud_risk


@pytest.fixture
def db():
    return AsyncMongoMockClient()["test_fraud"]


@pytest.fixture
async def repos(db):
    policy_repo = PolicyRepository(db)
    claim_repo = ClaimRepository(db)
    await policy_repo.ensure_indexes()
    await claim_repo.ensure_indexes()
    return policy_repo, claim_repo


@pytest.fixture
def settings():
    return Settings(_env_file=None, openai_api_key="unused")


def make_product_version(**overrides) -> PolicyProductVersion:
    defaults = dict(
        product_version_id="AUTO-GOLD-V1",
        product_id="AUTO-GOLD",
        policy_type="auto",
        excluded_claim_types=[],
        waiting_period_days=0,
    )
    defaults.update(overrides)
    return PolicyProductVersion(**defaults)


def make_issued_policy(**overrides) -> IssuedPolicy:
    defaults = dict(
        policy_id="POL-1",
        customer_id="CUST-1",
        product_version_id="AUTO-GOLD-V1",
        sum_insured=1_000_000.0,
        inception_date=datetime(2024, 1, 1, tzinfo=UTC),
    )
    defaults.update(overrides)
    return IssuedPolicy(**defaults)


def make_claim(**overrides) -> ClaimRequest:
    defaults = dict(
        claim_id="CLAIM-1",
        policy_id="POL-1",
        customer_id="CUST-1",
        claim_type="collision",
        description="Minor fender bender",
        amount=10_000.0,
        incident_date=datetime(2024, 6, 1, tzinfo=UTC),
        filed_at=datetime(2024, 6, 2, tzinfo=UTC),
    )
    defaults.update(overrides)
    return ClaimRequest(**defaults)


async def _seed(policy_repo, product_version=None, issued_policy=None):
    await policy_repo.upsert_product_version(product_version or make_product_version())
    await policy_repo.upsert_issued_policy(issued_policy or make_issued_policy())


async def test_low_risk_claim(repos, settings):
    policy_repo, claim_repo = repos
    await _seed(policy_repo)
    claim = make_claim()
    await claim_repo.create_claim(claim)

    result = await score_fraud_risk(settings, claim_repo, policy_repo, FraudRiskScoringInput(claim_id=claim.claim_id))

    assert result.risk_level == FraudRiskLevel.LOW
    assert result.signals == []


async def test_high_amount_claim_raises_risk(repos, settings):
    policy_repo, claim_repo = repos
    await _seed(policy_repo, issued_policy=make_issued_policy(sum_insured=2_000_000.0))
    claim = make_claim(amount=settings.fraud_high_amount_threshold + 1)
    await claim_repo.create_claim(claim)

    result = await score_fraud_risk(settings, claim_repo, policy_repo, FraudRiskScoringInput(claim_id=claim.claim_id))

    assert result.risk_score > 0
    assert any("high-value threshold" in s for s in result.signals)


async def test_early_claim_after_inception_raises_risk(repos, settings):
    policy_repo, claim_repo = repos
    inception = datetime(2024, 1, 1, tzinfo=UTC)
    await _seed(policy_repo, issued_policy=make_issued_policy(inception_date=inception))
    claim = make_claim(incident_date=inception + timedelta(days=5))
    await claim_repo.create_claim(claim)

    result = await score_fraud_risk(settings, claim_repo, policy_repo, FraudRiskScoringInput(claim_id=claim.claim_id))

    assert any("policy inception" in s for s in result.signals)


async def test_frequent_claims_raise_risk_to_high(repos, settings):
    policy_repo, claim_repo = repos
    await _seed(policy_repo)
    claim = make_claim(amount=settings.fraud_high_amount_threshold + 1)
    await claim_repo.create_claim(claim)

    for i in range(settings.fraud_frequency_claim_count_threshold):
        await claim_repo.create_claim(
            make_claim(
                claim_id=f"HIST-{i}",
                claim_type="collision",
                amount=5_000.0,
                incident_date=claim.incident_date - timedelta(days=10),
                filed_at=claim.filed_at - timedelta(days=10),
            )
        )

    result = await score_fraud_risk(settings, claim_repo, policy_repo, FraudRiskScoringInput(claim_id=claim.claim_id))

    assert result.risk_level == FraudRiskLevel.HIGH
    assert any("claims filed on this policy" in s for s in result.signals)
