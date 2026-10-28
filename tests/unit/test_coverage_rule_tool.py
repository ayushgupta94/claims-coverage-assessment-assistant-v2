from datetime import UTC, datetime, timedelta

import pytest
from mongomock_motor import AsyncMongoMockClient

from app.core.exceptions import ClaimValidationError, PolicyNotFoundError
from app.db.repositories.claim_repository import ClaimRepository
from app.db.repositories.policy_repository import PolicyRepository
from app.domain.models import ClaimRequest, IssuedPolicy, PolicyProductVersion
from app.tools.coverage_rule_tool import CoverageRuleCheckInput, check_coverage_rules


@pytest.fixture
def db():
    return AsyncMongoMockClient()["test_coverage_rules"]


@pytest.fixture
async def repos(db):
    policy_repo = PolicyRepository(db)
    claim_repo = ClaimRepository(db)
    await policy_repo.ensure_indexes()
    await claim_repo.ensure_indexes()
    return policy_repo, claim_repo


def make_product_version(**overrides) -> PolicyProductVersion:
    defaults = dict(
        product_version_id="AUTO-GOLD-V1",
        product_id="AUTO-GOLD",
        policy_type="auto",
        excluded_claim_types=["racing"],
        waiting_period_days=15,
    )
    defaults.update(overrides)
    return PolicyProductVersion(**defaults)


def make_issued_policy(**overrides) -> IssuedPolicy:
    defaults = dict(
        policy_id="POL-1",
        customer_id="CUST-1",
        product_version_id="AUTO-GOLD-V1",
        sum_insured=100_000.0,
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
        description="Rear-ended at a traffic light",
        amount=20_000.0,
        incident_date=datetime(2024, 3, 1, tzinfo=UTC),
    )
    defaults.update(overrides)
    return ClaimRequest(**defaults)


async def _seed(policy_repo, product_version=None, issued_policy=None):
    await policy_repo.upsert_product_version(product_version or make_product_version())
    await policy_repo.upsert_issued_policy(issued_policy or make_issued_policy())


async def test_covered_claim_passes_all_rules(repos):
    policy_repo, claim_repo = repos
    await _seed(policy_repo)
    claim = make_claim()
    await claim_repo.create_claim(claim)

    result = await check_coverage_rules(claim_repo, policy_repo, CoverageRuleCheckInput(claim_id=claim.claim_id))

    assert result.is_covered is True
    assert result.violated_rules == []


async def test_excluded_claim_type_is_not_covered(repos):
    policy_repo, claim_repo = repos
    await _seed(policy_repo, product_version=make_product_version(excluded_claim_types=["racing"]))
    claim = make_claim(claim_type="racing")
    await claim_repo.create_claim(claim)

    result = await check_coverage_rules(claim_repo, policy_repo, CoverageRuleCheckInput(claim_id=claim.claim_id))

    assert result.is_covered is False
    assert any("excluded" in rule for rule in result.violated_rules)


async def test_claim_within_waiting_period_is_not_covered(repos):
    policy_repo, claim_repo = repos
    inception = datetime(2024, 1, 1, tzinfo=UTC)
    await _seed(
        policy_repo,
        product_version=make_product_version(waiting_period_days=30),
        issued_policy=make_issued_policy(inception_date=inception),
    )
    claim = make_claim(incident_date=inception + timedelta(days=5))
    await claim_repo.create_claim(claim)

    result = await check_coverage_rules(claim_repo, policy_repo, CoverageRuleCheckInput(claim_id=claim.claim_id))

    assert result.is_covered is False
    assert any("waiting period" in rule for rule in result.violated_rules)


async def test_claim_amount_exceeding_sum_insured_is_not_covered(repos):
    policy_repo, claim_repo = repos
    await _seed(policy_repo, issued_policy=make_issued_policy(sum_insured=10_000.0))
    claim = make_claim(amount=20_000.0)
    await claim_repo.create_claim(claim)

    result = await check_coverage_rules(claim_repo, policy_repo, CoverageRuleCheckInput(claim_id=claim.claim_id))

    assert result.is_covered is False
    assert any("exceeds policy sum insured" in rule for rule in result.violated_rules)


async def test_missing_claim_raises_validation_error(repos):
    policy_repo, claim_repo = repos
    with pytest.raises(ClaimValidationError):
        await check_coverage_rules(claim_repo, policy_repo, CoverageRuleCheckInput(claim_id="does-not-exist"))


async def test_missing_issued_policy_raises_not_found(repos):
    policy_repo, claim_repo = repos
    claim = make_claim(policy_id="NO-SUCH-POLICY")
    await claim_repo.create_claim(claim)

    with pytest.raises(PolicyNotFoundError):
        await check_coverage_rules(claim_repo, policy_repo, CoverageRuleCheckInput(claim_id=claim.claim_id))


async def test_missing_product_version_raises_not_found(repos):
    policy_repo, claim_repo = repos
    # Issued policy exists but points at a product version that was never seeded.
    await policy_repo.upsert_issued_policy(make_issued_policy(product_version_id="GHOST-V1"))
    claim = make_claim()
    await claim_repo.create_claim(claim)

    with pytest.raises(PolicyNotFoundError):
        await check_coverage_rules(claim_repo, policy_repo, CoverageRuleCheckInput(claim_id=claim.claim_id))
