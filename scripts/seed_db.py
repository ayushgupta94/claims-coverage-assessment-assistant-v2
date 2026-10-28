"""Seed MongoDB: product version rules, issued policies, prior claim history,
regular indexes and the two Atlas search indexes. Idempotent.

Policy wording is loaded separately by the ingestion pipeline:
    python scripts/seed_db.py
    python scripts/ingest.py --all
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from app.config import get_settings  # noqa: E402
from app.core.exceptions import ConflictError  # noqa: E402
from app.db.mongo_client import MongoDatabase  # noqa: E402
from app.db.repositories.claim_repository import ClaimRepository  # noqa: E402
from app.db.repositories.policy_repository import PolicyRepository  # noqa: E402
from app.domain.models import ClaimRequest, ClaimStatus, IssuedPolicy, PolicyProductVersion  # noqa: E402

DATA_DIR = ROOT / "data"


def load(name: str) -> list[dict]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


async def main() -> None:
    settings = get_settings()
    mongo = MongoDatabase(settings)
    policies = PolicyRepository(mongo.db)
    claims = ClaimRepository(mongo.db)
    await policies.ensure_indexes()
    await claims.ensure_indexes()

    for raw in load("product_versions.json"):
        await policies.upsert_product_version(PolicyProductVersion.model_validate(raw))
    for raw in load("seed_issued_policies.json"):
        await policies.upsert_issued_policy(IssuedPolicy.model_validate(raw))

    created = 0
    for raw in load("seed_claims.json"):
        try:
            await claims.create_claim(ClaimRequest.model_validate(raw), status=ClaimStatus(raw["status"]))
            created += 1
        except ConflictError:
            pass  # already seeded

    new_indexes = await policies.ensure_search_indexes(settings.embedding_dimensions)
    print(f"Seeded product versions, issued policies and {created} new history claims.")
    print(f"Search indexes created: {new_indexes or 'none (already present)'}")
    mongo.close()


if __name__ == "__main__":
    asyncio.run(main())
