"""Ingest policy wording PDFs into the RAG corpus.

    python scripts/ingest.py --all
    python scripts/ingest.py data/policies/auto_gold_v1.pdf --product-version AUTO-GOLD-V1

Re-running is safe and free for unchanged wording: chunks whose content hash
is already stored are not re-embedded. New wording for a product should be
ingested under a new product version id (after adding its rules to
data/product_versions.json and re-running seed_db.py), so policies already
issued under the old version keep retrieving the wording they were sold with.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from app.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.db.mongo_client import MongoDatabase  # noqa: E402
from app.db.repositories.policy_repository import PolicyRepository  # noqa: E402
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402
from app.retrieval.embeddings import Embedder  # noqa: E402


async def main(jobs: list[tuple[Path, str]]) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    mongo = MongoDatabase(settings)
    pipeline = IngestionPipeline(
        PolicyRepository(mongo.db), Embedder(settings), max_chunk_chars=settings.max_chunk_chars
    )
    for pdf_path, product_version_id in jobs:
        report = await pipeline.ingest(pdf_path, product_version_id=product_version_id)
        print(report.model_dump_json())
    mongo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", nargs="?", type=Path, help="policy wording PDF")
    parser.add_argument("--product-version", help="product_version_id the wording belongs to")
    parser.add_argument("--all", action="store_true", help="ingest every PDF listed in data/product_versions.json")
    args = parser.parse_args()

    if args.all:
        versions = json.loads((ROOT / "data" / "product_versions.json").read_text(encoding="utf-8"))
        jobs = [(ROOT / "data" / v["wording_pdf"], v["product_version_id"]) for v in versions]
    elif args.pdf and args.product_version:
        jobs = [(args.pdf, args.product_version)]
    else:
        parser.error("give a PDF and --product-version, or --all")
    asyncio.run(main(jobs))
