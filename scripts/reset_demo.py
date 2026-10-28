"""Reset the public demo: delete claims submitted through the UI (claim_id
DEMO-*) and their LangGraph workflow state, plus workflow state left behind
by claims that no longer exist (e.g. eval runs, which delete their claims).

Every demo claim stays in the database, so repeated demo runs on the same
policy build up claim history and eventually trip the fraud-frequency signal.
Run this before showing the app. Seed history claims are never touched.

    python scripts/reset_demo.py          # dry run: shows what would be deleted
    python scripts/reset_demo.py --yes    # deletes it
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from app.config import get_settings  # noqa: E402
from app.db.mongo_client import MongoDatabase  # noqa: E402

DEMO_PREFIX = "DEMO-"  # set by the UI (src/app/static/index.html)
CHECKPOINT_COLLECTIONS = ("workflow_checkpoints", "workflow_checkpoint_writes")  # see main._mongo_checkpointer


async def main(delete: bool) -> None:
    settings = get_settings()
    mongo = MongoDatabase(settings)
    db = mongo.db

    demo_claims = {"claim_id": {"$regex": f"^{DEMO_PREFIX}"}}
    demo_count = await db["claims"].count_documents(demo_claims)
    existing = set(await db["claims"].distinct("claim_id", {"claim_id": {"$not": {"$regex": f"^{DEMO_PREFIX}"}}}))
    # Workflow threads are keyed by claim_id; any thread without a surviving claim goes.
    threads = set(await db[CHECKPOINT_COLLECTIONS[0]].distinct("thread_id"))
    stale_threads = sorted(threads - existing)

    print(f"Demo claims ({DEMO_PREFIX}*): {demo_count}")
    print(f"Workflow threads without a surviving claim: {len(stale_threads)}")
    if not delete:
        print("Dry run: nothing deleted. Re-run with --yes to delete.")
        mongo.close()
        return

    deleted = (await db["claims"].delete_many(demo_claims)).deleted_count
    for name in CHECKPOINT_COLLECTIONS:
        await db[name].delete_many({"thread_id": {"$in": stale_threads}})
    print(f"Deleted {deleted} demo claims and the workflow state of {len(stale_threads)} threads.")
    mongo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="actually delete (default is a dry run)")
    asyncio.run(main(parser.parse_args().yes))
