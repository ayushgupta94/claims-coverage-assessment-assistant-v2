"""Daily run counter -- the one cost guard on the public demo.

Each claim assessment spends OpenAI tokens, so runs are capped per UTC day
(MAX_RUNS_PER_DAY). The check-and-increment is a single atomic Mongo update,
so concurrent requests can't race past the cap.
"""

from __future__ import annotations

from datetime import UTC, datetime

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError


class UsageRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._usage = db["daily_usage"]

    async def try_consume_run(self, max_runs_per_day: int) -> bool:
        """Increment today's counter if it is below the cap. Returns False when
        the cap is reached (the conditional upsert then hits the unique _id)."""
        if max_runs_per_day <= 0:
            return False
        today = datetime.now(UTC).date().isoformat()
        try:
            await self._usage.update_one(
                {"_id": today, "runs": {"$lt": max_runs_per_day}},
                {"$inc": {"runs": 1}},
                upsert=True,
            )
        except DuplicateKeyError:
            return False
        return True
