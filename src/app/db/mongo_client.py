"""Mongo client lifecycle.

Wraps Motor's AsyncIOMotorClient so the rest of the app depends on a small
interface rather than importing motor everywhere. Points at MongoDB Atlas
(M0 free tier) in every environment -- Atlas also hosts the vector and
full-text search indexes used by retrieval. One instance per process,
created by main.create_app() (or a script), so one connection pool.
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import Settings


class MongoDatabase:
    def __init__(self, settings: Settings) -> None:
        self._client: AsyncIOMotorClient = AsyncIOMotorClient(settings.mongo_uri)
        self._db: AsyncIOMotorDatabase = self._client[settings.mongo_db_name]

    @property
    def db(self) -> AsyncIOMotorDatabase:
        return self._db

    async def ping(self) -> bool:
        try:
            await self._client.admin.command("ping")
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()
