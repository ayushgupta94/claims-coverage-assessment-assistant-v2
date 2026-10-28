"""Persistence for the product catalog, issued policies and the RAG corpus.

Collections:
  - `product_versions`  product rules (exclusions, waiting period)
  - `issued_policies`   one document per customer's purchased policy
  - `policy_sections`   parent chunks: whole policy sections (context for the LLM)
  - `policy_clauses`    child chunks: individual clauses + embeddings (what search matches)

Search runs inside MongoDB Atlas using two search indexes on `policy_clauses`
(Atlas M0 allows three):
  - `clause_vector_index`  vector index (HNSW / ANN, cosine) on `embedding`
  - `clause_text_index`    full-text (BM25) index on `title` + `text`
Both are pre-filtered by `product_version_id`, so a claim only ever retrieves
the wording its own policy was sold under.
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.operations import SearchIndexModel

from app.domain.models import IssuedPolicy, PolicyClause, PolicyProductVersion, PolicySection

VECTOR_INDEX = "clause_vector_index"
TEXT_INDEX = "clause_text_index"


class PolicyRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._product_versions = db["product_versions"]
        self._issued_policies = db["issued_policies"]
        self._sections = db["policy_sections"]
        self._clauses = db["policy_clauses"]

    async def ensure_indexes(self) -> None:
        await self._product_versions.create_index("product_version_id", unique=True)
        await self._issued_policies.create_index("policy_id", unique=True)
        await self._sections.create_index("section_id", unique=True)
        await self._clauses.create_index("clause_id", unique=True)
        await self._clauses.create_index([("product_version_id", 1), ("chunk_hash", 1)])

    async def ensure_search_indexes(self, embedding_dimensions: int) -> list[str]:
        """Creates the Atlas vector + text search indexes if missing. Returns the
        names created. Atlas builds them asynchronously (about a minute)."""
        existing = {index["name"] async for index in self._clauses.list_search_indexes()}
        wanted = [
            SearchIndexModel(
                name=VECTOR_INDEX,
                type="vectorSearch",
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": embedding_dimensions,
                            "similarity": "cosine",
                        },
                        {"type": "filter", "path": "product_version_id"},
                    ]
                },
            ),
            SearchIndexModel(
                name=TEXT_INDEX,
                type="search",
                definition={
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "title": {"type": "string", "analyzer": "lucene.english"},
                            "text": {"type": "string", "analyzer": "lucene.english"},
                            "product_version_id": {"type": "token"},
                        },
                    }
                },
            ),
        ]
        created = []
        for model in wanted:
            name = model.document["name"]
            if name not in existing:
                await self._clauses.create_search_index(model)
                created.append(name)
        return created

    # -- product versions and issued policies --------------------------------
    async def upsert_product_version(self, version: PolicyProductVersion) -> None:
        await self._product_versions.update_one(
            {"product_version_id": version.product_version_id}, {"$set": version.model_dump()}, upsert=True
        )

    async def get_product_version(self, product_version_id: str) -> PolicyProductVersion | None:
        doc = await self._product_versions.find_one({"product_version_id": product_version_id})
        return PolicyProductVersion.model_validate(doc) if doc else None

    async def upsert_issued_policy(self, policy: IssuedPolicy) -> None:
        await self._issued_policies.update_one(
            {"policy_id": policy.policy_id}, {"$set": policy.model_dump()}, upsert=True
        )

    async def get_issued_policy(self, policy_id: str) -> IssuedPolicy | None:
        doc = await self._issued_policies.find_one({"policy_id": policy_id})
        return IssuedPolicy.model_validate(doc) if doc else None

    # -- ingestion -------------------------------------------------------------
    async def get_embeddings_by_hash(self, product_version_id: str) -> dict[str, list[float]]:
        """chunk_hash -> embedding for chunks already stored: the embedding cache."""
        cursor = self._clauses.find(
            {"product_version_id": product_version_id, "chunk_hash": {"$exists": True}, "embedding": {"$ne": None}},
            {"chunk_hash": 1, "embedding": 1},
        )
        return {doc["chunk_hash"]: doc["embedding"] async for doc in cursor}

    async def replace_sections(self, product_version_id: str, sections: list[PolicySection]) -> None:
        await self._sections.delete_many({"product_version_id": product_version_id})
        if sections:
            await self._sections.insert_many([section.model_dump() for section in sections])

    async def upsert_clause(self, clause: PolicyClause) -> None:
        await self._clauses.update_one({"clause_id": clause.clause_id}, {"$set": clause.model_dump()}, upsert=True)

    async def delete_clauses_not_in(self, product_version_id: str, keep_clause_ids: list[str]) -> int:
        result = await self._clauses.delete_many(
            {"product_version_id": product_version_id, "clause_id": {"$nin": keep_clause_ids}}
        )
        return result.deleted_count

    # -- retrieval -------------------------------------------------------------
    async def vector_search(
        self, query_vector: list[float], *, product_version_id: str, limit: int
    ) -> list[tuple[PolicyClause, float]]:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": limit * 10,  # ANN: explore more nodes than we return
                    "limit": limit,
                    "filter": {"product_version_id": product_version_id},
                }
            },
            {"$project": {"_id": 0, "embedding": 0, "score": {"$meta": "vectorSearchScore"}}},
        ]
        return [(PolicyClause.model_validate(doc), doc["score"]) async for doc in self._clauses.aggregate(pipeline)]

    async def text_search(self, query: str, *, product_version_id: str, limit: int) -> list[PolicyClause]:
        pipeline = [
            {
                "$search": {
                    "index": TEXT_INDEX,
                    "compound": {
                        "must": [{"text": {"query": query, "path": ["title", "text"]}}],
                        "filter": [{"equals": {"path": "product_version_id", "value": product_version_id}}],
                    },
                }
            },
            {"$limit": limit},
            {"$project": {"_id": 0, "embedding": 0}},
        ]
        return [PolicyClause.model_validate(doc) async for doc in self._clauses.aggregate(pipeline)]

    async def get_sections(self, section_ids: list[str]) -> list[PolicySection]:
        cursor = self._sections.find({"section_id": {"$in": section_ids}}, {"_id": 0})
        return [PolicySection.model_validate(doc) async for doc in cursor]

    async def get_clauses(self, product_version_id: str) -> list[PolicyClause]:
        cursor = self._clauses.find({"product_version_id": product_version_id}, {"_id": 0})
        return [PolicyClause.model_validate(doc) async for doc in cursor]
