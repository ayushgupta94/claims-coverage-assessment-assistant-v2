"""RAG ingestion pipeline: policy wording PDF -> searchable chunks in Atlas.

    parse  -> extract clean text lines with page numbers  (pdf_parser.py)
    chunk  -> structure-aware clause chunks + parent sections (chunker.py)
    embed  -> embed only chunks whose content hash is new  (embedding cache)
    store  -> upsert chunks, replace sections, remove stale chunks

Idempotent: re-ingesting an unchanged PDF re-embeds nothing and leaves the
same documents in place. Versioned: wording belongs to a product version, so
ingesting new wording under a new product_version_id never changes what
already-issued policies retrieve.
"""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel

from app.core.logging import get_logger
from app.db.repositories.policy_repository import PolicyRepository
from app.ingestion.chunker import chunk_by_structure, embedding_text
from app.ingestion.pdf_parser import extract_lines
from app.retrieval.embeddings import Embedder

logger = get_logger(__name__)


class IngestionReport(BaseModel):
    product_version_id: str
    lines: int
    sections: int
    chunks: int
    embedded: int  # chunks sent to the embedding API
    reused: int  # chunks whose embedding came from the cache
    removed: int  # stale chunks deleted (wording that no longer exists)


class IngestionPipeline:
    def __init__(self, repository: PolicyRepository, embedder: Embedder, *, max_chunk_chars: int) -> None:
        self._repository = repository
        self._embedder = embedder
        self._max_chunk_chars = max_chunk_chars

    async def ingest(self, pdf_path: str | Path, *, product_version_id: str) -> IngestionReport:
        if await self._repository.get_product_version(product_version_id) is None:
            raise ValueError(f"Unknown product version {product_version_id!r}: seed its rules first.")

        started = time.perf_counter()
        lines = extract_lines(pdf_path)
        self._log_stage("parse", product_version_id, started, lines=len(lines), file=Path(pdf_path).name)

        started = time.perf_counter()
        sections, clauses = chunk_by_structure(
            lines, product_version_id=product_version_id, max_chars=self._max_chunk_chars
        )
        self._log_stage("chunk", product_version_id, started, sections=len(sections), chunks=len(clauses))

        started = time.perf_counter()
        cached = await self._repository.get_embeddings_by_hash(product_version_id)
        to_embed = [clause for clause in clauses if clause.chunk_hash not in cached]
        section_titles = {section.section_id: section.title for section in sections}
        vectors = []
        if to_embed:
            vectors = await self._embedder.embed(
                [embedding_text(section_titles[c.section_id], c.title, c.text) for c in to_embed]
            )
        new_embeddings = {clause.chunk_hash: vector for clause, vector in zip(to_embed, vectors, strict=True)}
        for clause in clauses:
            clause.embedding = new_embeddings.get(clause.chunk_hash) or cached[clause.chunk_hash]
        self._log_stage(
            "embed", product_version_id, started, embedded=len(to_embed), reused=len(clauses) - len(to_embed)
        )

        started = time.perf_counter()
        await self._repository.replace_sections(product_version_id, sections)
        for clause in clauses:
            await self._repository.upsert_clause(clause)
        removed = await self._repository.delete_clauses_not_in(product_version_id, [c.clause_id for c in clauses])
        self._log_stage("store", product_version_id, started, upserted=len(clauses), removed=removed)

        return IngestionReport(
            product_version_id=product_version_id,
            lines=len(lines),
            sections=len(sections),
            chunks=len(clauses),
            embedded=len(to_embed),
            reused=len(clauses) - len(to_embed),
            removed=removed,
        )

    @staticmethod
    def _log_stage(stage: str, product_version_id: str, started: float, **fields: object) -> None:
        logger.info(
            "ingestion_stage",
            extra={
                "ctx_stage": stage,
                "ctx_product_version_id": product_version_id,
                "ctx_latency_ms": round((time.perf_counter() - started) * 1000),
                **{f"ctx_{key}": value for key, value in fields.items()},
            },
        )
