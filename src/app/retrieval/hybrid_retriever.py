r"""Query-time retrieval: hybrid search + Reciprocal Rank Fusion + parent context.

    query --> embed --> Atlas $vectorSearch (ANN, semantic)  --\
          \-------->   Atlas $search      (BM25, keywords)  ---+--> RRF --> top-k clauses
                                                                            --> their parent sections

Why hybrid: vector search finds paraphrases ("car was stolen" ~ "theft of the
insured vehicle"); keyword search finds exact terms (claim types, "FIR",
"waiting period"). Each search returns `rag_candidates` results (optimise for
recall), RRF merges the two rankings, and only `rag_top_k` reach the LLM.

RRF scores a clause by its rank in each list: score = sum(1 / (k + rank)).
It needs no score normalisation between the two searches, which is why it is
the standard way to fuse lexical and vector results.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.config import Settings
from app.core.logging import get_logger
from app.core.tracing import observe
from app.db.repositories.policy_repository import PolicyRepository
from app.domain.models import ClauseMatch, PolicyClause, PolicySection
from app.retrieval.embeddings import Embedder

logger = get_logger(__name__)

RRF_K = 60  # the constant from the original RRF paper; dampens the weight of top ranks


class RetrievalResult(BaseModel):
    matches: list[ClauseMatch]
    sections: list[PolicySection]  # parent context, one per distinct section


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """rankings: lists of ids, best first. Returns id -> fused score."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, repository: PolicyRepository, embedder: Embedder, settings: Settings) -> None:
        self._repository = repository
        self._embedder = embedder
        self._mode = settings.retrieval_mode
        self._top_k = settings.rag_top_k
        self._candidates = settings.rag_candidates

    async def retrieve(self, query: str, *, product_version_id: str, mode: str | None = None) -> RetrievalResult:
        mode = mode or self._mode
        with observe("retrieval", as_type="retriever", input={"query": query, "mode": mode}) as obs:
            [query_vector] = await self._embedder.embed([query])
            vector_hits = await self._repository.vector_search(
                query_vector, product_version_id=product_version_id, limit=self._candidates
            )
            text_hits: list[PolicyClause] = []
            if mode == "hybrid":
                text_hits = await self._repository.text_search(
                    query, product_version_id=product_version_id, limit=self._candidates
                )

            clauses = {clause.clause_id: clause for clause, _ in vector_hits} | {c.clause_id: c for c in text_hits}
            vector_scores = {clause.clause_id: score for clause, score in vector_hits}
            fused = reciprocal_rank_fusion(
                [[clause.clause_id for clause, _ in vector_hits], [clause.clause_id for clause in text_hits]]
            )
            top_ids = sorted(fused, key=fused.get, reverse=True)[: self._top_k]
            matches = [
                ClauseMatch(
                    clause=clauses[clause_id].model_copy(update={"embedding": None}),
                    score=round(fused[clause_id], 5),
                    vector_score=vector_scores.get(clause_id),
                )
                for clause_id in top_ids
            ]
            section_ids = list(dict.fromkeys(match.clause.section_id for match in matches))
            sections = await self._repository.get_sections(section_ids)
            obs.update(output={"clause_ids": top_ids})

        logger.info(
            "retrieval",
            extra={
                "ctx_mode": mode,
                "ctx_vector_hits": len(vector_hits),
                "ctx_text_hits": len(text_hits),
                "ctx_returned": [m.clause.clause_id for m in matches],
            },
        )
        return RetrievalResult(matches=matches, sections=sections)
