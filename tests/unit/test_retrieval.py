"""Reciprocal Rank Fusion and the hybrid retriever. Atlas $vectorSearch/$search
are replaced by a fake repository, since they only exist on Atlas."""

from app.config import Settings
from app.domain.models import PolicyClause, PolicySection
from app.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from tests.fakes import FakeEmbedder


def test_rrf_rewards_items_ranked_well_by_both_searches():
    scores = reciprocal_rank_fusion([["A", "B", "C"], ["C", "A", "D"]], k=60)
    ranked = sorted(scores, key=scores.get, reverse=True)
    assert ranked[:2] == ["A", "C"]
    assert scores["A"] == 1 / 61 + 1 / 62


def clause(clause_id: str, section: str = "S2") -> PolicyClause:
    return PolicyClause(
        clause_id=clause_id,
        product_version_id="P",
        section_id=section,
        title=clause_id,
        text=f"text {clause_id}",
        page=1,
        chunk_hash=clause_id,
        embedding=[0.1, 0.2],
    )


class FakeSearchRepository:
    def __init__(self) -> None:
        self.text_search_called = False

    async def vector_search(self, query_vector, *, product_version_id, limit):
        return [(clause("P-2.2"), 0.91), (clause("P-2.1"), 0.80), (clause("P-3.2", "S3"), 0.70)]

    async def text_search(self, query, *, product_version_id, limit):
        self.text_search_called = True
        return [clause("P-3.2", "S3"), clause("P-2.2")]

    async def get_sections(self, section_ids):
        return [PolicySection(section_id=s, product_version_id="P", title=s, text="...", page=1) for s in section_ids]


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, openai_api_key="unused", rag_top_k=2, **overrides)


async def test_hybrid_retrieval_fuses_rankings_and_returns_parent_sections():
    repo = FakeSearchRepository()
    result = await HybridRetriever(repo, FakeEmbedder(), settings()).retrieve("theft", product_version_id="P")

    assert [m.clause.clause_id for m in result.matches] == ["P-2.2", "P-3.2"]
    assert result.matches[0].vector_score == 0.91
    assert result.matches[0].clause.embedding is None  # vectors never leave the retriever
    assert {s.section_id for s in result.sections} == {"S2", "S3"}


async def test_vector_mode_skips_keyword_search():
    repo = FakeSearchRepository()
    result = await HybridRetriever(repo, FakeEmbedder(), settings(retrieval_mode="vector")).retrieve(
        "theft", product_version_id="P"
    )
    assert repo.text_search_called is False
    assert [m.clause.clause_id for m in result.matches] == ["P-2.2", "P-2.1"]
