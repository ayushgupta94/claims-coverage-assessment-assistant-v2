"""Parsing, chunking and the ingestion pipeline (against the real policy PDF)."""

from pathlib import Path

import pytest
from mongomock_motor import AsyncMongoMockClient

from app.db.repositories.policy_repository import PolicyRepository
from app.domain.models import PolicyProductVersion
from app.ingestion.chunker import chunk_by_structure, chunk_fixed_size
from app.ingestion.pdf_parser import PageLine, extract_lines
from app.ingestion.pipeline import IngestionPipeline
from tests.fakes import FakeEmbedder

AUTO_PDF = Path(__file__).resolve().parents[2] / "data" / "policies" / "auto_gold_v1.pdf"


def test_parser_removes_running_headers_and_page_footers():
    lines = extract_lines(AUTO_PDF)
    texts = [line.text for line in lines]
    assert not any(text.startswith("Page ") for text in texts)
    assert not any(text.endswith("- Policy Wording") for text in texts)
    assert lines[-1].page > 1


def test_structure_chunking_follows_policy_numbering():
    sections, clauses = chunk_by_structure(extract_lines(AUTO_PDF), product_version_id="AUTO-GOLD-V1", max_chars=1200)

    assert [s.title for s in sections][3] == "4. Exclusions"
    racing = next(c for c in clauses if c.clause_id == "AUTO-GOLD-V1-4.1")
    assert racing.title == "4.1 Racing and Speed Testing"
    assert racing.section_id == "AUTO-GOLD-V1-S4"
    assert "racing" in racing.text and "4.1" not in racing.text
    assert "4.1 Racing and Speed Testing" in sections[3].text  # parent holds its children


def test_long_clauses_are_split_on_sentences():
    long_text = " ".join(f"Sentence number {i} of the clause." for i in range(60))
    lines = [PageLine(1, "1. Cover"), PageLine(1, "1.1 Long Clause"), PageLine(1, long_text)]
    _, clauses = chunk_by_structure(lines, product_version_id="P", max_chars=300)
    assert len(clauses) > 1
    assert [c.clause_id for c in clauses][:2] == ["P-1.1-p1", "P-1.1-p2"]
    assert all(len(c.text) <= 300 for c in clauses)


def test_fixed_size_chunks_overlap():
    chunks = chunk_fixed_size(extract_lines(AUTO_PDF), product_version_id="P", size=500, overlap=100)
    assert len(chunks) > 5
    assert chunks[0].text[-60:].strip()[:30] in chunks[1].text


@pytest.fixture
async def repository():
    repo = PolicyRepository(AsyncMongoMockClient()["test_ingestion"])
    await repo.ensure_indexes()
    await repo.upsert_product_version(
        PolicyProductVersion(product_version_id="AUTO-GOLD-V1", product_id="AUTO-GOLD", policy_type="auto")
    )
    return repo


async def test_reingesting_the_same_pdf_embeds_nothing_and_duplicates_nothing(repository):
    embedder = FakeEmbedder()
    pipeline = IngestionPipeline(repository, embedder, max_chunk_chars=1200)

    first = await pipeline.ingest(AUTO_PDF, product_version_id="AUTO-GOLD-V1")
    calls_after_first = len(embedder.embedded_texts)
    second = await pipeline.ingest(AUTO_PDF, product_version_id="AUTO-GOLD-V1")

    assert first.embedded == first.chunks == 28
    assert second.embedded == 0 and second.reused == 28
    assert len(embedder.embedded_texts) == calls_after_first
    assert len(await repository.get_clauses("AUTO-GOLD-V1")) == 28
    # The section title is embedded with the clause (contextual chunk header).
    assert any(text.startswith("4. Exclusions > 4.1 Racing") for text in embedder.embedded_texts)


async def test_ingesting_for_an_unknown_product_version_fails(repository):
    with pytest.raises(ValueError):
        await IngestionPipeline(repository, FakeEmbedder(), max_chunk_chars=1200).ingest(
            AUTO_PDF, product_version_id="NOPE-V9"
        )
