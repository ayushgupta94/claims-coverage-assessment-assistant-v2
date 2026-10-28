"""Offline evaluation against the golden dataset (evals/dataset.jsonl).

    python -m evals.run retrieval              # Atlas: vector-only vs hybrid (recall@k, MRR)
    python -m evals.run chunking               # in memory: fixed-size vs structure-aware chunks
    python -m evals.run decisions              # full pipeline over HTTP against a running app
    python -m evals.run decisions --smoke      # the 7 cases tagged "smoke"

Each command prints a Markdown scorecard and writes it to evals/reports/.
Retrieval and chunking need MONGO_URI / OPENAI_API_KEY; decisions also needs
the app running (default http://127.0.0.1:8000) and an access code in
ACCESS_CODES. Decision runs spend tokens (about $0.001 per claim with
gpt-4o-mini) and count towards MAX_RUNS_PER_DAY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.config import get_settings
from app.db.mongo_client import MongoDatabase
from app.db.repositories.policy_repository import PolicyRepository
from app.ingestion.chunker import chunk_by_structure, chunk_fixed_size, embedding_text
from app.ingestion.pdf_parser import extract_lines
from app.retrieval.embeddings import Embedder
from app.retrieval.hybrid_retriever import HybridRetriever
from evals.metrics import (
    cosine_similarity,
    hit_at_k,
    mean,
    percentile,
    recall_at_k,
    reciprocal_rank,
    review_recall_precision,
    same_outcome,
)

ROOT = Path(__file__).resolve().parent.parent
REPORTS = Path(__file__).resolve().parent / "reports"


def load_cases(smoke: bool = False) -> list[dict]:
    lines = (ROOT / "evals" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines]
    return [c for c in cases if "smoke" in c["tags"]] if smoke else cases


def query_for(case: dict) -> str:
    return f"{case['claim']['claim_type']}: {case['claim']['description']}"


def product_version_of(case: dict) -> str:
    return case["relevant_clause_ids"][0].rsplit("-", 1)[0]


def write_report(name: str, lines: list[str]) -> None:
    report = "\n".join([f"# Eval: {name}", f"_Run at {datetime.now(UTC):%Y-%m-%d %H:%M} UTC_", "", *lines]) + "\n"
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{name}.md").write_text(report, encoding="utf-8")
    print(report)


# ---------------------------------------------------------------------------
async def eval_retrieval(cases: list[dict]) -> None:
    """Same queries, same index, two retrieval modes."""
    settings = get_settings()
    mongo = MongoDatabase(settings)
    retriever = HybridRetriever(PolicyRepository(mongo.db), Embedder(settings), settings)
    k = settings.rag_top_k
    rows = ["| Mode | Hit@k | Recall@k | MRR |", "|---|---|---|---|"]
    for mode in ("vector", "hybrid"):
        hits, recalls, rrs = [], [], []
        for case in cases:
            result = await retriever.retrieve(query_for(case), product_version_id=product_version_of(case), mode=mode)
            ranked = [match.clause.clause_id for match in result.matches]
            relevant = set(case["relevant_clause_ids"])
            hits.append(hit_at_k(ranked, relevant, k))
            recalls.append(recall_at_k(ranked, relevant, k))
            rrs.append(reciprocal_rank(ranked, relevant))
        rows.append(f"| {mode} | {mean(hits):.2f} | {mean(recalls):.2f} | {mean(rrs):.2f} |")
    mongo.close()
    write_report("retrieval", [f"{len(cases)} queries, k={k}, Atlas indexes.", "", *rows])


# ---------------------------------------------------------------------------
async def eval_chunking(cases: list[dict]) -> None:
    """Fixed-size vs structure-aware chunks of the same PDFs, ranked by exact
    cosine similarity in memory (no index involved), so only chunking differs.
    A fixed-size chunk counts as relevant if it contains the start of a
    relevant clause's text."""
    settings = get_settings()
    embedder = Embedder(settings)
    versions = json.loads((ROOT / "data" / "product_versions.json").read_text(encoding="utf-8"))
    k = settings.rag_top_k

    strategies: dict[str, dict[str, list]] = {"fixed-size (800 chars, 150 overlap)": {}, "structure-aware": {}}
    clause_starts: dict[str, str] = {}
    for version in versions:
        pv = version["product_version_id"]
        lines = extract_lines(ROOT / "data" / version["wording_pdf"])
        sections, clauses = chunk_by_structure(lines, product_version_id=pv, max_chars=settings.max_chunk_chars)
        titles = {s.section_id: s.title for s in sections}
        clause_starts |= {c.clause_id: c.text[:40] for c in clauses}
        fixed = chunk_fixed_size(lines, product_version_id=pv)
        structure_vectors = await embedder.embed(
            [embedding_text(titles[c.section_id], c.title, c.text) for c in clauses]
        )
        fixed_vectors = await embedder.embed([c.text for c in fixed])
        strategies["structure-aware"][pv] = list(zip(clauses, structure_vectors, strict=True))
        strategies["fixed-size (800 chars, 150 overlap)"][pv] = list(zip(fixed, fixed_vectors, strict=True))

    query_vectors = await embedder.embed([query_for(case) for case in cases])
    rows = ["| Chunking | Chunks | Hit@k | MRR |", "|---|---|---|---|"]
    for name, by_version in strategies.items():
        hits, rrs = [], []
        for case, query_vector in zip(cases, query_vectors, strict=True):
            chunks = by_version[product_version_of(case)]
            ranked = sorted(chunks, key=lambda item: cosine_similarity(query_vector, item[1]), reverse=True)
            starts = [clause_starts[cid] for cid in case["relevant_clause_ids"]]
            # A chunk id is "relevant" if the chunk contains a relevant clause's opening text.
            ranked_ids = [c.clause_id for c, _ in ranked]
            relevant = {c.clause_id for c, _ in ranked if any(start in c.text for start in starts)}
            hits.append(hit_at_k(ranked_ids, relevant, k))
            rrs.append(reciprocal_rank(ranked_ids, relevant))
        total = sum(len(chunks) for chunks in by_version.values())
        rows.append(f"| {name} | {total} | {mean(hits):.2f} | {mean(rrs):.2f} |")
    write_report("chunking", [f"{len(cases)} queries, k={k}, exact cosine (no ANN index).", "", *rows])


# ---------------------------------------------------------------------------
async def eval_decisions(cases: list[dict], base_url: str) -> None:
    """Submits every case to the running app, then deletes it again so cases
    don't add to each other's claim-history (fraud frequency) signal."""
    settings = get_settings()
    headers = {"X-Access-Code": settings.access_code_list[0]}
    mongo = MongoDatabase(settings)
    run_tag = datetime.now(UTC).strftime("%H%M%S")
    results = []

    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=120) as client:
        for case in cases:
            claim_id = f"EVAL-{case['id']}-{run_tag}"
            started = time.perf_counter()
            response = await client.post("/claims", json={**case["claim"], "claim_id": claim_id})
            latency = time.perf_counter() - started
            await mongo.db["claims"].delete_one({"claim_id": claim_id})
            if response.status_code != 201:
                print(f"{case['id']}: HTTP {response.status_code} {response.text[:200]}")
                continue
            record = response.json()
            decision = record["decision"]
            results.append({"case": case, "decision": decision, "usage": record["usage"], "latency": latency})
            print(
                f"{case['id']}: {decision['coverage_outcome']:<17} review={decision['requires_human_review']!s:<5} "
                f"expected={case['expected_outcome']}/{case['expected_review']} reasons={decision['review_reasons']}"
            )
    mongo.close()

    correct = [same_outcome(r["decision"]["coverage_outcome"], r["case"]["expected_outcome"]) for r in results]
    review_recall, review_precision = review_recall_precision(
        [r["decision"]["requires_human_review"] for r in results], [r["case"]["expected_review"] for r in results]
    )
    cited = [(len(r["decision"]["citations"]), len(r["decision"]["dropped_citations"])) for r in results]
    citation_validity = sum(v for v, _ in cited) / max(1, sum(v + d for v, d in cited))
    citation_relevance = mean(
        [
            hit_at_k([c["clause_id"] for c in r["decision"]["citations"]], set(r["case"]["relevant_clause_ids"]), 10)
            for r in results
        ]
    )
    injection = [r for r in results if "injection" in r["case"]["tags"]]
    resisted = [
        "injection_suspected" in r["decision"]["review_reasons"]
        and same_outcome(r["decision"]["coverage_outcome"], r["case"]["expected_outcome"])
        for r in injection
    ]
    latencies = [r["latency"] for r in results]
    costs = [r["usage"]["cost_usd"] for r in results]

    by_tag: dict[str, list[bool]] = {}
    for r, ok in zip(results, correct, strict=True):
        for tag in r["case"]["tags"]:
            by_tag.setdefault(tag, []).append(ok)

    write_report(
        "decisions",
        [
            f"{len(results)}/{len(cases)} cases completed, model `{settings.openai_model}`.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Outcome accuracy | {mean(correct):.0%} |",
            f"| Review-routing recall (needed review and got it) | {review_recall:.0%} |",
            f"| Review-routing precision | {review_precision:.0%} |",
            f"| Citation validity (cited clauses that were retrieved) | {citation_validity:.0%} |",
            f"| Citation relevance (cites a labelled clause) | {citation_relevance:.0%} |",
            f"| Injection resistance ({len(injection)} attacks) | {mean(resisted):.0%} |",
            f"| Latency p50 / p95 | {percentile(latencies, 50):.1f}s / {percentile(latencies, 95):.1f}s |",
            f"| Cost per claim (avg) | ${mean(costs):.4f} |",
            "",
            "| Case tag | Outcome accuracy | Cases |",
            "|---|---|---|",
            *[f"| {tag} | {mean(oks):.0%} | {len(oks)} |" for tag, oks in sorted(by_tag.items())],
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suite", choices=["retrieval", "chunking", "decisions"])
    parser.add_argument("--smoke", action="store_true", help="only the cases tagged 'smoke'")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    selected = load_cases(args.smoke)
    if args.suite == "retrieval":
        asyncio.run(eval_retrieval(selected))
    elif args.suite == "chunking":
        asyncio.run(eval_chunking(selected))
    else:
        asyncio.run(eval_decisions(selected, args.base_url))
