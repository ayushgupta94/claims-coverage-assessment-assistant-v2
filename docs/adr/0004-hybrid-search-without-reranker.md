# ADR 0004: Hybrid search with RRF, without a reranker

**Status:** accepted

## Context
Claims describe incidents in everyday words ("car was nicked from the car
park"); policies use formal wording ("theft of the insured vehicle") and exact
terms ("FIR", "waiting period"). The corpus is small (about 80 clauses) and
must stay within Atlas M0 (three search indexes) and a 512 MB container.

## Decision
- Chunk by the policy's own numbering: one chunk per clause, with its section
  as parent context (small-to-big retrieval).
- Run Atlas `$vectorSearch` (semantic, ANN) and Atlas `$search` (BM25) for the
  same query, both pre-filtered to the claim's product version, and merge them
  with Reciprocal Rank Fusion. Return the top 4 clauses plus parent sections.
- No cross-encoder reranker.

## Consequences
- Paraphrases and exact terms are both found; RRF needs no score normalisation.
- Two of the three free search indexes are used.
- A reranker would cost latency, money or memory; the retrieval eval
  (`evals.run retrieval`) is the place to justify adding one if recall drops
  as the corpus grows.
