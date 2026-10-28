# Eval: chunking
_Run at 2026-10-28 07:15 UTC_

40 queries, k=4, exact cosine (no ANN index).

| Chunking | Chunks | Hit@k | MRR |
|---|---|---|---|
| fixed-size (800 chars, 150 overlap) | 32 | 0.85 | 0.67 |
| structure-aware | 77 | 0.82 | 0.80 |
