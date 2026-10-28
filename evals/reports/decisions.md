# Eval: decisions
_Run at 2026-10-28 07:29 UTC_

7/7 cases completed, model `gpt-4o-mini`.

| Metric | Value |
|---|---|
| Outcome accuracy | 86% |
| Review-routing recall (needed review and got it) | 100% |
| Review-routing precision | 100% |
| Citation validity (cited clauses that were retrieved) | 78% |
| Citation relevance (cites a labelled clause) | 71% |
| Injection resistance (1 attacks) | 100% |
| Latency p50 / p95 | 8.4s / 15.8s |
| Cost per claim (avg) | $0.0007 |

| Case tag | Outcome accuracy | Cases |
|---|---|---|
| boundary | 100% | 1 |
| clean | 100% | 3 |
| disguised_exclusion | 0% | 1 |
| exclusion | 100% | 1 |
| injection | 100% | 1 |
| smoke | 86% | 7 |
| waiting_period | 100% | 1 |
