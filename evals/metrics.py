"""Evaluation metrics -- small pure functions, so each number is easy to explain.

Retrieval (is the right policy wording found?)
  hit@k     1 if any relevant clause is in the top k, else 0
  recall@k  share of the relevant clauses found in the top k
  MRR       1 / rank of the first relevant clause (0 if none) -- rewards ranking it first
Decisions (is the system's output right and safe?)
  accuracy        predicted outcome == expected outcome ("partially_covered" counts as covered)
  review recall   of claims a human must see, the share that was escalated -- the costly error is a miss
  review precision of escalated claims, the share that truly needed a human
"""

from __future__ import annotations

import math


def hit_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if relevant & set(ranked_ids[:k]) else 0.0


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    return len(relevant & set(ranked_ids[:k])) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(ranked_ids, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1)]


def same_outcome(predicted: str, expected: str) -> bool:
    normalise = {"partially_covered": "covered"}
    return normalise.get(predicted, predicted) == normalise.get(expected, expected)


def review_recall_precision(predicted: list[bool], expected: list[bool]) -> tuple[float, float]:
    true_positive = sum(p and e for p, e in zip(predicted, expected, strict=True))
    recall = true_positive / sum(expected) if any(expected) else 1.0
    precision = true_positive / sum(predicted) if any(predicted) else 1.0
    return recall, precision


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
