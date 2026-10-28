from evals.metrics import hit_at_k, percentile, recall_at_k, reciprocal_rank, review_recall_precision, same_outcome


def test_retrieval_metrics():
    ranked = ["A", "B", "C", "D"]
    assert hit_at_k(ranked, {"C"}, 2) == 0.0 and hit_at_k(ranked, {"C"}, 3) == 1.0
    assert recall_at_k(ranked, {"B", "Z"}, 4) == 0.5
    assert reciprocal_rank(ranked, {"C", "D"}) == 1 / 3
    assert reciprocal_rank(ranked, {"Z"}) == 0.0


def test_decision_metrics():
    assert same_outcome("partially_covered", "covered") and not same_outcome("covered", "not_covered")
    recall, precision = review_recall_precision([True, True, False, False], [True, False, True, False])
    assert (recall, precision) == (0.5, 0.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0 and percentile([1.0, 2.0, 3.0, 4.0], 95) == 4.0
