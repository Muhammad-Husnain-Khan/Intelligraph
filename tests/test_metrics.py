from eval import metrics as m


def test_precision_recall_f1():
    ret, rel = ["a", "b", "c", "d"], {"a", "c", "z"}
    p, r = m.precision(ret, rel), m.recall(ret, rel)
    assert p == 0.5 and abs(r - 2 / 3) < 1e-9
    assert abs(m.f1(p, r) - 2 * p * r / (p + r)) < 1e-9
    assert m.f1(0, 0) == 0


def test_rank_metrics():
    assert m.reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3
    assert m.reciprocal_rank(["x"], {"a"}) == 0
    assert m.recall_at_k(["x", "a"], {"a"}, 1) == 0 and m.recall_at_k(["x", "a"], {"a"}, 2) == 1


def test_auc():
    assert m.auc_roc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert m.auc_roc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert m.auc_roc([0.5, 0.5], [1, 0]) == 0.5
    assert m.auc_roc([0.5, 0.6], [1, 1]) is None


def test_summarize():
    s = m.summarize([{"retrieved": ["a"], "relevant": {"a"}}, {"retrieved": ["b"], "relevant": {"a"}}], 5)
    assert s["precision"] == 0.5 and s["mrr"] == 0.5

