"""Retrieval metrics. All functions are pure and unit-tested offline.

Definitions (per question):
  context precision = retrieved chunks that are relevant / retrieved
  context recall    = required chunks retrieved / required chunks
  F1                = harmonic mean of the two
  Recall@K, MRR     = as promised in the Phase 1 report plan
AUC-ROC is computed over (similarity score, is_relevant) pairs pooled across
questions; it is only defined for architectures that produce a score (vector, hybrid).
"""
from __future__ import annotations

from typing import Sequence


def precision(retrieved: Sequence[str], relevant: set[str]) -> float:
    return sum(r in relevant for r in retrieved) / len(retrieved) if retrieved else 0.0


def recall(retrieved: Sequence[str], relevant: set[str]) -> float:
    return len(relevant & set(retrieved)) / len(relevant) if relevant else 0.0


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return recall(list(retrieved)[:k], relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def auc_roc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Rank-based AUC (Mann-Whitney); None if only one class is present."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def roc_points(scores: Sequence[float], labels: Sequence[int]) -> list[tuple[float, float]]:
    """(FPR, TPR) pairs sweeping the similarity cutoff, for plotting."""
    P = sum(labels)
    N = len(labels) - P
    if not P or not N:
        return []
    pts = [(0.0, 0.0)]
    for t in sorted(set(scores), reverse=True):
        tp = sum(s >= t and y for s, y in zip(scores, labels))
        fp = sum(s >= t and not y for s, y in zip(scores, labels))
        pts.append((fp / N, tp / P))
    return pts


def summarize(per_question: list[dict], k: int) -> dict:
    """Average per-question dicts with keys retrieved, relevant (set)."""
    n = len(per_question) or 1
    ps = [precision(q["retrieved"], q["relevant"]) for q in per_question]
    rs = [recall(q["retrieved"], q["relevant"]) for q in per_question]
    return {
        "n": len(per_question),
        "precision": sum(ps) / n,
        "recall": sum(rs) / n,
        "f1": sum(f1(p, r) for p, r in zip(ps, rs)) / n,
        f"recall@{k}": sum(recall_at_k(q["retrieved"], q["relevant"], k) for q in per_question) / n,
        "mrr": sum(reciprocal_rank(q["retrieved"], q["relevant"]) for q in per_question) / n,
    }

