"""Run the golden set through all three architectures and compute metrics.

Relevance ground truth: a chunk is relevant if it contains any answer string of the
question (whitespace-normalised). Results are keyed by ``chunk_id`` so they stay
traceable to Phase 2 provenance.

Modes:
  default            retrieval only (no Gemini key needed) - what the metrics use
  --generate         also call Gemini for answers (needs INTELLIGRAPH_GEMINI_API_KEY)
  --api URL          query a running FastAPI instead of calling modules in-process

Run: ``python -m eval.run_eval [--generate] [--k 5]``
Output: ``eval/results.json`` and ``eval/metrics_summary.json``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from data_scripts import config
from data_scripts.utils import read_jsonl
from retrieval import graph_retriever, hybrid_retriever, vector_retriever

from . import metrics

HERE = Path(__file__).parent
ARCHS = ("vector", "graph", "hybrid")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def relevant_ids(chunks: list[dict], answers: list[str]) -> set[str]:
    return {c["chunk_id"] for c in chunks if any(a in _norm(c["text"]) for a in answers)}


def retrieve_ids(arch: str, question: str, k: int) -> list[tuple[str, float | None]]:
    """Ranked (chunk_id, similarity-or-None). Graph has no similarity score."""
    if arch == "vector":
        return [(c["chunk_id"], c["score"]) for c in vector_retriever.retrieve(question, k)]
    triples = graph_retriever.retrieve(question)
    gids = list(dict.fromkeys(t["chunk_id"] for t in triples))[:k]
    if arch == "graph":
        return [(g, None) for g in gids]
    vec = [(c["chunk_id"], c["score"]) for c in vector_retriever.retrieve(question, k)]
    seen = {c for c, _ in vec}
    extra = [g for g in gids if g not in seen]
    sc = vector_retriever.score_chunks(question, extra)
    return vec + [(g, sc.get(g)) for g in extra]


def run(k: int = config.TOP_K, generate: bool = False, limit: int | None = None) -> dict:
    qa = json.loads((HERE / "golden_qa.json").read_text(encoding="utf-8"))[:limit]
    chunks = read_jsonl(config.CHUNKS_PATH)
    answer_fn = {"vector": vector_retriever.answer, "graph": graph_retriever.answer,
                 "hybrid": hybrid_retriever.answer}
    results = []
    for q in qa:
        rel = relevant_ids(chunks, q["answers"])
        row: dict[str, Any] = {
            "id": q["id"],
            "question": q["question"],
            "language": q["language"],
            "relevant": sorted(rel),
            "arch": {},
        }
        for arch in ARCHS:
            ids = retrieve_ids(arch, q["question"], k)
            entry: dict[str, Any] = {"retrieved": [i for i, _ in ids], "scores": [s for _, s in ids]}
            if generate:
                try:
                    entry["answer"] = answer_fn[arch](q["question"], k)["answer"]
                except Exception as exc:  # keep going; record the failure
                    entry["answer_error"] = str(exc)
            row["arch"][arch] = entry
        results.append(row)
    (HERE / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = summarize(results, k)
    (HERE / "metrics_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def per_arch(results: list[dict], arch: str) -> list[dict]:
    return [{"retrieved": r["arch"][arch]["retrieved"], "relevant": set(r["relevant"])} for r in results]


def pooled_scores(results: list[dict], arch: str) -> tuple[list[float], list[int]]:
    s, y = [], []
    for r in results:
        rel = set(r["relevant"])
        for cid, sc in zip(r["arch"][arch]["retrieved"], r["arch"][arch]["scores"]):
            if sc is not None:
                s.append(sc)
                y.append(int(cid in rel))
    return s, y


def summarize(results: list[dict], k: int) -> dict:
    out = {}
    for arch in ARCHS:
        m = metrics.summarize(per_arch(results, arch), k)
        if arch == "graph":
            m["auc_roc"] = "n/a (no similarity score)"
        else:
            m["auc_roc"] = metrics.auc_roc(*pooled_scores(results, arch))
        out[arch] = m
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--limit", type=int)
    print(json.dumps(run(**vars(ap.parse_args())), indent=1))

