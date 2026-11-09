"""Embedding-model sweep on the baseline (300/50) chunks. Output: results_embeddings.csv.

Only local sentence-transformers models are swept (no paid/quota APIs).

Run: ``python -m tuning.sweep_embeddings [--limit 20]``
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from data_scripts import config
from data_scripts.utils import read_jsonl
from retrieval.embed import embed

from .sweep_chunking import GOLDEN, hit

OUT = Path(__file__).with_name("results_embeddings.csv")
MODELS = ["all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1", "all-mpnet-base-v2"]


def run(limit: int = 20, k: int | None = None) -> list[dict]:
    k = k or config.TOP_K
    qa = json.loads(GOLDEN.read_text(encoding="utf-8"))[:limit]
    chunks = read_jsonl(config.CHUNKS_PATH)
    rows = []
    for name in MODELS:
        t0 = time.perf_counter()
        mat = np.array(embed([c["text"] for c in chunks], name))
        index_s = time.perf_counter() - t0
        qv = np.array(embed([q["question"] for q in qa], name))
        precs = []
        for q, v in zip(qa, qv):
            top = np.argsort(-(mat @ v))[:k]
            p = sum(hit(chunks[i]["text"], q["answers"]) for i in top) / k
            precs.append(p)
        rows.append({"model": name, "dim": mat.shape[1], f"precision@{k}": round(sum(precs) / len(precs), 4),
                     "index_seconds": round(index_s, 1)})
        print(rows[-1])
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    run(ap.parse_args().limit)

