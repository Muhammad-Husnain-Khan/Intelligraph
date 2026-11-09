"""Chunk size / overlap sweep. Reuses ``chunk_data.chunk_record`` (no new chunking logic).

For each setting: re-chunk, embed in memory, and measure precision@K on the golden
subset (relevance = chunk text contains an answer string), plus latency.
Output: ``tuning/results_chunking.csv``.

Run: ``python -m tuning.sweep_chunking [--limit 20]``
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from data_scripts import config
from data_scripts.chunk_data import chunk_record
from data_scripts.utils import read_jsonl, write_jsonl
from retrieval.embed import embed

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden_qa.json"
OUT = Path(__file__).with_name("results_chunking.csv")
SETTINGS = [(150, 25), (300, 50), (600, 100)]


def hit(text: str, answers: list[str]) -> bool:
    t = text.lower()
    return any(a.lower() in t for a in answers)


def run(limit: int = 20, k: int | None = None) -> list[dict]:
    k = k or config.TOP_K
    qa = json.loads(GOLDEN.read_text(encoding="utf-8"))[:limit]
    records = read_jsonl(config.CLEAN_RECORDS)
    rows = []
    for max_words, overlap in SETTINGS:
        chunks = [c for r in records for c in chunk_record(r, max_words, overlap, 50)]
        write_jsonl(config.PROCESSED_DIR / f"chunks_{max_words}w.jsonl", chunks)
        mat = np.array(embed([c["text"] for c in chunks]))
        precs, t0 = [], time.perf_counter()
        qv = np.array(embed([q["question"] for q in qa]))
        for q, v in zip(qa, qv):
            top = np.argsort(-(mat @ v))[:k]
            precs.append(sum(hit(chunks[i]["text"], q["answers"]) for i in top) / k)
        rows.append({"max_words": max_words, "overlap": overlap, "n_chunks": len(chunks),
                     f"precision@{k}": round(sum(precs) / len(precs), 4),
                     "query_ms": round((time.perf_counter() - t0) * 1000 / len(qa), 2)})
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

