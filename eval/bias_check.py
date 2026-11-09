"""Source-level breakdown of the metrics (does quality depend on the kind of source?).

The corpus is English-only. This module evaluates retrieval performance per source domain
(Wikipedia, official documentation/company portals, technical articles, culinary guides)
and identifies qualitative failure examples (graph missed what vector found, and vice versa).

Reads ``eval/results.json`` written by ``run_eval``. Run: ``python -m eval.bias_check``
Output: ``eval/bias_report.json`` (+ printed table).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from data_scripts import config
from data_scripts.utils import read_jsonl

from . import metrics
from .run_eval import ARCHS, per_arch, pooled_scores

HERE = Path(__file__).parent


def question_sources(results: list[dict]) -> dict[int, str]:
    """Domain of the source holding each question's relevant chunk(s)."""
    uri = {c["chunk_id"]: urlparse(c["source_uri"]).netloc for c in read_jsonl(config.CHUNKS_PATH)}
    out = {}
    for r in results:
        doms = sorted({uri.get(cid, "unknown") for cid in r["relevant"]})
        out[r["id"]] = doms[0] if doms else "unknown"
    return out


def by_source(results: list[dict], k: int) -> dict:
    src = question_sources(results)
    groups = defaultdict(list)
    for r in results:
        groups[src[r["id"]]].append(r)
    table = {}
    for dom, sub in sorted(groups.items()):
        table[dom] = {}
        for arch in ARCHS:
            m = metrics.summarize(per_arch(sub, arch), k)
            s, y = pooled_scores(sub, arch)
            m["auc_roc"] = metrics.auc_roc(s, y) if arch != "graph" else None
            table[dom][arch] = m
    return table


def failure_examples(results: list[dict], limit: int = 5) -> dict:
    """Questions where one retriever found a relevant chunk and the other did not."""
    graph_miss, vector_miss = [], []
    for r in results:
        rel = set(r["relevant"])
        v = rel & set(r["arch"]["vector"]["retrieved"])
        g = rel & set(r["arch"]["graph"]["retrieved"])
        item = {"id": r["id"], "question": r["question"]}
        if v and not g:
            graph_miss.append(item)
        if g and not v:
            vector_miss.append(item)
    return {
        "graph_missed_vector_found": graph_miss[:limit],
        "vector_missed_graph_found": vector_miss[:limit],
        "counts": {
            "graph_missed_vector_found": len(graph_miss),
            "vector_missed_graph_found": len(vector_miss),
        },
    }


def run(k: int = config.TOP_K) -> dict:
    results = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    table = by_source(results, k)
    report = {"by_source": table, "failure_examples": failure_examples(results)}
    (HERE / "bias_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for dom, archs in table.items():
        for a, m in archs.items():
            auc_str = f"{m['auc_roc']:.3f}" if isinstance(m.get('auc_roc'), (int, float)) else "N/A"
            print(
                f"{dom[:28]:28s} n={m['n']:2d} {a:7s} P={m['precision']:.3f} R={m['recall']:.3f} "
                f"F1={m['f1']:.3f} MRR={m['mrr']:.3f} AUC={auc_str}"
            )
    print("Failure counts:", report["failure_examples"]["counts"])
    return report


if __name__ == "__main__":
    run()
