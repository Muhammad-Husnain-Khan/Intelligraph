"""Load data/processed/chunks.jsonl incrementally as origin=library: ``python -m scripts.seed_library``."""
from __future__ import annotations

import sys

from data_scripts import config
from data_scripts.utils import read_jsonl
from retrieval import store
from retrieval.embed import embed
from retrieval.graph_build import triples


def main() -> int:
    if not config.CHUNKS_PATH.exists():
        print(f"{config.CHUNKS_PATH} does not exist. Generate it first with:\n    python -m data_scripts.run_pipeline")
        return 0
    chunks = read_jsonl(config.CHUNKS_PATH)
    store.init()
    by_source: dict[str, list[dict]] = {}
    for c in chunks:
        by_source.setdefault(c["source_id"], []).append(c)
    known = {s["source_id"]: s for s in store.list_sources()}
    done = skipped = 0
    for sid, group in by_source.items():
        if known.get(sid, {}).get("status") == "ready" and known[sid]["chunks"] == len(group):
            skipped += 1
            continue
        first = group[0]
        store.upsert_source(sid, title=first["title"], uri=first["source_uri"], origin="library",
                            kind=first.get("source_type", "library"), status="running", error=None)
        vecs: list[list[float]] = []
        for i in range(0, len(group), 64):
            vecs += embed([c["text"] for c in group[i:i + 64]])
        store.add_chunks(group, vecs, "library")
        store.add_edges(sid, [t for c in group for t in triples(c)])
        store.upsert_source(sid, status="ready", chunks=len(group), entities=store.count_entities(sid))
        done += 1
        print(f"seeded {first['title'][:60]!r}: {len(group)} chunks")
    print(f"done: {done} sources loaded, {skipped} already present")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
