"""Build the Neo4j knowledge graph from ``chunks.jsonl``.

Entity extraction uses lightweight rule-based NER (proper nouns, capitalized terms,
acronyms) to capture named entities. Entities co-occurring within the same sentence
are linked by ``RELATED_TO`` edges tagged with the source ``chunk_id`` so every
triple stays traceable back to original document provenance.

Run: ``python -m retrieval.graph_build``
"""
from __future__ import annotations

import itertools
import re

from neo4j import GraphDatabase

from data_scripts import config
from data_scripts.utils import get_logger, read_jsonl

logger = get_logger(__name__)

_ENTITY = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9\-]+|[A-Z]{2,})(?:\s+(?:of|the|for|and)?\s*(?:[A-Z][a-zA-Z0-9\-]+|[A-Z]{2,}))*\b"
)
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_STOP = {"The", "This", "That", "These", "It", "In", "On", "For", "And", "But", "We", "He",
         "She", "They", "A", "An", "As", "At", "By", "With", "From", "Click", "Here", "Your"}
MAX_PER_SENT = 8


def extract_entities(sentence: str) -> list[str]:
    found = []
    for m in _ENTITY.finditer(sentence):
        e = re.sub(r"\s+", " ", m.group(0)).strip()
        if e in _STOP or len(e) < 3:
            continue
        found.append(e)
    return list(dict.fromkeys(found))[:MAX_PER_SENT]


def triples(chunk: dict) -> list[tuple[str, str, str]]:
    out = []
    for sent in _SENT.split(chunk["text"]):
        ents = extract_entities(sent)
        for a, b in itertools.combinations(ents, 2):
            out.append((a, b, chunk["chunk_id"]))
    return out


def run(chunks_path=None) -> tuple[int, int]:
    chunks = read_jsonl(chunks_path or config.CHUNKS_PATH)
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    n_edges = 0
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
        s.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
        for c in chunks:
            rows = [{"a": a, "b": b, "cid": cid} for a, b, cid in triples(c)]
            if not rows:
                continue
            s.run(
                """UNWIND $rows AS r
                   MERGE (a:Entity {name: r.a}) MERGE (b:Entity {name: r.b})
                   MERGE (a)-[rel:RELATED_TO {chunk_id: r.cid}]->(b)""",
                parameters={"rows": rows},
            )
            n_edges += len(rows)
        record = s.run("MATCH (e:Entity) RETURN count(e) AS n").single()
        n_nodes = int(record["n"]) if record is not None else 0
    driver.close()
    logger.info("graph: %d entities, %d edge candidates", n_nodes, n_edges)
    return n_nodes, n_edges


if __name__ == "__main__":
    run()
