"""Architecture 2: graph-only retrieval (Neo4j traversal) + Gemini generation.

There is no similarity score here, so AUC-ROC does not apply to this architecture.
"""
from __future__ import annotations

import re

from neo4j import GraphDatabase, Query

from data_scripts import config

from .gemini_client import generate

PROMPT = (
    "Answer the question using ONLY the knowledge-graph facts below. "
    "If they do not contain the answer, say you cannot find it.\n\n"
    "[KNOWLEDGE GRAPH FACTS]\n{facts}\n\nQuestion: {question}\nAnswer:"
)
_TOKEN = re.compile(r"[\w\-]{3,}", re.UNICODE)


CYPHER_QUERY = """
MATCH (seed:Entity)
WHERE any(t IN $terms WHERE toLower(seed.name) CONTAINS t)
WITH seed, size([t IN $terms WHERE toLower(seed.name) CONTAINS t]) AS hits
ORDER BY hits DESC LIMIT 10
MATCH p=(seed)-[:RELATED_TO*1..2]-(o:Entity)
UNWIND relationships(p) AS r
WITH DISTINCT startNode(r).name AS s, endNode(r).name AS o,
              r.chunk_id AS chunk_id, hits
RETURN s, o, chunk_id ORDER BY hits DESC LIMIT $k
"""

CYPHER_1_HOP = """
MATCH (seed:Entity)
WHERE any(t IN $terms WHERE toLower(seed.name) CONTAINS t)
WITH seed, size([t IN $terms WHERE toLower(seed.name) CONTAINS t]) AS hits
ORDER BY hits DESC LIMIT 10
MATCH p=(seed)-[:RELATED_TO]-(o:Entity)
UNWIND relationships(p) AS r
WITH DISTINCT startNode(r).name AS s, endNode(r).name AS o,
              r.chunk_id AS chunk_id, hits
RETURN s, o, chunk_id ORDER BY hits DESC LIMIT $k
"""


def _terms(query: str) -> list[str]:
    return list(dict.fromkeys(t.lower() for t in _TOKEN.findall(query)))


def retrieve(query: str, k: int | None = None, hops: int = 2) -> list[dict]:
    """Match query terms to entity nodes and return traversed triples with chunk_ids."""
    k = k or config.TOP_K * 4
    terms = _terms(query)
    if not terms:
        return []
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    cypher = CYPHER_1_HOP if hops == 1 else CYPHER_QUERY
    with driver.session() as s:
        res = s.run(Query(cypher), parameters={"terms": terms, "k": k})
        rows = [r.data() for r in res]
    driver.close()
    return rows


def format_facts(triples: list[dict]) -> str:
    return "\n".join(f"{t['s']} -[RELATED_TO]-> {t['o']} (chunk {t['chunk_id']})" for t in triples)


def answer(query: str, k: int | None = None) -> dict:
    triples = retrieve(query, k)
    text = generate(PROMPT.format(facts=format_facts(triples) or "(none)", question=query))
    return {"architecture": "graph", "answer": text, "triples": triples,
            "chunk_ids": list(dict.fromkeys(t["chunk_id"] for t in triples))}
