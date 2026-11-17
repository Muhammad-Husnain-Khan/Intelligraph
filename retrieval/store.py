"""Shared connections and scoped, incremental storage (Postgres+pgvector and Neo4j).

Upserts only: this module never drops a table or wipes the graph.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from functools import lru_cache

from data_scripts import config

DIM = 384
_lock = threading.Lock()
_pool = None
_driver = None


def init() -> None:
    """Open the Postgres pool and the single Neo4j driver, create schema if missing."""
    global _pool, _driver
    from neo4j import GraphDatabase
    from pgvector.psycopg2 import register_vector
    from psycopg2.pool import ThreadedConnectionPool
    import psycopg2

    with _lock:
        if _pool is None:
            boot = psycopg2.connect(config.PGVECTOR_DSN)
            boot.autocommit = True
            with boot.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            boot.close()
            _pool = ThreadedConnectionPool(1, 8, config.PGVECTOR_DSN)
            conn = _pool.getconn()
            try:
                register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute(f"""CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY, source_id TEXT, source_uri TEXT, title TEXT,
                        section_heading TEXT, language TEXT, text TEXT, embedding VECTOR({DIM}))""")
                    cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS origin TEXT DEFAULT 'library'")
                    cur.execute("CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source_id)")
                    cur.execute("""CREATE TABLE IF NOT EXISTS sources (
                        source_id TEXT PRIMARY KEY, title TEXT, uri TEXT, origin TEXT, kind TEXT,
                        status TEXT, error TEXT, chunks INT DEFAULT 0, entities INT DEFAULT 0,
                        created_at TIMESTAMPTZ DEFAULT now())""")
                    # backfill rows for chunks indexed by the older batch CLI (idempotent)
                    cur.execute("""INSERT INTO sources (source_id, title, uri, origin, kind, status, chunks)
                                   SELECT source_id, min(title), min(source_uri), 'library', 'library', 'ready', count(*)
                                   FROM chunks WHERE source_id IS NOT NULL GROUP BY source_id
                                   ON CONFLICT (source_id) DO NOTHING""")
                    cur.execute("UPDATE chunks SET origin='library' WHERE origin IS NULL")
                conn.commit()
            finally:
                _pool.putconn(conn)
        if _driver is None:
            _driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
            with _driver.session() as s:
                s.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
                # Backfill source_id on legacy edges (chunk_id prefix before '-') and update entity counts
                s.run("MATCH ()-[r:RELATED_TO]->() WHERE r.source_id IS NULL AND r.chunk_id IS NOT NULL "
                      "SET r.source_id = split(r.chunk_id, '-')[0]")
                rows = s.run("""MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity) WHERE r.source_id IS NOT NULL
                                WITH r.source_id AS sid, collect(DISTINCT a.name) + collect(DISTINCT b.name) AS ns
                                UNWIND ns AS n RETURN sid, count(DISTINCT n) AS cnt""")
                ent_counts = {r["sid"]: int(r["cnt"]) for r in rows}
            if ent_counts and _pool is not None:
                conn = _pool.getconn()
                try:
                    with conn.cursor() as cur:
                        for sid, cnt in ent_counts.items():
                            cur.execute("UPDATE sources SET entities=%s WHERE source_id=%s AND (entities IS NULL OR entities=0)", (cnt, sid))
                    conn.commit()
                finally:
                    _pool.putconn(conn)


def close() -> None:
    global _pool, _driver
    if _pool is not None:
        _pool.closeall()
        _pool = None
    if _driver is not None:
        _driver.close()
        _driver = None


@contextmanager
def pg():
    """Pooled connection with pgvector registered; commits on success."""
    from pgvector.psycopg2 import register_vector

    if _pool is None:
        init()
    conn = _pool.getconn()  # type: ignore[union-attr]
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)  # type: ignore[union-attr]


def neo():
    if _driver is None:
        init()
    return _driver


# --------------------------------------------------------------------------- sources
def upsert_source(source_id: str, **fields) -> None:
    cols = ["title", "uri", "origin", "kind", "status", "error", "chunks", "entities"]
    vals = {c: fields.get(c) for c in cols if c in fields}
    with pg() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO sources (source_id) VALUES (%s) ON CONFLICT DO NOTHING", (source_id,))
        if vals:
            sets = ", ".join(f"{c}=%s" for c in vals)
            cur.execute(f"UPDATE sources SET {sets} WHERE source_id=%s", (*vals.values(), source_id))


def list_sources() -> list[dict]:
    with pg() as conn, conn.cursor() as cur:
        cur.execute("""SELECT source_id, title, uri, origin, kind, status, error, chunks, entities, created_at
                       FROM sources ORDER BY created_at""")
        keys = ("source_id", "title", "uri", "origin", "kind", "status", "error", "chunks", "entities", "created_at")
        rows = [dict(zip(keys, r)) for r in cur.fetchall()]
    for r in rows:
        r["created_at"] = r["created_at"].isoformat() if r["created_at"] else None
    return rows


def counts() -> dict:
    with pg() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        chunks = cur.fetchone()[0]
        cur.execute("SELECT count(DISTINCT source_id) FROM chunks")
        sources = cur.fetchone()[0]
    return {"sources": sources, "chunks": chunks}


def add_chunks(chunks: list[dict], vectors: list[list[float]], origin: str) -> None:
    with pg() as conn, conn.cursor() as cur:
        for c, v in zip(chunks, vectors):
            cur.execute(
                """INSERT INTO chunks (chunk_id, source_id, source_uri, title, section_heading,
                                       language, text, embedding, origin)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (chunk_id) DO UPDATE SET text=EXCLUDED.text, embedding=EXCLUDED.embedding,
                       title=EXCLUDED.title, section_heading=EXCLUDED.section_heading, origin=EXCLUDED.origin""",
                (c["chunk_id"], c["source_id"], c["source_uri"], c["title"], c.get("section_heading", ""),
                 c["language"], c["text"], v, origin),
            )


def add_edges(source_id: str, triples: list[tuple[str, str, str]]) -> int:
    """MERGE entities and RELATED_TO edges (tagged with chunk_id and source_id)."""
    if not triples:
        return 0
    rows = [{"a": a, "b": b, "cid": cid, "sid": source_id} for a, b, cid in triples]
    with neo().session() as s:
        s.run("""UNWIND $rows AS r
                 MERGE (a:Entity {name: r.a}) MERGE (b:Entity {name: r.b})
                 MERGE (a)-[rel:RELATED_TO {chunk_id: r.cid}]->(b)
                 SET rel.source_id = r.sid""", rows=rows)
    return len(rows)


def count_entities(source_id: str) -> int:
    with neo().session() as s:
        rec = s.run("""MATCH (a:Entity)-[r:RELATED_TO {source_id: $sid}]-(b:Entity)
                       WITH collect(DISTINCT a) + collect(DISTINCT b) AS ns
                       UNWIND ns AS n RETURN count(DISTINCT n) AS n""", sid=source_id).single()
    return int(rec["n"]) if rec else 0


def delete_source(source_id: str) -> bool:
    """Remove chunks, edges and orphan entities for one source."""
    with neo().session() as s:
        s.run("MATCH ()-[r:RELATED_TO {source_id: $sid}]-() DELETE r", sid=source_id)
        s.run("MATCH (e:Entity) WHERE NOT (e)--() DELETE e")
    with pg() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE source_id=%s", (source_id,))
        cur.execute("DELETE FROM sources WHERE source_id=%s", (source_id,))
        return cur.rowcount > 0


# --------------------------------------------------------------------------- embeddings
@lru_cache(maxsize=256)
def _embed_query_cached(text: str) -> tuple[float, ...]:
    from retrieval.embed import embed

    return tuple(embed([text])[0])


def embed_query(text: str) -> list[float]:
    return list(_embed_query_cached(text))


def preload_embedder() -> None:
    from retrieval.embed import embed

    embed(["warm up"])
