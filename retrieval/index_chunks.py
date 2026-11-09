"""Embed ``chunks.jsonl`` and load it into the pgvector ``chunks`` table.

Run: ``python -m retrieval.index_chunks``
"""
from __future__ import annotations

import psycopg2
from pgvector.psycopg2 import register_vector

from data_scripts import config
from data_scripts.utils import get_logger, read_jsonl

from .embed import embed

logger = get_logger(__name__)
DIM = 384  # all-MiniLM-L6-v2; paraphrase-multilingual-MiniLM-L12-v2 is also 384


def connect():
    conn = psycopg2.connect(config.PGVECTOR_DSN)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def run(chunks_path=None, dim: int = DIM, batch: int = 64) -> int:
    chunks = read_jsonl(chunks_path or config.CHUNKS_PATH)
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS chunks")
        cur.execute(
            f"""CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY, source_id TEXT, source_uri TEXT, title TEXT,
                section_heading TEXT, language TEXT, text TEXT, embedding VECTOR({dim}))"""
        )
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        vecs = embed([c["text"] for c in part])
        with conn.cursor() as cur:
            for c, v in zip(part, vecs):
                cur.execute(
                    "INSERT INTO chunks VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (c["chunk_id"], c["source_id"], c["source_uri"], c["title"],
                     c.get("section_heading", ""), c["language"], c["text"], v),
                )
        conn.commit()
    conn.close()
    logger.info("indexed %d chunks", len(chunks))
    return len(chunks)


if __name__ == "__main__":
    run()
