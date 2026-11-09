"""Architecture 1: vector-only retrieval from pgvector + Gemini generation."""
from __future__ import annotations

from data_scripts import config

from .embed import embed
from .gemini_client import generate
from .index_chunks import connect

PROMPT = (
    "Answer the question using ONLY the context below. Cite sources as [n]. "
    "If the context does not contain the answer, say you cannot find it.\n\n"
    "[RETRIEVED TEXT]\n{context}\n\nQuestion: {question}\nAnswer:"
)


def retrieve(query: str, k: int | None = None) -> list[dict]:
    """Top-k chunks with cosine similarity score and provenance fields."""
    k = k or config.TOP_K
    qv = embed([query])[0]
    conn = connect()
    with conn.cursor() as cur:
        cur.execute(
            """SELECT chunk_id, source_uri, title, section_heading, language, text,
                      1 - (embedding <=> %s::vector) AS score
               FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s""",
            (qv, qv, k),
        )
        rows = cur.fetchall()
    conn.close()
    keys = ("chunk_id", "source_uri", "title", "section_heading", "language", "text", "score")
    return [dict(zip(keys, r)) for r in rows]


def score_chunks(query: str, chunk_ids: list[str]) -> dict[str, float]:
    """Cosine similarity of the query to specific chunks (used to score graph hits)."""
    if not chunk_ids:
        return {}
    qv = embed([query])[0]
    conn = connect()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_id, 1 - (embedding <=> %s::vector) FROM chunks WHERE chunk_id = ANY(%s)",
            (qv, list(chunk_ids)),
        )
        out = {r[0]: float(r[1]) for r in cur.fetchall()}
    conn.close()
    return out


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] ({c['title']} | {c['source_uri']})\n{c['text']}" for i, c in enumerate(chunks, 1)
    )


def answer(query: str, k: int | None = None) -> dict:
    chunks = retrieve(query, k)
    text = generate(PROMPT.format(context=format_context(chunks), question=query))
    return {"architecture": "vector", "answer": text, "chunks": chunks}
