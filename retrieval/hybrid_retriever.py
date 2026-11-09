"""Architecture 3: hybrid (vector text + graph facts) retrieval, one Gemini call."""
from __future__ import annotations

from . import graph_retriever, vector_retriever
from .gemini_client import generate

PROMPT = (
    "You answer questions strictly from the supplied evidence. Cite text passages as [n]. "
    "Use the knowledge-graph facts to connect entities across passages. "
    "If neither contains the answer, reply that the source has no evidence.\n\n"
    "[RETRIEVED TEXT]\n{context}\n\n[KNOWLEDGE GRAPH FACTS]\n{facts}\n\n"
    "Question: {question}\nAnswer:"
)


def retrieve(query: str, k: int | None = None) -> dict:
    chunks = vector_retriever.retrieve(query, k)
    try:
        triples = graph_retriever.retrieve(query)
    except Exception:  # graph DB down: degrade to vector-only rather than fail
        triples = []
    return {"chunks": chunks, "triples": triples}


def answer(query: str, k: int | None = None) -> dict:
    ev = retrieve(query, k)
    prompt = PROMPT.format(
        context=vector_retriever.format_context(ev["chunks"]),
        facts=graph_retriever.format_facts(ev["triples"]) or "(none)",
        question=query,
    )
    return {"architecture": "hybrid", "answer": generate(prompt), **ev}
