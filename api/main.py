"""FastAPI service: ``uvicorn api.main:app --reload``."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from retrieval import graph_retriever, hybrid_retriever, vector_retriever

app = FastAPI(title="IntelliGraph")


class Query(BaseModel):
    question: str
    k: int | None = None


def _run(fn, q: Query):
    try:
        return fn(q.question, q.k)
    except RuntimeError as exc:  # e.g. missing Gemini key
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query/vector")
def query_vector(q: Query):
    return _run(vector_retriever.answer, q)


@app.post("/query/graph")
def query_graph(q: Query):
    return _run(graph_retriever.answer, q)


@app.post("/query/hybrid")
def query_hybrid(q: Query):
    return _run(hybrid_retriever.answer, q)
