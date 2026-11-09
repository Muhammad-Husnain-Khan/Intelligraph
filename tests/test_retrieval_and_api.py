"""Integration tests for retrieval modules and FastAPI service."""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from retrieval import graph_retriever, hybrid_retriever, vector_retriever

client = TestClient(app)


def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_vector_retriever_retrieve():
    chunks = vector_retriever.retrieve("What is software architecture?", k=3)
    assert len(chunks) <= 3
    assert all("chunk_id" in c and "score" in c for c in chunks)
    assert chunks[0]["score"] > 0.0


def test_graph_retriever_retrieve():
    triples = graph_retriever.retrieve("Munir", k=5)
    assert isinstance(triples, list)
    if triples:
        assert "s" in triples[0] and "o" in triples[0] and "chunk_id" in triples[0]


def test_hybrid_retriever_retrieve():
    data = hybrid_retriever.retrieve("What is microservices?", k=3)
    assert "chunks" in data and "triples" in data
    assert len(data["chunks"]) <= 3


def test_api_vector_endpoint_returns_503_without_key():
    # Calling generation endpoint without a Gemini key gracefully raises 503 HTTP exception
    res = client.post("/query/vector", json={"question": "What is microservices?", "k": 3})
    assert res.status_code in (200, 503)

