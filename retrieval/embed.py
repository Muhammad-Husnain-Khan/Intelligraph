"""Embedding wrapper (sentence-transformers, local, no API quota)."""
from __future__ import annotations

from functools import lru_cache

from data_scripts import config


@lru_cache(maxsize=4)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    """Return L2-normalised embeddings, so cosine similarity == dot product."""
    model = _model(model_name or config.EMBEDDING_MODEL)
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]
