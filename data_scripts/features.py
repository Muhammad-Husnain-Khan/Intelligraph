"""Feature strategy: scaling and encoding of corpus descriptors (pandas only).

NOTE: these features are for data inspection and quality analysis only. No supervised
model is trained in this project; the corpus is knowledge data for retrieval.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .utils import PipelineError, get_logger, read_jsonl

logger = get_logger(__name__)

NUMERIC_COLS = ["char_count", "word_count", "sentence_count", "paragraph_count"]


def build_features(records: list[dict]) -> pd.DataFrame:
    """Min-max scale word_count, z-score char_count, one-hot source_type and language."""
    df = pd.DataFrame(records)
    out = df[["source_id", *NUMERIC_COLS, "is_outlier"]].copy()

    wc = df["word_count"].astype(float)
    span = wc.max() - wc.min()
    out["word_count_minmax"] = (wc - wc.min()) / span if span > 0 else 0.0

    cc = df["char_count"].astype(float)
    std = cc.std(ddof=0)
    out["char_count_z"] = (cc - cc.mean()) / std if std > 0 else 0.0

    for col in ("source_type", "language"):
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        out = pd.concat([out, dummies[sorted(dummies.columns)]], axis=1)
    return out


def run() -> pd.DataFrame:
    """Build and save ``data/processed/features.csv``."""
    records = read_jsonl(config.CLEAN_RECORDS)
    if not records:
        raise PipelineError(f"No cleaned records at {config.CLEAN_RECORDS}; run cleaning first.")
    features = build_features(records)
    config.FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(config.FEATURES_CSV, index=False)
    logger.info("features: %d rows x %d cols -> %s", *features.shape, config.FEATURES_CSV)
    return features


if __name__ == "__main__":
    run()
