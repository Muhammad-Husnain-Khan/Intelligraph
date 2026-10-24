"""Cleaning + audit: raw JSONL records -> ``clean_records.jsonl`` and ``cleaning_audit.json``."""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import unicodedata
from typing import Any

from . import config
from .utils import PipelineError, get_logger, read_jsonl, write_jsonl

logger = get_logger(__name__)

REQUIRED = ("source_id", "source_uri", "text")
_SENTENCE_RE = re.compile(r"[.!?\u06D4]+")


def normalize_text(text: str) -> str:
    """NFC, null chars -> space, collapse spaces/tabs, keep at most one blank line.

    Case and Urdu characters are preserved.
    """
    text = unicodedata.normalize("NFC", text).replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def content_hash(source_uri: str, text: str) -> str:
    """SHA-256 of ``source_uri + normalized text`` (same text, different URI is kept)."""
    return hashlib.sha256((source_uri + "\n" + text).encode("utf-8")).hexdigest()


def compute_counts(text: str) -> dict[str, int]:
    """char, word, sentence (. ! ? and Urdu full stop) and paragraph counts."""
    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
    }


def iqr_fences(values: list[int]) -> tuple[float, float] | None:
    """Return ``(lower, upper)`` 1.5*IQR fences, or None with fewer than 4 values."""
    if len(values) < config.MIN_RECORDS_FOR_OUTLIERS:
        return None
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    iqr = q3 - q1
    return q1 - config.IQR_K * iqr, q3 + config.IQR_K * iqr


def clean_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run the cleaning steps in order; returns ``(clean_records, audit)``."""
    audit = {
        "raw": len(records),
        "rejected_missing": 0,
        "imputed_title": 0,
        "imputed_language": 0,
        "removed_duplicates": 0,
        "excluded_short": 0,
        "flagged_outliers": 0,
        "final": 0,
    }

    # 1. reject records missing required fields
    valid = []
    for rec in records:
        if all(isinstance(rec.get(k), str) and rec[k].strip() for k in REQUIRED):
            valid.append(dict(rec))
        else:
            audit["rejected_missing"] += 1

    # 2. impute optional fields
    for rec in valid:
        if not (isinstance(rec.get("title"), str) and rec["title"].strip()):
            rec["title"] = rec["source_uri"]
            audit["imputed_title"] += 1
        if not (isinstance(rec.get("language"), str) and rec["language"].strip()):
            rec["language"] = "unknown"
            audit["imputed_language"] += 1

    # 3. normalize text
    for rec in valid:
        rec["text"] = normalize_text(rec["text"])

    # 4. remove repeated captures (same URI + same text)
    seen: set[str] = set()
    unique = []
    for rec in valid:
        digest = content_hash(rec["source_uri"], rec["text"])
        if digest in seen:
            audit["removed_duplicates"] += 1
            continue
        seen.add(digest)
        rec["content_hash"] = digest
        unique.append(rec)

    # 5. exclude very short records
    kept = []
    for rec in unique:
        if len(rec["text"].split()) < config.MIN_WORDS:
            audit["excluded_short"] += 1
        else:
            kept.append(rec)

    # 6. counts
    for rec in kept:
        rec.update(compute_counts(rec["text"]))

    # 7. flag outliers (flag and keep)
    fences = iqr_fences([r["word_count"] for r in kept])
    for rec in kept:
        flagged = bool(
            fences and (rec["word_count"] < fences[0] or rec["word_count"] > fences[1])
        )
        rec["is_outlier"] = flagged
        audit["flagged_outliers"] += int(flagged)

    audit["final"] = len(kept)
    return kept, audit


def run() -> dict[str, int]:
    """Load raw records, clean them, and write the outputs. Returns the audit."""
    config.ensure_dirs()
    records = read_jsonl(config.WEB_RECORDS) + read_jsonl(config.FILE_RECORDS)
    if not records:
        raise PipelineError(
            "No raw records found (web_records.jsonl / file_records.jsonl are empty or "
            "missing). Run the collection stages first; nothing is fabricated."
        )
    kept, audit = clean_records(records)
    write_jsonl(config.CLEAN_RECORDS, kept)
    config.AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.AUDIT_JSON.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    logger.info("cleaning audit: %s", audit)
    return audit


if __name__ == "__main__":
    run()
