"""Hybrid chunking: topic-wise (headings/paragraphs) first, size-capped windows second.

Reads ``data/processed/clean_records.jsonl`` and writes ``data/processed/chunks.jsonl``.

Strategy:
1. Split each document into paragraphs (blank-line separated).
2. Websites: group paragraphs under the nearest heading (a short paragraph without
   sentence punctuation that is followed by more text). Boilerplate sections such as
   References / External links are dropped. PDFs/DOCX/TXT have no reliable heading
   markup after extraction, so they are treated as one section of paragraphs.
3. A section within ``max_words`` becomes one chunk. A longer section is cut into
   sentence-aligned windows of about ``max_words`` with ``overlap`` words of overlap.
4. Chunks shorter than ``min_words`` are merged into a neighbour.

Run: ``python -m data_scripts.chunk_data``.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from typing import Any

from . import config
from .utils import PipelineError, get_logger, read_jsonl, write_jsonl

logger = get_logger(__name__)

MAX_WORDS = 300
OVERLAP_WORDS = 50
MIN_WORDS = 50
HEADING_MAX_WORDS = 10
SKIP_SECTIONS = {
    "references",
    "further reading",
    "external links",
    "see also",
    "notes",
    "bibliography",
    "citations",
}
CHUNKS_PATH = config.PROCESSED_DIR / "chunks.jsonl"
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\u06D4])\s+")
_END_PUNCT = (".", "!", "?", "\u06d4", ":", ";", ",")


def _wc(text: str) -> int:
    return len(text.split())


def split_paragraphs(text: str) -> list[str]:
    """Blank-line separated paragraphs with inner whitespace collapsed."""
    paras = (re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text))
    return [p for p in paras if p]


def is_heading(par: str) -> bool:
    """Heuristic: short, no sentence punctuation at the end."""
    return _wc(par) <= HEADING_MAX_WORDS and not par.endswith(_END_PUNCT)


def build_sections(paragraphs: list[str], detect_headings: bool) -> list[tuple[str, list[str]]]:
    """Return ``[(heading, [paragraphs])]``; boilerplate sections are dropped."""
    sections: list[tuple[str, list[str]]] = []
    heading, body = "", []
    for i, par in enumerate(paragraphs):
        if detect_headings and i + 1 < len(paragraphs) and is_heading(par):
            if body:
                sections.append((heading, body))
            heading, body = par, []
        else:
            body.append(par)
    if body:
        sections.append((heading, body))
    return [
        (h, b) for h, b in sections if h.strip().lower() not in SKIP_SECTIONS
    ]


def window_text(text: str, max_words: int, overlap: int, min_words: int) -> list[str]:
    """Sentence-aligned windows of about ``max_words`` with ``overlap`` words overlap."""
    sentences: list[str] = []
    for sent in _SENTENCE_SPLIT.split(text):
        words = sent.split()
        if len(words) <= max_words:
            sentences.append(sent)
            continue
        # a single huge "sentence" (e.g. PDF text without punctuation) is hard-split
        # into overlapping pieces
        step = max(max_words - overlap, 1)
        for j in range(0, len(words), step):
            sentences.append(" ".join(words[j : j + max_words]))
            if j + max_words >= len(words):
                break

    windows: list[list[str]] = []
    cur: list[str] = []
    cur_words = 0
    new_words = 0  # words added since the last window was closed
    for sent in sentences:
        n = _wc(sent)
        if cur and cur_words + n > max_words:
            windows.append(cur)
            keep: list[str] = []
            kept = 0
            for prev in reversed(cur):  # carry the overlap forward
                if kept >= overlap or kept + _wc(prev) > 2 * overlap:
                    break
                keep.insert(0, prev)
                kept += _wc(prev)
            cur, cur_words, new_words = keep, kept, 0
        cur.append(sent)
        cur_words += n
        new_words += n
    if cur:
        if windows and new_words < min_words:
            windows[-1].extend(cur[len(cur) - _count_new(cur, new_words) :])
        else:
            windows.append(cur)
    return [" ".join(w) for w in windows]


def _count_new(cur: list[str], new_words: int) -> int:
    """Number of trailing sentences of ``cur`` that make up ``new_words`` words."""
    count, total = 0, 0
    for sent in reversed(cur):
        if total >= new_words:
            break
        total += _wc(sent)
        count += 1
    return count


def chunk_record(
    rec: dict[str, Any], max_words: int, overlap: int, min_words: int
) -> list[dict[str, Any]]:
    """Chunk one cleaned record into chunk dicts (with provenance metadata)."""
    paragraphs = split_paragraphs(rec["text"])
    sections = build_sections(paragraphs, rec["source_type"] == "website")

    pieces: list[tuple[str, str]] = []  # (heading, text)
    for heading, body in sections:
        text = "\n\n".join(body)
        if _wc(text) <= max_words:
            pieces.append((heading, text))
        else:
            pieces.extend((heading, w) for w in window_text(text, max_words, overlap, min_words))

    merged: list[tuple[str, str]] = []
    for heading, text in pieces:  # merge small chunks into the previous one
        if merged and _wc(text) < min_words:
            prev_h, prev_t = merged[-1]
            merged[-1] = (prev_h, prev_t + "\n\n" + text)
        else:
            merged.append((heading, text))
    if len(merged) > 1 and _wc(merged[0][1]) < min_words:  # small first chunk -> next
        (h0, t0), (h1, t1) = merged[0], merged[1]
        merged[:2] = [(h0 or h1, t0 + "\n\n" + t1)]

    chunks = []
    current_word_offset = 0
    for i, (heading, text) in enumerate(merged):
        wc = _wc(text)
        offset_start = current_word_offset
        offset_end = current_word_offset + wc
        current_word_offset = offset_end
        chunks.append(
            {
                "chunk_id": f"{rec['source_id']}-{i:04d}",
                "source_id": rec["source_id"],
                "source_uri": rec["source_uri"],
                "title": rec["title"],
                "source_type": rec["source_type"],
                "language": rec["language"],
                "section_heading": heading,
                "chunk_index": i,
                "word_offset_start": offset_start,
                "word_offset_end": offset_end,
                "word_count": wc,
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
            }
        )
    return chunks


def run(max_words: int = MAX_WORDS, overlap: int = OVERLAP_WORDS, min_words: int = MIN_WORDS) -> int:
    """Chunk all cleaned records. Returns the number of chunks written."""
    records = read_jsonl(config.CLEAN_RECORDS)
    if not records:
        raise PipelineError(
            f"No cleaned records at {config.CLEAN_RECORDS}. Run the Phase 2 pipeline first."
        )
    chunks = [c for rec in records for c in chunk_record(rec, max_words, overlap, min_words)]
    write_jsonl(CHUNKS_PATH, chunks)
    per_type = Counter(c["source_type"] for c in chunks)
    sizes = sorted(c["word_count"] for c in chunks)
    logger.info(
        "%d chunks from %d documents (%s); words/chunk min=%d median=%d max=%d -> %s",
        len(chunks),
        len(records),
        ", ".join(f"{k}: {v}" for k, v in sorted(per_type.items())),
        sizes[0] if sizes else 0,
        sizes[len(sizes) // 2] if sizes else 0,
        sizes[-1] if sizes else 0,
        CHUNKS_PATH,
    )
    return len(chunks)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Hybrid chunking of cleaned records")
    p.add_argument("--max-words", type=int, default=MAX_WORDS)
    p.add_argument("--overlap", type=int, default=OVERLAP_WORDS)
    p.add_argument("--min-words", type=int, default=MIN_WORDS)
    a = p.parse_args()
    try:
        run(a.max_words, a.overlap, a.min_words)
    except PipelineError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
