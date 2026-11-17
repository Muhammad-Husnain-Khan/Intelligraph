"""Extract text from PDF/DOCX/TXT files in ``data/raw/uploads/`` into ``file_records.jsonl``."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from . import config
from .utils import (
    PipelineError,
    detect_language,
    get_logger,
    log_event,
    merge_jsonl,
    source_id,
    utc_now_iso,
)

logger = get_logger(__name__)

SUPPORTED = {".pdf": "pdf", ".docx": "docx", ".txt": "txt"}
_ENCODINGS = ("utf-8-sig", "utf-16", "cp1256", "cp1252", "latin-1")


class ExtractionError(Exception):
    """A file could not be turned into text; message is the logged reason."""


def _format_pdf_page(page_text: str, page_num: int) -> str:
    """Preserve short heading lines inside a PDF page; add Page N fallback if no heading starts the page."""
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    if not lines:
        return ""
    out: list[str] = []
    has_heading = False
    for idx, ln in enumerate(lines):
        wc = len(ln.split())
        is_h = (
            0 < wc <= 10
            and not ln.endswith((".", "!", "?", ":", ";", ","))
            and (
                re.match(r"^(\d+(\.\d+)*\.?\s+[A-Z]|#+\s+\S)", ln)
                or ln.isupper()
                or (idx + 1 < len(lines) and len(lines[idx + 1].split()) >= 12 and ln[0].isupper())
            )
        )
        if is_h:
            has_heading = True
            out.append(f"\n\n{ln}\n\n")
        else:
            out.append(ln)
    joined = "\n".join(out)
    if not has_heading:
        return f"Page {page_num}\n\n{joined}"
    return joined


def extract_pdf(path: Path) -> tuple[str, str, dict[str, Any]]:
    """Return ``(title, text, metadata)`` for a PDF; raises for scanned PDFs."""
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        raw_pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:  # pypdf raises many exception types
        raise ExtractionError(f"unreadable PDF: {type(exc).__name__}") from exc
    raw_text = "\n\n".join(p.strip() for p in raw_pages if p.strip())
    n_pages = max(len(raw_pages), 1)
    if len(raw_text) / n_pages < config.SCANNED_MIN_CHARS_PER_PAGE:
        raise ExtractionError("needs OCR")
    text = "\n\n".join(_format_pdf_page(p, i + 1) for i, p in enumerate(raw_pages) if p.strip())
    title = ""
    try:
        title = (reader.metadata.title or "").strip() if reader.metadata else ""
    except Exception:  # noqa: BLE001
        title = ""
    return title, text, {"page_count": len(raw_pages)}


def extract_docx(path: Path) -> tuple[str, str, dict[str, Any]]:
    """Return ``(title, text, metadata)`` for a DOCX file."""
    import docx

    try:
        document = docx.Document(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"unreadable DOCX: {type(exc).__name__}") from exc
    paras: list[str] = []
    for p in document.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        style_name = (getattr(p.style, "name", "") or "").lower()
        if "heading" in style_name or "title" in style_name:
            t = re.sub(r"[*_#`]+", "", t).strip().rstrip(".:;")
        paras.append(t)
    title = (document.core_properties.title or "").strip()
    return title, "\n\n".join(paras), {"paragraph_count": len(paras)}


def extract_txt(path: Path) -> tuple[str, str, dict[str, Any]]:
    """Return ``("", text, metadata)`` for a TXT file, trying several encodings."""
    raw = path.read_bytes()
    for enc in _ENCODINGS:
        try:
            decoded = raw.decode(enc)
            # Ensure single-line markdown (#) or numbered headings are separated by blank lines
            lines = decoded.splitlines()
            out: list[str] = []
            for ln in lines:
                s = ln.strip()
                if s and (s.startswith("#") or (len(s.split()) <= 10 and re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", s) and not s.endswith("."))):
                    out.append(f"\n{s}\n")
                else:
                    out.append(ln)
            return "", "\n".join(out), {"encoding": enc}
        except UnicodeDecodeError:
            continue
    raise ExtractionError("undecodable text encoding")  # pragma: no cover


_EXTRACTORS = {"pdf": extract_pdf, "docx": extract_docx, "txt": extract_txt}


def _paper_titles() -> dict[str, str]:
    """Map ``<arxiv_id>.pdf`` file names to titles from papers.csv, if present."""
    if not config.PAPERS_CSV.exists():
        return {}
    with config.PAPERS_CSV.open("r", encoding="utf-8", newline="") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    return {
        f"{r['arxiv_id'].strip().replace('/', '_')}.pdf": (r.get("title") or "").strip()
        for r in csv.DictReader(lines)
        if (r.get("arxiv_id") or "").strip()
    }


def run() -> int:
    """Process every supported file in uploads/. Returns the number of records."""
    config.ensure_dirs()
    files = sorted(
        p for p in config.UPLOADS_DIR.iterdir() if p.suffix.lower() in SUPPORTED
    )
    if not files:
        raise PipelineError(
            f"No PDF/DOCX/TXT files found in {config.UPLOADS_DIR}. "
            "Run download_papers and/or place your own files there."
        )
    titles = _paper_titles()
    records: list[dict[str, Any]] = []
    for path in files:
        kind = SUPPORTED[path.suffix.lower()]
        try:
            title, text, meta = _EXTRACTORS[kind](path)
            if not text.strip():
                raise ExtractionError("empty text")
        except ExtractionError as exc:
            log_event("file", path.name, False, None, str(exc))
            logger.warning("fail %s (%s)", path.name, exc)
            continue
        log_event("file", path.name, True, None, "ok")
        records.append(
            {
                "source_id": source_id(path.name),
                "title": titles.get(path.name) or title or path.stem,
                "source_type": kind,
                "source_uri": path.name,
                "collected_at": utc_now_iso(),
                "language": detect_language(text),
                "text": text,
                "metadata": {**meta, "language_source": "script heuristic (auto)"},
            }
        )
    added, replaced, total = merge_jsonl(config.FILE_RECORDS, records)
    logger.info(
        "files: %d/%d extracted (%d new, %d replaced); %d total records in %s",
        len(records), len(files), added, replaced, total, config.FILE_RECORDS,
    )
    return len(records)


if __name__ == "__main__":
    run()
