"""Runtime ingestion: fetch -> extract -> clean -> chunk -> embed -> graph, with real per-stage events."""
from __future__ import annotations

import io
import re
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable

from data_scripts import config
from data_scripts.chunk_data import MAX_WORDS, MIN_WORDS, OVERLAP_WORDS, chunk_record
from data_scripts.clean_data import clean_records
from data_scripts.collect_files import (ExtractionError, extract_docx, extract_pdf, extract_txt)
from data_scripts.utils import RobotsChecker, detect_language, fetch_with_retries, make_session, utc_now_iso
from retrieval import store
from retrieval.embed import embed
from retrieval.graph_build import triples as make_triples

STAGES = ("fetch", "extract", "clean", "chunk", "embed", "graph")


class IngestError(Exception):
    """Carries the real, user-facing reason a source failed."""


# --------------------------------------------------------------------------- event log
class EventLog:
    def __init__(self) -> None:
        self._events: dict[str, list[dict]] = {}
        self._cv = threading.Condition()

    def emit(self, sid: str, **ev: Any) -> None:
        with self._cv:
            self._events.setdefault(sid, []).append({"t": time.time(), **ev})
            self._cv.notify_all()

    def since(self, sid: str, idx: int, timeout: float = 15.0) -> list[dict]:
        with self._cv:
            if len(self._events.get(sid, [])) <= idx:
                self._cv.wait(timeout)
            return self._events.get(sid, [])[idx:]


events = EventLog()


# --------------------------------------------------------------------------- validation
def sanitize_filename(name: str) -> str:
    name = Path(name or "upload").name
    name = re.sub(r"[^\w.\- ]", "_", name).strip(" .") or "upload"
    return name[:120]


def sniff_kind(data: bytes) -> str:
    """Decide the file type from magic bytes, not the extension."""
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                if "word/document.xml" in z.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            pass
        raise IngestError("The file is a ZIP archive but not a DOCX document.")
    if b"\x00" in data[:4096] and data[:2] not in (b"\xff\xfe", b"\xfe\xff"):
        raise IngestError("Unsupported file type. Use PDF, DOCX or TXT.")
    return "txt"


def check_size(n: int) -> None:
    if n > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise IngestError(f"File is larger than the {config.MAX_UPLOAD_MB} MB limit.")


# --------------------------------------------------------------------------- pipeline
def _stage(sid: str, name: str, fn: Callable[[], Any]) -> Any:
    events.emit(sid, stage=name, state="start")
    t0 = time.monotonic()
    out = fn()
    return out, int((time.monotonic() - t0) * 1000)


def _extract_file(path: Path, kind: str) -> tuple[str, str]:
    try:
        title, text, _ = {"pdf": extract_pdf, "docx": extract_docx, "txt": extract_txt}[kind](path)
    except ExtractionError as exc:
        if str(exc) == "needs OCR":
            raise IngestError("This PDF has no text layer (scanned). OCR is not supported.") from exc
        raise IngestError(str(exc)) from exc
    if not text.strip():
        raise IngestError("No text could be extracted from the file.")
    return title, text


def _fetch_url(url: str) -> tuple[str, str]:
    from data_scripts.collect_web import _is_html, extract_text

    session = make_session()
    ok, why = RobotsChecker(session).allowed(url)
    if not ok:
        raise IngestError(f"Blocked by robots.txt: {why}.")
    resp, code, reason = fetch_with_retries(session, url)
    if resp is None or not resp.ok:
        raise IngestError(f"Could not fetch the page: {reason}.")
    if not _is_html(resp.headers.get("Content-Type", "")):
        raise IngestError(f"Not an HTML page ({resp.headers.get('Content-Type') or 'no content type'}).")
    if len(resp.content) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise IngestError(f"Page is larger than the {config.MAX_UPLOAD_MB} MB limit.")
    title, text = extract_text(resp.content)
    if not text.strip():
        raise IngestError("The page has no extractable text (it may need JavaScript to render).")
    return title, text


def run_ingest(sid: str, *, kind: str, uri: str, title_hint: str, loader: Callable[[], tuple[str, str]],
               origin: str = "upload") -> None:
    """Run all stages for one source; never raises (failures become events + a failed row)."""
    try:
        store.upsert_source(sid, title=title_hint, uri=uri, origin=origin, kind=kind, status="running", error=None)
        timings: dict[str, int] = {}

        events.emit(sid, stage="fetch" if kind == "url" else "extract", state="start")
        t0 = time.monotonic()
        title, text = loader()
        ms = int((time.monotonic() - t0) * 1000)
        words = len(text.split())
        events.emit(sid, stage="fetch" if kind == "url" else "extract", state="done", ms=ms, words=words,
                    chars=len(text))
        if kind == "url":
            events.emit(sid, stage="extract", state="done", ms=0, words=words)
        title = title or title_hint

        rec = {"source_id": sid, "title": title, "source_type": "website" if kind == "url" else kind,
               "source_uri": uri, "collected_at": utc_now_iso(), "language": detect_language(text), "text": text,
               "detect_headings": True}
        events.emit(sid, stage="clean", state="start")
        t0 = time.monotonic()
        cleaned, audit = clean_records([rec])
        if not cleaned:
            raise IngestError(f"Too short after cleaning ({words} words; at least {config.MIN_WORDS} needed).")
        cleaned[0]["detect_headings"] = True
        events.emit(sid, stage="clean", state="done", ms=int((time.monotonic() - t0) * 1000),
                    words=cleaned[0]["word_count"])

        events.emit(sid, stage="chunk", state="start")
        t0 = time.monotonic()
        chunks = chunk_record(cleaned[0], MAX_WORDS, OVERLAP_WORDS, MIN_WORDS)
        if not chunks:
            raise IngestError("No passages could be produced from the text.")
        events.emit(sid, stage="chunk", state="done", ms=int((time.monotonic() - t0) * 1000), chunks=len(chunks))

        events.emit(sid, stage="embed", state="start")
        t0 = time.monotonic()
        vecs: list[list[float]] = []
        for i in range(0, len(chunks), 32):
            vecs += embed([c["text"] for c in chunks[i:i + 32]])
        store.add_chunks(chunks, vecs, origin)
        events.emit(sid, stage="embed", state="done", ms=int((time.monotonic() - t0) * 1000), chunks=len(chunks))

        events.emit(sid, stage="graph", state="start")
        t0 = time.monotonic()
        trips = [t for c in chunks for t in make_triples(c)]
        n_edges = store.add_edges(sid, trips)
        n_ent = store.count_entities(sid)
        events.emit(sid, stage="graph", state="done", ms=int((time.monotonic() - t0) * 1000),
                    entities=n_ent, edges=n_edges)

        store.upsert_source(sid, title=title, status="ready", chunks=len(chunks), entities=n_ent, error=None)
        events.emit(sid, stage="complete", state="done", chunks=len(chunks), entities=n_ent)
    except IngestError as exc:
        _fail(sid, str(exc))
    except Exception as exc:  # noqa: BLE001 - report real cause
        _fail(sid, f"Unexpected error: {type(exc).__name__}: {str(exc)[:200]}")


def _fail(sid: str, reason: str) -> None:
    try:
        store.upsert_source(sid, status="failed", error=reason)
    except Exception:  # noqa: BLE001
        pass
    events.emit(sid, stage="error", state="failed", reason=reason)


def _spawn(*args: Any, **kw: Any) -> None:
    threading.Thread(target=run_ingest, args=args, kwargs=kw, daemon=True).start()


def start_file(filename: str, data: bytes) -> dict:
    sid = uuid.uuid4().hex[:12]
    safe = sanitize_filename(filename)
    store.upsert_source(sid, title=safe, uri=safe, origin="upload", status="queued")
    try:
        check_size(len(data))
        kind = sniff_kind(data)
    except IngestError as exc:
        _fail(sid, str(exc))
        return {"source_id": sid, "status": "failed", "error": str(exc)}
    folder = config.UPLOADS_DIR / sid
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (Path(safe).stem + {"pdf": ".pdf", "docx": ".docx", "txt": ".txt"}[kind])
    path.write_bytes(data)
    _spawn(sid, kind=kind, uri=safe, title_hint=Path(safe).stem, loader=lambda: _extract_file(path, kind))
    return {"source_id": sid, "status": "queued"}


def start_text(title: str, text: str) -> dict:
    sid = uuid.uuid4().hex[:12]
    title = (title or "Pasted text").strip()[:120]
    store.upsert_source(sid, title=title, uri="pasted text", origin="upload", status="queued")
    try:
        check_size(len(text.encode("utf-8")))
    except IngestError as exc:
        _fail(sid, str(exc))
        return {"source_id": sid, "status": "failed", "error": str(exc)}
    _spawn(sid, kind="txt", uri="pasted text", title_hint=title, loader=lambda: ("", text))
    return {"source_id": sid, "status": "queued"}


def start_url(url: str) -> dict:
    from data_scripts.utils import is_safe_url

    sid = uuid.uuid4().hex[:12]
    store.upsert_source(sid, title=url, uri=url, origin="upload", kind="url", status="queued")
    ok, why = is_safe_url(url)
    if not ok:
        _fail(sid, f"URL rejected: {why}.")
        return {"source_id": sid, "status": "failed", "error": f"URL rejected: {why}."}
    _spawn(sid, kind="url", uri=url, title_hint=url, loader=lambda: _fetch_url(url))
    return {"source_id": sid, "status": "queued"}
