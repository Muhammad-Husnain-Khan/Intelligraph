"""Tests for the Track A backend with the LLM and stores mocked (no live services needed)."""
from __future__ import annotations

import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from ingest import pipeline
from retrieval import chat, llm


class FakeProvider:
    name, model = "fake", "fake-1"

    def __init__(self, text="Microservices split apps [1]. Bogus [9]."):
        self.text = text

    def generate(self, prompt, system=""):
        return "standalone query"

    def stream(self, prompt, system=""):
        for w in self.text.split(" "):
            yield w + " "


PASSAGES = [
    {"chunk_id": "a-0000", "source_id": "a", "uri": "u", "title": "T", "heading": "H", "text": "Microservices text.",
     "score": 0.8},
    {"chunk_id": "b-0000", "source_id": "b", "uri": "u2", "title": "T2", "heading": "", "text": "Other.", "score": 0.5},
]


@pytest.fixture
def mocked(monkeypatch):
    monkeypatch.setattr(chat.store, "embed_query", lambda q: [0.0] * 384)
    monkeypatch.setattr(chat, "vector_search", lambda qv, k, sids: [dict(p) for p in PASSAGES])
    monkeypatch.setattr(chat, "graph_walk", lambda q, sids, lim=80: {
        "seeds": ["Microservices"], "triples": [("Microservices", "Netflix", "a-0000"), ("Netflix", "AWS", "c-0000")]})
    monkeypatch.setattr(chat, "fetch_chunks", lambda ids: [
        {"chunk_id": "c-0000", "source_id": "c", "uri": "u3", "title": "T3", "heading": "", "text": "AWS.", "score": None}])
    monkeypatch.setitem(llm._state, "primary", FakeProvider())
    monkeypatch.setitem(llm._state, "fallback", None)


def test_valid_citations_drops_invalid():
    text, cites = chat.valid_citations("A [1] B [7] C [2][1]", 2)
    assert "[7]" not in text and cites == [1, 2]


def test_chat_contract_and_bridge(mocked):
    out = chat.run_chat_blocking({"question": "What are microservices?", "mode": "hybrid", "k": 2})
    ev = out["evidence"]
    vias = {p["chunk_id"]: p["via"] for p in ev["passages"]}
    assert vias == {"a-0000": "both", "b-0000": "vector", "c-0000": "graph"}
    assert [p["n"] for p in ev["passages"]] == [1, 2, 3]
    assert ev["graph"]["nodes"] and ev["graph"]["edges"]
    done = out["done"]
    assert done["citations"] == [1] and done["grounded"] is True
    assert "[9]" not in done["answer"]
    assert done["model"] == "fake-1" and done["fallback"] is False


def test_no_evidence_sets_grounded_false(mocked, monkeypatch):
    monkeypatch.setitem(llm._state, "primary", FakeProvider("NO_EVIDENCE"))
    out = chat.run_chat_blocking({"question": "q", "mode": "vector"})
    assert out["done"]["grounded"] is False and out["done"]["answer"] == ""


def test_uncited_answer_is_ungrounded(mocked, monkeypatch):
    monkeypatch.setitem(llm._state, "primary", FakeProvider("A confident answer with no citation."))
    assert chat.run_chat_blocking({"question": "q", "mode": "vector"})["done"]["grounded"] is False


def test_followup_runs_condense(mocked):
    out = chat.run_chat_blocking({"question": "and its founder?", "history": [{"role": "user", "content": "x"}]})
    assert any(s["name"] == "condense" and s.get("detail") == "standalone query" for s in out["stages"])


def test_llm_down_yields_retryable_error_with_evidence(mocked, monkeypatch):
    class Down(FakeProvider):
        def stream(self, prompt, system=""):
            raise llm.LLMError("llm_rate_limited", "quota", True)
            yield  # pragma: no cover

    monkeypatch.setitem(llm._state, "primary", Down())
    out = chat.run_chat_blocking({"question": "q", "mode": "vector"})
    assert out["evidence"]["passages"] and out["error"]["code"] == "llm_rate_limited" and out["error"]["retryable"]


def test_fallback_is_reported(mocked, monkeypatch):
    class Down(FakeProvider):
        def stream(self, prompt, system=""):
            raise llm.LLMError("llm_rate_limited", "quota", True)
            yield  # pragma: no cover

    monkeypatch.setitem(llm._state, "primary", Down())
    monkeypatch.setitem(llm._state, "fallback", FakeProvider("Fine [1]"))
    done = chat.run_chat_blocking({"question": "q", "mode": "vector"})["done"]
    assert done["fallback"] is True


# --------------------------------------------------------------------------- ingestion validation
def _docx_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", "<w/>")
    return buf.getvalue()


def test_sniff_kind_by_magic_bytes():
    assert pipeline.sniff_kind(b"%PDF-1.7 ...") == "pdf"
    assert pipeline.sniff_kind(_docx_bytes()) == "docx"
    assert pipeline.sniff_kind("plain text".encode()) == "txt"
    with pytest.raises(pipeline.IngestError):
        pipeline.sniff_kind(b"\x00\x01\x02binary")
    with pytest.raises(pipeline.IngestError):
        pipeline.sniff_kind(b"PK\x03\x04junk")


def test_sanitize_filename():
    assert pipeline.sanitize_filename("../../etc/pass wd?.pdf") == "pass wd_.pdf"


# --------------------------------------------------------------------------- API
@pytest.fixture
def client(mocked, monkeypatch):
    from api import main

    monkeypatch.setattr(main.store, "counts", lambda: {"sources": 1, "chunks": 2})
    monkeypatch.setattr(main.store, "list_sources", lambda: [{"source_id": "a", "title": "T"}])
    monkeypatch.setattr(main.store, "delete_source", lambda sid: sid == "a")
    return TestClient(main.app)


def test_api_sources_and_delete(client):
    assert client.get("/api/sources").json()[0]["source_id"] == "a"
    assert client.delete("/api/sources/a").status_code == 200
    assert client.delete("/api/sources/zzz").status_code == 404


def test_api_chat_stream_sse(client):
    r = client.post("/api/chat/stream", json={"question": "What are microservices?", "mode": "vector"})
    assert r.status_code == 200
    events = [ln[7:] for ln in r.text.splitlines() if ln.startswith("event: ")]
    assert events[0] == "stage" and "evidence" in events and "token" in events and events[-1] == "done"
    done = [json.loads(ln[6:]) for ln in r.text.splitlines() if ln.startswith("data: ")][-1]
    assert done["grounded"] is True


def test_api_chat_rejects_bad_mode(client):
    assert client.post("/api/chat", json={"question": "q", "mode": "nope"}).status_code == 422


def test_legacy_health_unchanged(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_api_eval_summary_and_starters(client, monkeypatch):
    from api import main

    ev = client.get("/api/eval/summary").json()
    assert "48 English questions" in ev["caption"]
    assert ev["selected"]["architecture"] == "hybrid" and ev["selected"]["k"] == 5
    assert "5" in ev["k_sweep_summary"] and "hybrid" in ev["k_sweep_summary"]["5"]

    monkeypatch.setattr(main.store, "list_sources", lambda: [{"source_id": "a", "origin": "library", "status": "ready"}])
    st = client.get("/api/starters").json()
    assert len(st) >= 3 and "What is a Big Ball of Mud?" in st

    monkeypatch.setattr(main.store, "list_sources", lambda: [{"source_id": "u", "origin": "upload", "status": "ready"}])
    assert client.get("/api/starters").json() == []


def test_api_compare_live_run(client):
    res = client.post("/api/compare", json={"question": "What is a Big Ball of Mud?", "k": 5}).json()
    assert res["label"] == "live run"
    assert set(res["modes"].keys()) == {"vector", "graph", "hybrid"}


def test_answer_cache_opt_in(mocked, monkeypatch):
    monkeypatch.setattr(chat.config, "ANSWER_CACHE", True)
    chat._ANSWER_CACHE.clear()
    first = chat.run_chat_blocking({"question": "Cache test?", "mode": "vector", "k": 2})
    second = chat.run_chat_blocking({"question": "Cache test?", "mode": "vector", "k": 2})
    assert first["done"]["cached"] is False
    assert second["done"]["cached"] is True


def test_heading_cleaning_and_txt_section_detection():
    from pathlib import Path
    from data_scripts.chunk_data import chunk_record, clean_heading

    assert clean_heading("## **1.2 _System_ `Design`**") == "1.2 System Design"
    body1 = " ".join(["alpha"] * 65) + "."
    body2 = " ".join(["bravo"] * 65) + "."
    body3 = " ".join(["charlie"] * 65) + "."
    raw_txt = f"# **1. Overview**\n\n{body1}\n\n## *2. Consensus Protocol*\n\n{body2}\n\n3. Fault Tolerance\n\n{body3}"
    rec = {
        "source_id": "t1", "source_uri": "notes.txt", "title": "notes",
        "source_type": "txt", "language": "en", "text": raw_txt,
    }
    chunks = chunk_record(rec, 300, 50, 50)
    headings = [c["section_heading"] for c in chunks]
    assert headings == ["1. Overview", "2. Consensus Protocol", "3. Fault Tolerance"]
    assert all(not any(ch in h for ch in "*_#`") for h in headings)


def test_thinking_levels_shared_config():
    from pathlib import Path
    import re

    cfg_text = Path("web/src/config.ts").read_text(encoding="utf-8")
    assert re.search(r'quick:\s*\{\s*label:\s*"Quick",\s*k:\s*3\s*\}', cfg_text)
    assert re.search(r'standard:\s*\{\s*label:\s*"Standard",\s*k:\s*5\s*\}', cfg_text)
    assert re.search(r'deep:\s*\{\s*label:\s*"Deep",\s*k:\s*10\s*\}', cfg_text)
