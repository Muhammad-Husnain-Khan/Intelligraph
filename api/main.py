"""FastAPI service: ``uvicorn api.main:app``.

Legacy: GET /health, POST /query/{vector,graph,hybrid} (unchanged behaviour).
New:    /api/* (health, sources, chat stream, compare, graph).
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from data_scripts import config
from ingest import pipeline
from retrieval import chat, graph_retriever, hybrid_retriever, llm, store, vector_retriever

_status: dict = {"pg": "not started", "neo4j": "not started", "embedder": "not loaded"}


def _startup() -> None:
    try:
        store.init()
        _status["pg"] = "ok"
        _status["neo4j"] = "ok"
    except Exception as exc:  # noqa: BLE001 - reported by /api/health, not fatal
        msg = f"{type(exc).__name__}: {str(exc)[:120]}"
        if _status["pg"] != "ok":
            _status["pg"] = msg
        _status["neo4j"] = msg if "neo4j" in msg.lower() or _status["pg"] == "ok" else "not checked"
    try:
        store.preload_embedder()
        _status["embedder"] = "loaded"
    except Exception as exc:  # noqa: BLE001
        _status["embedder"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    llm.probe()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.get_running_loop().run_in_executor(None, _startup)
    yield
    store.close()


app = FastAPI(title="IntelliGraph", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in config.FRONTEND_ORIGIN.split(",") if o.strip()],
                   allow_methods=["*"], allow_headers=["*"])

_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path.startswith("/api/") and request.method != "OPTIONS" and not request.url.path.endswith("/events"):
        ip = request.client.host if request.client else "?"
        now = time.time()
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= config.RATE_LIMIT_PER_MIN:
            return JSONResponse({"detail": "Too many requests. Wait a minute and retry."}, status_code=429)
        q.append(now)
    return await call_next(request)


# --------------------------------------------------------------------------- legacy
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


# --------------------------------------------------------------------------- /api
@app.get("/api/health")
def api_health():
    out = {"status": "ok", "components": dict(_status), "llm": llm.last_probe().as_dict(),
           "counts": {"sources": 0, "chunks": 0}}
    try:
        out["counts"] = store.counts()
        out["components"]["pg"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["components"]["pg"] = f"{type(exc).__name__}"
        out["status"] = "degraded"
    try:
        with store.neo().session() as s:
            s.run("RETURN 1").consume()
        out["components"]["neo4j"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["components"]["neo4j"] = f"{type(exc).__name__}"
        out["status"] = "degraded"
    pr = llm.get_providers()
    out["llm"]["privacy_note"] = (
        "If you use the Gemini free tier, Google may use prompts to improve its products. "
        "Do not upload private documents.") if pr and pr[0].name == "gemini" else None
    fb = llm._state.get("fallback")
    out["llm"]["fallback_configured"] = fb is not None
    out["llm"]["fallback_model"] = getattr(fb, "model", None) if fb else None
    return out


@app.get("/api/sources")
def api_sources():
    return store.list_sources()


@app.post("/api/sources/files")
async def api_upload(files: list[UploadFile] = File(...)):
    out = []
    for f in files:
        data = await f.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
        out.append(await asyncio.get_running_loop().run_in_executor(None, pipeline.start_file, f.filename or "upload", data))
    return out


class UrlBody(BaseModel):
    url: str = Field(max_length=2048)


class TextBody(BaseModel):
    text: str
    title: str = ""


@app.post("/api/sources/url")
async def api_url(b: UrlBody):
    return await asyncio.get_running_loop().run_in_executor(None, pipeline.start_url, b.url.strip())


@app.post("/api/sources/text")
async def api_text(b: TextBody):
    return await asyncio.get_running_loop().run_in_executor(None, pipeline.start_text, b.title, b.text)


@app.delete("/api/sources/{source_id}")
def api_delete(source_id: str):
    if not store.delete_source(source_id):
        raise HTTPException(404, "No such source.")
    return {"deleted": source_id}


@app.get("/api/sources/{source_id}/events")
async def api_events(source_id: str, request: Request):
    async def gen():
        idx = 0
        loop = asyncio.get_running_loop()
        while not await request.is_disconnected():
            evs = await loop.run_in_executor(None, pipeline.events.since, source_id, idx, 10.0)
            for ev in evs:
                idx += 1
                yield f"data: {json.dumps(ev)}\n\n"
                if ev.get("stage") in ("complete", "error"):
                    return
            if not evs:
                yield ": ping\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class ChatReq(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["hybrid", "vector", "graph"] = "hybrid"
    k: int | None = Field(default=None, ge=1, le=20)
    source_ids: list[str] | None = None
    history: list[dict] | None = None


@app.post("/api/chat")
def api_chat(r: ChatReq):
    return chat.run_chat_blocking(r.model_dump())


@app.post("/api/chat/stream")
async def api_chat_stream(r: ChatReq, request: Request):
    cancel = threading.Event()
    q: queue.Queue = queue.Queue()
    END = object()

    def worker():
        try:
            for item in chat.run_chat(r.model_dump(), cancel):
                q.put(item)
        except Exception as exc:  # noqa: BLE001
            q.put(("error", {"code": "internal", "message": f"{type(exc).__name__}", "retryable": True}))
        finally:
            q.put(END)

    threading.Thread(target=worker, daemon=True).start()

    async def gen():
        loop = asyncio.get_running_loop()
        last_beat = time.monotonic()
        try:
            while True:
                try:
                    item = await loop.run_in_executor(None, lambda: q.get(timeout=1.0))
                except queue.Empty:
                    if await request.is_disconnected():
                        return
                    if time.monotonic() - last_beat >= 15:
                        last_beat = time.monotonic()
                        yield ": heartbeat\n\n"
                    continue
                if item is END:
                    return
                last_beat = time.monotonic()
                yield f"event: {item[0]}\ndata: {json.dumps(item[1])}\n\n"
        finally:
            cancel.set()  # client gone or finished: abort generation server-side

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/compare")
def api_compare(r: ChatReq):
    """Live run of all three retrieval modes; no generation."""
    k = r.k or config.TOP_K
    out = {}
    for mode in ("vector", "graph", "hybrid"):
        res = chat.retrieve(r.question, mode, k, r.source_ids)
        out[mode] = {"timings": res["timings"],
                     "passages": [{"chunk_id": p["chunk_id"], "title": p["title"], "heading": p.get("heading", ""),
                                   "score": p["score"], "via": p["via"]}
                                  for p in res["passages"]],
                     "triples": [{"s": a, "o": b, "chunk_id": c} for a, b, c in res["triples"][:30]]}
    return {"label": "live run", "question": r.question, "modes": out}


@app.get("/api/graph")
def api_graph(source_ids: str = "", limit: int = 150):
    ids = [s for s in source_ids.split(",") if s]
    limit = max(1, min(limit, 150))
    with store.neo().session() as s:
        rows = s.run("""MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
                        WHERE size($ids) = 0 OR r.source_id IN $ids
                        RETURN a.name AS a, b.name AS b, r.chunk_id AS c LIMIT 5000""", ids=ids)
        triples = [(x["a"], x["b"], x["c"]) for x in rows]
    g = chat.build_graph([], triples, set())
    top = {n["id"] for n in sorted(g["nodes"], key=lambda n: -n["degree"])[:limit]}
    return {"nodes": [n for n in g["nodes"] if n["id"] in top],
            "edges": [e for e in g["edges"] if e["source"] in top and e["target"] in top]}


@app.get("/api/starters")
def api_starters():
    """Starter questions from eval/golden_qa.json, returned only when the seeded library is present."""
    try:
        lib = [s for s in store.list_sources() if s.get("origin") == "library" and s.get("status") == "ready"]
    except Exception:  # noqa: BLE001
        return []
    if not lib:
        return []
    qa_path = config.ROOT / "eval" / "golden_qa.json"
    if not qa_path.exists():
        return []
    items = json.loads(qa_path.read_text(encoding="utf-8"))
    by_id = {int(q["id"]): q["question"] for q in items if "id" in q and "question" in q}
    # Showcase questions from report/main2.tex Case Studies (ids 5, 36, 1) plus domain coverage (18, 44)
    pick_ids = [5, 36, 1, 18, 44]
    return [by_id[i] for i in pick_ids if i in by_id]


@app.get("/api/eval/summary")
def api_eval_summary():
    """Recorded evaluation summary from eval/*.json (Section 6 P2-A & main2.tex)."""
    eval_dir = config.ROOT / "eval"
    metrics_p = eval_dir / "metrics_summary.json"
    ksweep_p = eval_dir / "k_sweep_summary.json"
    bias_p = eval_dir / "bias_report.json"
    return {
        "caption": "48 English questions. A chunk counts as relevant if it contains an answer string. Hybrid counts graph-added chunks beyond K.",
        "selected": {
            "architecture": "hybrid",
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_words": 300,
            "overlap_words": 50,
            "k": 5,
        },
        "metrics_summary": json.loads(metrics_p.read_text(encoding="utf-8")) if metrics_p.exists() else {},
        "k_sweep_summary": json.loads(ksweep_p.read_text(encoding="utf-8")) if ksweep_p.exists() else {},
        "bias_report": json.loads(bias_p.read_text(encoding="utf-8")) if bias_p.exists() else {},
    }
