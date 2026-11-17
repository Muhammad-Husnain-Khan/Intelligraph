import { useCallback, useEffect, useRef, useState } from "react";
import { addText, addUrl, deleteSource, uploadFiles, watchSource } from "./api";
import { humaniseTitle } from "./config";
import type { Health, IngestEvent, Source } from "./types";

interface Props {
  sources: Source[];
  scope: Set<string>;
  setScope: (s: Set<string>) => void;
  refresh: () => void;
  health: Health | null;
  devMode: boolean;
}

const STAGE_LABEL: Record<string, string> = {
  fetch: "fetched", extract: "extracted text", clean: "cleaned", chunk: "chunked", embed: "embedded", graph: "graph built",
};
function describe(e: IngestEvent): string | null {
  if (e.state !== "done" || !STAGE_LABEL[e.stage]) return null;
  const n = e.stage === "chunk" || e.stage === "embed" ? `${e.chunks} passages`
    : e.stage === "graph" ? `${e.entities} entities`
    : e.words != null ? `${e.words.toLocaleString("en-US")} words` : "";
  return `${STAGE_LABEL[e.stage]} · ${n}${n && e.ms != null ? " · " : ""}${e.ms != null ? `${e.ms} ms` : ""}`;
}

export default function Library({ sources, scope, setScope, refresh, health, devMode }: Props) {
  const [log, setLog] = useState<Record<string, string[]>>({});
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const [confirmRemove, setConfirmRemove] = useState<string | null>(null);
  const [url, setUrl] = useState("");
  const [paste, setPaste] = useState(false);
  const [text, setText] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const watched = useRef(new Set<string>());

  const watch = useCallback((id: string) => {
    if (watched.current.has(id)) return;
    watched.current.add(id);
    setBusy((b) => ({ ...b, [id]: true }));
    watchSource(id, (e) => {
      const line = describe(e);
      if (line) setLog((l) => ({ ...l, [id]: [...(l[id] ?? []), line] }));
      if (e.stage === "complete" || e.stage === "error") {
        setBusy((b) => ({ ...b, [id]: false }));
        refresh();
      }
    });
  }, [refresh]);

  useEffect(() => {
    sources.filter((s) => s.status === "queued" || s.status === "running").forEach((s) => watch(s.source_id));
  }, [sources, watch]);

  const handle = async (p: Promise<{ source_id: string; status: string; error?: string }[] | { source_id: string; status: string; error?: string }>) => {
    setMsg(null);
    try {
      const res = await p;
      (Array.isArray(res) ? res : [res]).forEach((r) => { if (r.status !== "failed") watch(r.source_id); });
    } catch (e) { setMsg((e as Error).message); }
    refresh();
  };

  const toggle = (id: string) => {
    const n = new Set(scope.size === 0 ? sources.map((s) => s.source_id) : scope);
    n.has(id) ? n.delete(id) : n.add(id);
    if (n.size === 0) return;
    setScope(n.size === sources.length ? new Set() : n);
  };
  const remove = async (id: string) => {
    await deleteSource(id);
    const n = new Set(scope);
    n.delete(id);
    setScope(n);
    setConfirmRemove(null);
    refresh();
  };
  const retryFailed = async (s: Source) => {
    await deleteSource(s.source_id);
    if (s.kind === "url" && s.uri.startsWith("http")) {
      await handle(addUrl(s.uri));
    } else {
      refresh();
      fileRef.current?.click();
    }
  };
  const note = health?.llm.privacy_note;

  return (
    <section aria-label="Library" className="library">
      <h2 className="label">Library</h2>
      {note && <p className="note mono">{note}</p>}
      <div
        className={over ? "drop over" : "drop"}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); const f = [...e.dataTransfer.files]; if (f.length) handle(uploadFiles(f)); }}
      >
        <button className="drop-btn" onClick={() => fileRef.current?.click()}>
          Drop a PDF, DOCX or TXT here, or paste a link
        </button>
        <input ref={fileRef} type="file" hidden multiple accept=".pdf,.docx,.txt"
          onChange={(e) => { const f = [...(e.target.files ?? [])]; if (f.length) handle(uploadFiles(f)); e.target.value = ""; }} />
        <form className="row" onSubmit={(e) => { e.preventDefault(); if (url.trim()) { handle(addUrl(url.trim())); setUrl(""); } }}>
          <input aria-label="Web page address" placeholder="https://" value={url} onChange={(e) => setUrl(e.target.value)} />
          <button type="submit" disabled={!url.trim()}>Add link</button>
        </form>
        <button className="link" onClick={() => setPaste(!paste)}>{paste ? "Hide pasted text" : "Paste text instead"}</button>
        {paste && (
          <form onSubmit={(e) => { e.preventDefault(); if (text.trim()) { handle(addText(text, "")); setText(""); setPaste(false); } }}>
            <textarea aria-label="Text to add" rows={5} value={text} onChange={(e) => setText(e.target.value)} />
            <button type="submit" disabled={!text.trim()}>Add text</button>
          </form>
        )}
      </div>
      {msg && <p className="err" role="alert">{msg}</p>}

      <ul className="sources">
        {sources.map((s) => {
          const displayTitle = humaniseTitle(s.title);
          const isFailed = s.status === "failed";
          const isIndexing = s.status === "queued" || s.status === "running" || busy[s.source_id];
          return (
            <li key={s.source_id} className={isFailed ? "src failed" : "src"}>
              <div className="src-main">
                <label className="srow">
                  <input type="checkbox" checked={scope.size === 0 ? true : scope.has(s.source_id)}
                    onChange={() => toggle(s.source_id)} aria-label={`Search ${displayTitle}`} />
                  <span className="stitle">{displayTitle}</span>
                </label>
                <span className="mono dim src-count">
                  {s.status === "ready" ? `${s.chunks} ${s.chunks === 1 ? "passage" : "passages"}`
                    : isFailed ? "failed" : "indexing…"}
                </span>
              </div>

              {isFailed && (
                <div className="src-fail-bar">
                  <p className="err">{s.error || "Ingestion failed."}</p>
                  <div className="row-actions mono">
                    <button className="link small" onClick={() => retryFailed(s)}>Retry</button>
                    <button className="link small" onClick={() => remove(s.source_id)}>Remove</button>
                  </div>
                </div>
              )}

              <details
                key={`${s.source_id}-${devMode ? "dev" : "user"}`}
                className="src-details mono"
                open={devMode ? true : undefined}
              >
                <summary className="src-summary">Source details</summary>
                <div className="src-details-body dim">
                  <div>Origin: {s.origin} · {s.entities} {s.entities === 1 ? "entity" : "entities"}</div>
                  <div>File / URI: {s.uri}</div>
                  {devMode && <div>Source ID: {s.source_id}</div>}
                  {(log[s.source_id] ?? []).map((l, i) => <div key={i} className="stage">{l}</div>)}
                  {isIndexing && <div className="stage">working…</div>}
                  {!isFailed && (
                    confirmRemove === s.source_id ? (
                      <div className="confirm-remove">
                        <span>Remove this source?</span>{" "}
                        <button className="link small" onClick={() => remove(s.source_id)}>Confirm remove</button>{" "}
                        <button className="link small" onClick={() => setConfirmRemove(null)}>Cancel</button>
                      </div>
                    ) : (
                      <button className="link small" onClick={() => setConfirmRemove(s.source_id)}>Remove</button>
                    )
                  )}
                </div>
              </details>
            </li>
          );
        })}
        {!sources.length && <li className="empty">No sources yet.</li>}
      </ul>
    </section>
  );
}
