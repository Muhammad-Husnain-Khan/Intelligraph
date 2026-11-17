import type { CompareResult, EvalSummary, Health, IngestEvent, Mode, Source } from "./types";

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let d = `HTTP ${r.status}`;
    try { d = (await r.json()).detail ?? d; } catch { /* keep status text */ }
    throw new Error(typeof d === "string" ? d : JSON.stringify(d));
  }
  return r.json() as Promise<T>;
}

export const getHealth = () => fetch("/api/health").then((r) => json<Health>(r));
export const getSources = () => fetch("/api/sources").then((r) => json<Source[]>(r));
export const getStarters = () => fetch("/api/starters").then((r) => json<string[]>(r));
export const getEvalSummary = () => fetch("/api/eval/summary").then((r) => json<EvalSummary>(r));
export const runCompare = (body: { question: string; k?: number; source_ids?: string[] }) =>
  fetch("/api/compare", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...body, mode: "hybrid" }) })
    .then((r) => json<CompareResult>(r));
export const deleteSource = (id: string) => fetch(`/api/sources/${id}`, { method: "DELETE" }).then((r) => json(r));

export const uploadFiles = (files: File[]) => {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  return fetch("/api/sources/files", { method: "POST", body: fd }).then((r) => json<{ source_id: string; status: string; error?: string }[]>(r));
};
export const addUrl = (url: string) =>
  fetch("/api/sources/url", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }) })
    .then((r) => json<{ source_id: string; status: string; error?: string }>(r));
export const addText = (text: string, title: string) =>
  fetch("/api/sources/text", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, title }) })
    .then((r) => json<{ source_id: string; status: string; error?: string }>(r));

/** Real ingestion stages over SSE. Returns a closer. */
export function watchSource(id: string, onEvent: (e: IngestEvent) => void): () => void {
  const es = new EventSource(`/api/sources/${id}/events`);
  es.onmessage = (m) => {
    const e = JSON.parse(m.data) as IngestEvent;
    onEvent(e);
    if (e.stage === "complete" || e.stage === "error") es.close();
  };
  es.onerror = () => es.close();
  return () => es.close();
}

export interface ChatBody {
  question: string; mode: "hybrid" | "vector" | "graph"; k?: number; source_ids?: string[]; history?: { role: string; content: string }[];
}

/** POST + SSE over fetch (EventSource cannot POST). Resolves when the stream ends. */
export async function chatStream(
  body: ChatBody, on: (event: string, data: any) => void, signal: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/chat/stream", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
  });
  if (!res.ok || !res.body) {
    let msg = `HTTP ${res.status}`;
    try { const d = await res.json(); msg = typeof d.detail === "string" ? d.detail : msg; } catch { /* */ }
    on("error", { code: res.status === 429 ? "rate_limited" : "http_error", message: msg, retryable: true });
    return;
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i: number;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, i);
      buf = buf.slice(i + 2);
      let ev = "message", data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) ev = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) on(ev, JSON.parse(data));
    }
  }
}
