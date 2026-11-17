import { Fragment, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { chatStream, getEvalSummary, getHealth, getSources, getStarters, runCompare } from "./api";
import { cleanHeading, DEFAULT_THINKING_LEVEL, humaniseTitle, THINKING_LEVELS, type ThinkingLevel } from "./config";
import Library from "./Library";
import type { EvalSummary, Exchange, Graph, Health, Mode, Passage, Selection, SseEventEntry, Source } from "./types";

const HISTORY_KEY = "intelligraph.history.v1";
const DEV_MODE_KEY = "intelligraph.devMode.v1";
const EMPTY_GRAPH: Graph = { nodes: [], edges: [] };
const uid = () => Math.random().toString(36).slice(2, 10);
const num = (n: number) => n.toLocaleString("en-US");

function loadHistory(): Exchange[] {
  try { return (JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]") as Exchange[]).map((e) => ({ ...e, pending: false })); }
  catch { return []; }
}

function loadDevMode(): boolean {
  try { return localStorage.getItem(DEV_MODE_KEY) === "1"; }
  catch { return false; }
}

function tickerLine(s: { name: string; ms?: number; detail?: string }): string | null {
  if (s.ms == null) return null;
  const t = `${s.ms} ms`;
  switch (s.name) {
    case "condense": return `rewrote follow-up · “${s.detail}” · ${t}`;
    case "embed": return `embedded query · ${t}`;
    case "vector": return `vector search · ${s.detail} · ${t}`;
    case "graph": return `graph walk · ${s.detail} · ${t}`;
    default: return `${s.name} · ${t}`;
  }
}

/** Plain working line in User mode reflecting the real current stage (Section 6). */
function userWorkingLine(ex: Exchange): string | null {
  if (!ex.pending || ex.text) return null;
  const st = ex.currentStage;
  if (st === "generate" || ex.passages.length > 0) return "Writing the answer";
  if (st === "graph") return "Following connections";
  return "Searching your sources";
}

export default function App() {
  const [exchanges, setExchanges] = useState<Exchange[]>(loadHistory);
  const [devMode, setDevMode] = useState<boolean>(loadDevMode);
  const [thinking, setThinking] = useState<ThinkingLevel>(DEFAULT_THINKING_LEVEL);
  const [mode, setMode] = useState<Mode>("hybrid");
  const [kVal, setKVal] = useState<number>(THINKING_LEVELS[DEFAULT_THINKING_LEVEL].k);
  const [sources, setSources] = useState<Source[]>([]);
  const [starters, setStarters] = useState<string[]>([]);
  const [evalSummary, setEvalSummary] = useState<EvalSummary | null>(null);
  const [evalOpen, setEvalOpen] = useState(false);
  const [scope, setScope] = useState<Set<string>>(new Set());
  const [health, setHealth] = useState<Health | null>(null);
  const [healthErr, setHealthErr] = useState(false);
  const [sel, setSel] = useState<Selection>(null);
  const [active, setActive] = useState<string | null>(null);
  const [view, setView] = useState<"ask" | "evidence" | "library">("ask");
  const [drawer, setDrawer] = useState(false);
  const [statusOpen, setStatusOpen] = useState(false);
  const [input, setInput] = useState("");
  const [hoverCite, setHoverCite] = useState<{ ex: string; n: number } | null>(null);
  const [expandedPassages, setExpandedPassages] = useState<Set<string>>(new Set());
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const askRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  const toggleDevMode = useCallback(() => {
    setDevMode((d) => {
      const next = !d;
      localStorage.setItem(DEV_MODE_KEY, next ? "1" : "0");
      if (!next) {
        setEvalOpen(false);
        setStatusOpen(false);
      }
      return next;
    });
  }, []);

  const selectThinking = (lvl: ThinkingLevel) => {
    setThinking(lvl);
    setKVal(THINKING_LEVELS[lvl].k);
  };

  const refresh = useCallback(() => {
    getSources().then(setSources).catch(() => {});
    getStarters().then(setStarters).catch(() => {});
    getHealth().then((h) => { setHealth(h); setHealthErr(false); }).catch(() => setHealthErr(true));
  }, []);
  useEffect(() => { refresh(); getEvalSummary().then(setEvalSummary).catch(() => {}); }, [refresh]);
  useEffect(() => {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(exchanges.filter((e) => !e.pending).slice(-40)));
    setTimeout(() => endRef.current?.scrollIntoView({ block: "end" }), 30);
  }, [exchanges]);

  const current = exchanges.find((e) => e.id === active) ?? exchanges[exchanges.length - 1] ?? null;
  const hiPassages = useMemo(() => {
    const s = new Set<string>();
    if (sel?.kind === "passage") s.add(sel.id);
    return s;
  }, [sel]);

  const patch = (id: string, f: (e: Exchange) => Exchange) => setExchanges((xs) => xs.map((x) => (x.id === id ? f(x) : x)));

  const ask = useCallback(async (question: string, retryId?: string, overrideMode?: Mode) => {
    if (!question.trim() || exchanges.some((e) => e.pending)) return;
    const effectiveMode: Mode = devMode ? (overrideMode ?? mode) : "hybrid";
    const effectiveK: number = devMode ? kVal : THINKING_LEVELS[thinking].k;
    const id = retryId ?? uid();
    const ex: Exchange = {
      id, question, mode: effectiveMode, k: effectiveK, thinkingLevel: thinking,
      stages: [], currentStage: "embed", passages: [], graph: EMPTY_GRAPH, text: "", pending: true, sseLog: [],
    };
    const done = exchanges.filter((e) => !e.pending && e.done && e.id !== retryId).slice(-3);
    const history = done.flatMap((e) => [{ role: "user", content: e.question }, { role: "assistant", content: e.done?.answer ?? "" }]);
    setExchanges((xs) => (retryId ? xs.map((x) => (x.id === id ? ex : x)) : [...xs, ex]));
    setActive(id); setSel(null); setView("ask");

    if (effectiveMode === "compare") {
      try {
        const cmp = await runCompare({ question, k: effectiveK, source_ids: scope.size ? [...scope] : undefined });
        patch(id, (x) => ({ ...x, compare: cmp, pending: false }));
      } catch (e) {
        patch(id, (x) => ({ ...x, pending: false, error: { code: "compare_failed", message: (e as Error).message, retryable: true } }));
      }
      endRef.current?.scrollIntoView({ block: "end" });
      return;
    }

    const ctl = new AbortController();
    abort.current = ctl;
    const t0 = performance.now();
    try {
      await chatStream({ question, mode: effectiveMode, k: effectiveK, source_ids: scope.size ? [...scope] : undefined, history }, (ev, d) => {
        const atMs = Math.round(performance.now() - t0);
        patch(id, (x) => {
          const sseLog: SseEventEntry[] = [...(x.sseLog ?? []).slice(-80), { event: ev, data: d, atMs }];
          switch (ev) {
            case "stage":
              return {
                ...x,
                sseLog,
                currentStage: d.name ?? x.currentStage,
                stages: d.ms != null ? [...x.stages, d] : x.stages,
              };
            case "evidence": return { ...x, sseLog, passages: d.passages, graph: d.graph };
            case "token": return { ...x, sseLog, text: x.text + d.text };
            case "done": return { ...x, sseLog, done: d, pending: false, text: d.answer };
            case "error": return { ...x, sseLog, error: d, pending: false };
            default: return { ...x, sseLog };
          }
        });
      }, ctl.signal);
    } catch (e) {
      if ((e as Error).name !== "AbortError")
        patch(id, (x) => ({ ...x, pending: false, error: { code: "network", message: "The server could not be reached.", retryable: true } }));
    }
    patch(id, (x) => (x.pending ? { ...x, pending: false } : x));
    endRef.current?.scrollIntoView({ block: "end" });
  }, [exchanges, devMode, mode, kVal, thinking, scope]);

  const submit = () => { const q = input.trim(); if (q) { setInput(""); ask(q); } };

  // keyboard map
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === "D" || e.key === "d")) {
        e.preventDefault();
        toggleDevMode();
        return;
      }
      const tag = (e.target as HTMLElement).tagName;
      const typing = tag === "INPUT" || tag === "TEXTAREA";
      if (e.key === "Escape") { setSel(null); setEvalOpen(false); setStatusOpen(false); return; }
      if (typing) return;
      if (e.key === "/") { e.preventDefault(); setView("ask"); askRef.current?.focus(); }
      else if (e.key === "p") { setView("evidence"); }
      else if ((e.key === "[" || e.key === "]") && current) {
        const cited = (current.done?.citations ?? []).map((n) => current.passages[n - 1]).filter(Boolean);
        if (!cited.length) return;
        const i = sel?.kind === "passage" ? cited.findIndex((p) => p.chunk_id === sel.id) : -1;
        const j = e.key === "]" ? (i + 1) % cited.length : (i <= 0 ? cited.length - 1 : i - 1);
        setSel({ kind: "passage", id: cited[j].chunk_id });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [current, sel, toggleDevMode]);

  const exportMd = () => {
    const md = exchanges.filter((e) => !e.pending).map((e, i) => {
      const cited = (e.done?.citations ?? []).map((n) => e.passages[n - 1]).filter(Boolean);
      const lvl = e.thinkingLevel ? THINKING_LEVELS[e.thinkingLevel].label : "Standard";
      return `## Q${i + 1}. ${e.question}\n*Thinking level: ${lvl} (K=${e.k ?? 5})*\n\n${e.done?.grounded ? e.text : "The sources don’t say."}\n\n` +
        cited.map((p) => {
          const h = cleanHeading(p.heading);
          return `${p.n}. ${humaniseTitle(p.title)}${h ? ` — ${h}` : ""} (${p.uri}, ${p.chunk_id})`;
        }).join("\n");
    }).join("\n\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([`# IntelliGraph transcript\n\n${md}\n`], { type: "text/markdown" }));
    a.download = "intelligraph-transcript.md"; a.click();
  };

  const copyRunJson = (ex: Exchange) => {
    const payload = {
      question: ex.question,
      mode: ex.mode,
      k: ex.k ?? 5,
      thinkingLevel: ex.thinkingLevel ?? "standard",
      done: ex.done ?? null,
      evidence: { passages: ex.passages },
    };
    navigator.clipboard?.writeText(JSON.stringify(payload, null, 2)).catch(() => {});
    setCopiedId(ex.id);
    setTimeout(() => setCopiedId((c) => (c === ex.id ? null : c)), 1500);
  };

  const libraryIds = sources.filter((s) => s.origin === "library" && s.status === "ready");
  const hasSeededInScope = libraryIds.length > 0 && (scope.size === 0 || libraryIds.some((s) => scope.has(s.source_id)));
  const llmDown = health ? !health.llm.ok : false;
  const dot = healthErr ? "bad" : !health ? "wait" : health.status === "ok" && !llmDown ? "ok" : "warn";

  /** Tiny markdown subset for answer prose only: bold (**), italic (*), and lists; headings in model output stripped to plain text. */
  const renderInline = (str: string, ex: Exchange, keyPrefix: string): ReactNode[] => {
    const tokens = str.split(/(\[\d+\]|\*\*[^*\n]+\*\*|\*[^*\n]+\*)/g);
    return tokens.map((tok, idx) => {
      const k = `${keyPrefix}-${idx}`;
      const mCite = /^\[(\d+)\]$/.exec(tok);
      if (mCite) {
        const n = Number(mCite[1]);
        const ps = ex.passages[n - 1];
        if (!ps) return null;
        const graphy = ps.via !== "vector";
        const on = hiPassages.has(ps.chunk_id);
        const cleanTitle = humaniseTitle(ps.title);
        return (
          <sup key={k}>
            <a href={`#fn-${ex.id}-${n}`} id={`ref-${ex.id}-${n}`}
              className={`cite ${graphy ? "via-graph" : ""} ${on ? "on" : ""}`}
              aria-label={`Passage ${n}: ${cleanTitle}`}
              onMouseEnter={() => { setHoverCite({ ex: ex.id, n }); setActive(ex.id); }} onMouseLeave={() => setHoverCite(null)}
              onFocus={() => setHoverCite({ ex: ex.id, n })} onBlur={() => setHoverCite(null)}
              onClick={(e) => { e.preventDefault(); setActive(ex.id); setSel({ kind: "passage", id: ps.chunk_id }); setView("evidence"); }}>
              {n}
            </a>
          </sup>
        );
      }
      if (tok.startsWith("**") && tok.endsWith("**") && tok.length > 4) {
        return <strong key={k}>{renderInline(tok.slice(2, -2), ex, k)}</strong>;
      }
      if (tok.startsWith("*") && tok.endsWith("*") && tok.length > 2) {
        return <em key={k}>{renderInline(tok.slice(1, -1), ex, k)}</em>;
      }
      return <Fragment key={k}>{tok}</Fragment>;
    });
  };

  const renderAnswerProse = (ex: Exchange) => {
    const blocks = ex.text.split(/\n{2,}/);
    return blocks.map((block, bIdx) => {
      const lines = block.split("\n").map((l) => l.replace(/^#{1,6}\s+/, ""));
      const isBulletList = lines.length > 0 && lines.every((l) => /^[-*•]\s+/.test(l.trim()));
      if (isBulletList) {
        return (
          <ul key={bIdx} className="answer-list">
            {lines.map((l, lIdx) => (
              <li key={lIdx}>{renderInline(l.trim().replace(/^[-*•]\s+/, ""), ex, `b${bIdx}-l${lIdx}`)}</li>
            ))}
          </ul>
        );
      }
      return (
        <p key={bIdx} className="answer" lang="en" dir="auto">
          {lines.map((l, lIdx) => (
            <Fragment key={lIdx}>
              {lIdx > 0 && <br />}
              {renderInline(l, ex, `b${bIdx}-l${lIdx}`)}
            </Fragment>
          ))}
        </p>
      );
    });
  };

  const toggleExpandPassage = (cid: string) => {
    setExpandedPassages((prev) => {
      const next = new Set(prev);
      next.has(cid) ? next.delete(cid) : next.add(cid);
      return next;
    });
  };

  const slip = (p: Passage) => {
    const cleanTitle = humaniseTitle(p.title);
    const cleanHead = cleanHeading(p.heading);
    const isExpanded = expandedPassages.has(p.chunk_id);
    const isLong = p.text.length > 260;
    return (
      <li key={p.chunk_id} id={`slip-${p.chunk_id}`} className={`slip ${hiPassages.has(p.chunk_id) ? "sel" : ""}`}>
        <div
          role="button"
          tabIndex={0}
          className="slip-btn"
          aria-pressed={hiPassages.has(p.chunk_id)}
          onClick={() => setSel(sel?.kind === "passage" && sel.id === p.chunk_id ? null : { kind: "passage", id: p.chunk_id })}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              setSel(sel?.kind === "passage" && sel.id === p.chunk_id ? null : { kind: "passage", id: p.chunk_id });
            }
          }}
        >
          <span className="slip-head">
            <span className="num">{p.n}</span>{" "}
            <strong>{cleanTitle}</strong>
            {cleanHead ? <span className="heading"> — {cleanHead}</span> : null}
          </span>
          {devMode && (
            <span className="score mono">
              {p.score == null ? "via graph" : `similarity ${p.score.toFixed(3)}`} · via {p.via} · {p.chunk_id}
            </span>
          )}
          <span className={isExpanded ? "slip-text expanded" : "slip-text clamped"}>
            {p.text}
          </span>
        </div>
        <div className="slip-foot">
          {isLong && (
            <button
              type="button"
              className="link small slip-more"
              onClick={(e) => { e.stopPropagation(); toggleExpandPassage(p.chunk_id); }}
            >
              {isExpanded ? "Show less" : "Show full passage"}
            </button>
          )}
          <details
            key={`slip-src-${p.chunk_id}-${devMode ? "dev" : "user"}`}
            className="slip-details mono"
            open={devMode ? true : undefined}
          >
            <summary>Source details</summary>
            <div className="prov">
              <div>URI / Filename: {p.uri}</div>
              <div>Chunk ID: {p.chunk_id}</div>
              {p.via !== "vector" && <div>Found through: {p.via === "both" ? "search and connection" : "a connection"}</div>}
            </div>
          </details>
        </div>
      </li>
    );
  };

  const cls = `app view-${view} ${drawer ? "drawer-open" : ""}`;
  return (
    <div className={cls}>
      <header className="masthead">
        <h1>IntelliGraph</h1>
        <p className="mono counts" aria-live="polite">
          {health ? <><span key={health.counts.sources} className="settle">{num(health.counts.sources)}</span> sources · <span key={health.counts.chunks} className="settle">{num(health.counts.chunks)}</span> passages</> : "connecting…"}
        </p>
        {devMode && <span className="dev-badge mono">Developer</span>}
        {devMode ? (
          <button className="status" onClick={() => setStatusOpen(!statusOpen)} aria-expanded={statusOpen} aria-label="System status">
            <span className={`dot ${dot}`} />
          </button>
        ) : (
          <span className="status" role="status" aria-label={`System status: ${dot === "ok" ? "healthy" : dot}`}>
            <span className={`dot ${dot}`} />
          </span>
        )}
        <button className="link lib-toggle" onClick={() => setDrawer(!drawer)}>Library</button>
        {devMode && statusOpen && (
          <div className="status-pop mono" role="status">
            {healthErr ? <div>Backend unreachable.</div> : health && <>
              {Object.entries(health.components).map(([k, v]) => <div key={k}>{k}: {v}</div>)}
              <div>llm: {health.llm.provider} · {health.llm.model || "none"} — {health.llm.ok ? "ok" : health.llm.detail}</div>
              <div>fallback: {health.llm.fallback_configured ? `configured (${health.llm.fallback_model ?? "ready"})` : "not configured"}</div>
            </>}
          </div>
        )}
      </header>

      <aside className="zone library-zone" aria-label="Library zone">
        <Library sources={sources} scope={scope} setScope={setScope} refresh={refresh} health={health} devMode={devMode} />
      </aside>

      <main className="zone transcript-zone">
        <div className="toolbar">
          <div className="thinking-group">
            <span className="mono dim toolbar-label">Thinking</span>
            <div className="seg" role="radiogroup" aria-label="Thinking level">
              {(["quick", "standard", "deep"] as ThinkingLevel[]).map((lvl) => {
                const cfg = THINKING_LEVELS[lvl];
                const isActive = thinking === lvl && (!devMode || kVal === cfg.k);
                return (
                  <button
                    key={lvl}
                    type="button"
                    role="radio"
                    aria-checked={isActive}
                    className={isActive ? "on" : ""}
                    onClick={() => selectThinking(lvl)}
                  >
                    {cfg.label}{devMode ? ` (K=${cfg.k})` : ""}
                  </button>
                );
              })}
            </div>
            <details className="thinking-help mono">
              <summary aria-label="About Thinking level">?</summary>
              <div className="thinking-help-pop">
                Quick reads {THINKING_LEVELS.quick.k} passages, Standard reads {THINKING_LEVELS.standard.k}, Deep reads {THINKING_LEVELS.deep.k} and follows more connections.
              </div>
            </details>
          </div>

          {devMode && (
            <>
              <div className="seg" role="radiogroup" aria-label="Retrieval mode">
                {(["hybrid", "vector", "graph", "compare"] as Mode[]).map((m) => (
                  <button key={m} type="button" role="radio" aria-checked={mode === m} className={mode === m ? "on" : ""} onClick={() => setMode(m)}>
                    {m[0].toUpperCase() + m.slice(1)}
                  </button>
                ))}
              </div>
              <label className="k-input-wrap mono">
                <span>K</span>
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={kVal}
                  aria-label="Numeric K retrieval depth (1 to 20)"
                  onChange={(e) => {
                    const v = Math.max(1, Math.min(20, Number(e.target.value) || 1));
                    setKVal(v);
                  }}
                />
              </label>
            </>
          )}

          <span className="spacer" />
          {devMode && evalSummary && (
            <button type="button" className="link" onClick={() => setEvalOpen(!evalOpen)} aria-expanded={evalOpen}>
              Recorded eval
            </button>
          )}
          <button type="button" className="link" onClick={exportMd} disabled={!exchanges.length}>Export Markdown</button>
          <button type="button" className="link" onClick={() => window.print()} disabled={!exchanges.length}>Print</button>
          <button type="button" className="link" onClick={() => { setExchanges([]); setSel(null); }} disabled={!exchanges.length}>Clear</button>
        </div>

        {devMode && evalOpen && evalSummary && (
          <section className="eval-drawer" aria-label="Recorded evaluation summary">
            <div className="eval-head">
              <h2 className="label">Recorded Evaluation (eval/k_sweep_summary.json)</h2>
              <button className="link small" onClick={() => setEvalOpen(false)}>Close</button>
            </div>
            <p className="mono dim">{evalSummary.caption}</p>
            <p className="mono dim">
              Selected baseline: <strong>{evalSummary.selected.architecture.toUpperCase()}</strong> ·{" "}
              {evalSummary.selected.embedding_model} · {evalSummary.selected.chunk_words}/{evalSummary.selected.overlap_words} words · K={evalSummary.selected.k}
            </p>
            <table className="eval-table mono">
              <thead>
                <tr><th>Cutoff</th><th>Architecture</th><th>Precision</th><th>Recall</th><th>F1</th><th>MRR</th><th>AUC-ROC</th></tr>
              </thead>
              <tbody>
                {(["3", "5", "10"] as const).map((k) =>
                  (["vector", "graph", "hybrid"] as const).map((arch) => {
                    const row = evalSummary.k_sweep_summary?.[k]?.[arch];
                    if (!row) return null;
                    const win = k === "5" && arch === "hybrid";
                    return (
                      <tr key={`${k}-${arch}`} className={win ? "win-row" : ""}>
                        <td>K={k}</td>
                        <td>{arch}{win ? " ★" : ""}</td>
                        <td>{Number(row.precision).toFixed(3)}</td>
                        <td>{Number(row.recall).toFixed(3)}</td>
                        <td>{Number(row.f1).toFixed(3)}</td>
                        <td>{Number(row.mrr).toFixed(3)}</td>
                        <td>{typeof row.auc_roc === "number" ? row.auc_roc.toFixed(3) : "n/a"}</td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </section>
        )}

        <div className="scroll">
          {!exchanges.length && (
            <div className="empty-state">
              <p>Add a source in the Library: a PDF, DOCX or TXT file, a web page address, or pasted text.</p>
              <p>Then ask a question. Answers use only those sources and cite the passages they come from.</p>
              {hasSeededInScope && starters.length > 0 && (
                <div className="starters">
                  <h3 className="label">From the seeded library</h3>
                  <ul className="starter-list">
                    {starters.map((sq) => (
                      <li key={sq}><button className="starter-btn" onClick={() => ask(sq)}>{sq}</button></li>
                    ))}
                  </ul>
                </div>
              )}
              {libraryIds.length === 0 && sources.length === 0 && <p className="mono dim">The library is empty.</p>}
            </div>
          )}
          {exchanges.map((ex, i) => {
            const cited = (ex.done?.citations ?? []).map((n) => ex.passages[n - 1]).filter(Boolean);
            const hc = hoverCite?.ex === ex.id ? ex.passages[hoverCite.n - 1] : null;
            const notInSources = !!ex.done && !ex.done.grounded && !ex.error;
            const retrievalOnly = !!ex.error?.code.startsWith("llm_") && ex.passages.length > 0;
            const t = ex.done?.timings;
            const workingMsg = userWorkingLine(ex);
            const lvlLabel = ex.thinkingLevel ? THINKING_LEVELS[ex.thinkingLevel].label : "Standard";
            return (
              <article key={ex.id} className={`exchange ${current?.id === ex.id ? "current" : ""}`}>
                <h2 className="q" onClick={() => setActive(ex.id)}><span className="qn mono">Q{i + 1}</span> {ex.question}</h2>

                {devMode ? (
                  <div className="ticker mono" aria-label="Search stages">
                    {ex.stages.map((s, j) => { const l = tickerLine(s); return l ? <div key={j}>{l}</div> : null; })}
                    {ex.pending && !ex.text && ex.passages.length > 0 && <div>generating answer…</div>}
                    {ex.pending && ex.passages.length === 0 && <div>searching…</div>}
                  </div>
                ) : (
                  workingMsg && <div className="working-line mono dim" role="status" aria-live="polite">{workingMsg}</div>
                )}

                {ex.compare && devMode ? (
                  <div className="paper compare-paper">
                    <div className="compare-head">
                      <span className="label">Compare Retrieval · {ex.compare.label} (K={ex.k ?? 5})</span>
                      <button onClick={() => ask(ex.question, ex.id, "hybrid")}>Generate Hybrid Answer</button>
                    </div>
                    <div className="compare-grid">
                      {(["vector", "graph", "hybrid"] as const).map((m) => {
                        const md = ex.compare!.modes[m];
                        return (
                          <div key={m} className="compare-col">
                            <h3 className="mono"><strong>{m.toUpperCase()}</strong> · {md.passages.length} passages · {md.triples.length} triples</h3>
                            <div className="mono prov-line">
                              {Object.entries(md.timings).map(([tk, tv]) => `${tk} ${tv} ms`).join(" · ")}
                            </div>
                            <ol className="plain compare-list">
                              {md.passages.map((p) => (
                                <li key={p.chunk_id}>
                                  <span className="mono">{p.chunk_id}</span> — {humaniseTitle(p.title)}
                                  {cleanHeading(p.heading) ? ` — ${cleanHeading(p.heading)}` : ""}{" "}
                                  <span className="mono prov-line">({p.score != null ? p.score.toFixed(3) : "via graph"}, {p.via})</span>
                                </li>
                              ))}
                              {!md.passages.length && <li>No passages retrieved.</li>}
                            </ol>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  <div className="paper">
                    {hc && (
                      <aside className="marginalia mono" aria-hidden="true">
                        {humaniseTitle(hc.title)}
                        {cleanHeading(hc.heading) ? ` — ${cleanHeading(hc.heading)}` : ""}
                        {devMode && <><br />{hc.score == null ? "via graph" : hc.score.toFixed(3)}</>}
                      </aside>
                    )}
                    {ex.text && !notInSources && (
                      <div className="answer-body" aria-live="polite">
                        {renderAnswerProse(ex)}
                      </div>
                    )}
                    {notInSources && (
                      <div className="not-in"><p className="answer"><strong>The sources don’t say.</strong></p>
                        <p>{ex.passages.length ? "The closest passages are listed under Evidence, so you can judge." : "No passages matched in the selected sources."}</p></div>
                    )}
                    {ex.error && !retrievalOnly && (
                      <div className="error" role="alert">
                        <p>The request could not be completed. Please try again.</p>
                        {ex.error.retryable && <button onClick={() => ask(ex.question, ex.id)}>Retry</button>}
                        <details
                          key={`err-${ex.id}-${devMode ? "dev" : "user"}`}
                          className="tech-details mono"
                          open={devMode ? true : undefined}
                        >
                          <summary>Technical detail</summary>
                          <div className="prov-line">{ex.error.code}: {ex.error.message}</div>
                        </details>
                      </div>
                    )}
                    {retrievalOnly && (
                      <div className="error" role="alert">
                        <p>The language model is unavailable. These are the passages the search found.</p>
                        <ol className="plain">
                          {ex.passages.map((p) => {
                            const ch = cleanHeading(p.heading);
                            return (
                              <li key={p.chunk_id}>
                                <strong>{humaniseTitle(p.title)}</strong>{ch ? ` — ${ch}` : ""}: {p.text.slice(0, 280)}…
                              </li>
                            );
                          })}
                        </ol>
                        <button onClick={() => ask(ex.question, ex.id)}>Retry</button>
                        <details
                          key={`err-${ex.id}-${devMode ? "dev" : "user"}`}
                          className="tech-details mono"
                          open={devMode ? true : undefined}
                        >
                          <summary>Technical detail</summary>
                          <div className="prov-line">{ex.error!.code}: {ex.error!.message}</div>
                        </details>
                      </div>
                    )}
                    {cited.length > 0 && !notInSources && (
                      <div className="sources-block">
                        <div className="sources-heading mono">Sources</div>
                        <ol className="footnotes">
                          {cited.map((p) => {
                            const ch = cleanHeading(p.heading);
                            return (
                              <li key={p.n} id={`fn-${ex.id}-${p.n}`} value={p.n} className={hiPassages.has(p.chunk_id) ? "sel" : ""}>
                                <span className="fn-title">{humaniseTitle(p.title)}</span>
                                {ch ? <span className="fn-section"> — {ch}</span> : null}{" "}
                                <a href={`#ref-${ex.id}-${p.n}`} aria-label="Back to text">↩</a>
                                {devMode && (
                                  <div className="mono prov-line">
                                    {p.uri} · {p.chunk_id}{p.score != null ? ` · ${p.score.toFixed(3)}` : " · via graph"}
                                  </div>
                                )}
                              </li>
                            );
                          })}
                        </ol>
                      </div>
                    )}
                    {(t || ex.stages.length > 0) && !ex.pending && (
                      <details
                        key={`ans-${ex.id}-${devMode ? "dev" : "user"}`}
                        className="ans-details mono"
                        open={devMode ? true : undefined}
                      >
                        <summary>Details</summary>
                        <div className="ans-details-body">
                          {ex.stages.map((s, j) => {
                            const l = tickerLine(s);
                            return l ? <div key={j}>{l}</div> : null;
                          })}
                          {t && (
                            <div>
                              {ex.done?.cached && <>from cache · </>}
                              {ex.done?.fallback && <>fallback provider · </>}
                              {t.first_token != null && <>first token {t.first_token} ms · </>}
                              retrieval {t.retrieval_total} ms
                              {t.llm != null && <> · model {t.llm} ms</>}
                              {" · "}{ex.done?.model ?? "no model call"}
                              {" · "}Thinking: {lvlLabel} (K={ex.k ?? 5}, {ex.mode})
                            </div>
                          )}
                          {devMode && (
                            <div className="dev-run-actions">
                              <button type="button" className="link small" onClick={() => copyRunJson(ex)}>
                                {copiedId === ex.id ? "Copied JSON" : "Copy run as JSON"}
                              </button>
                            </div>
                          )}
                        </div>
                      </details>
                    )}
                    {devMode && ex.sseLog && ex.sseLog.length > 0 && !ex.pending && (
                      <details className="ans-details mono sse-log-details">
                        <summary>SSE event log ({ex.sseLog.length} events)</summary>
                        <pre className="sse-log-pre">
                          {ex.sseLog.map((e) => `+${e.atMs}ms [${e.event}] ${JSON.stringify(e.data)}`).join("\n")}
                        </pre>
                      </details>
                    )}
                  </div>
                )}
              </article>
            );
          })}
          <div ref={endRef} />
        </div>

        <form className="ask" onSubmit={(e) => { e.preventDefault(); submit(); }}>
          <textarea ref={askRef} rows={1} value={input} placeholder="Ask about your sources"
            aria-label="Ask a question" onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }} />
          <button type="submit" disabled={!input.trim() || exchanges.some((e) => e.pending)}>Ask</button>
        </form>
      </main>

      <aside className="zone evidence-zone" aria-label="Evidence">
        <div className="evidence-head">
          <h2 className="label">Passages</h2>
        </div>
        <div className="evidence-body">
          {current?.passages.length ? (
            <ol className="slips">{current.passages.map((p) => slip(p))}</ol>
          ) : (
            <p className="empty">{current ? "No passages were retrieved for this question." : "Passages for the current answer appear here."}</p>
          )}
        </div>
      </aside>

      <nav className="tabbar" aria-label="Sections">
        {(["ask", "evidence", "library"] as const).map((v) => (
          <button key={v} className={view === v ? "on" : ""} onClick={() => setView(v)}>
            {v[0].toUpperCase() + v.slice(1)}
          </button>
        ))}
      </nav>

      <footer className="foot mono">
        <details><summary>Keys</summary>
          <div>/ ask line · Enter send · Shift+Enter newline · [ ] move through citations · p Passages · Esc clear selection · Ctrl+Shift+D Developer mode</div>
        </details>
        <button
          type="button"
          className={`link dev-toggle ${devMode ? "on" : ""}`}
          aria-pressed={devMode}
          onClick={toggleDevMode}
        >
          Developer
        </button>
      </footer>
    </div>
  );
}
