import type { ThinkingLevel } from "./config";

export type Via = "vector" | "graph" | "both";
export interface Passage {
  n: number; chunk_id: string; source_id: string; title: string; uri: string;
  heading: string; text: string; score: number | null; via: Via;
}
export interface GNode { id: string; label: string; seed: boolean; degree: number; bridge?: boolean }
export interface GEdge { id: string; source: string; target: string; chunk_ids: string[]; weight: number }
export interface Graph { nodes: GNode[]; edges: GEdge[] }
export interface StageEvent { name: string; ms?: number; detail?: string; state?: string }
export interface Done {
  answer: string; citations: number[]; grounded: boolean; timings: Record<string, number>;
  model: string | null; fallback?: boolean; cached?: boolean; no_passages?: boolean;
}
export interface ErrorEv { code: string; message: string; retryable: boolean }
export type Mode = "hybrid" | "vector" | "graph" | "compare";

export interface CompareModeData {
  timings: Record<string, number>;
  passages: { chunk_id: string; title: string; heading?: string; score: number | null; via: Via }[];
  triples: { s: string; o: string; chunk_id: string }[];
}
export interface CompareResult {
  label: string;
  question: string;
  modes: Record<"vector" | "graph" | "hybrid", CompareModeData>;
}

export interface SseEventEntry {
  event: string;
  data: unknown;
  atMs: number;
}

export interface Exchange {
  id: string;
  question: string;
  mode: Mode;
  k?: number;
  thinkingLevel?: ThinkingLevel;
  stages: StageEvent[];
  currentStage?: string;
  passages: Passage[];
  graph: Graph;
  text: string;
  done?: Done;
  error?: ErrorEv;
  pending: boolean;
  compare?: CompareResult;
  sseLog?: SseEventEntry[];
}

export interface Source {
  source_id: string; title: string; uri: string; origin: "library" | "upload"; kind: string;
  status: string; error: string | null; chunks: number; entities: number;
}
export interface IngestEvent {
  stage: string; state: string; ms?: number; words?: number; chunks?: number;
  entities?: number; edges?: number; reason?: string;
}
export interface Health {
  status: string; components: Record<string, string>;
  llm: {
    provider: string; model: string; ok: boolean; detail: string;
    privacy_note?: string | null; fallback_configured?: boolean; fallback_model?: string | null;
  };
  counts: { sources: number; chunks: number };
}

export type Selection =
  | { kind: "passage"; id: string }
  | { kind: "node"; id: string }
  | { kind: "edge"; id: string }
  | null;

export interface EvalArchMetrics {
  n: number; precision: number; recall: number; f1: number; mrr: number; auc_roc: number | string;
  [key: string]: number | string;
}
export interface EvalSummary {
  caption: string;
  selected: { architecture: string; embedding_model: string; chunk_words: number; overlap_words: number; k: number };
  metrics_summary: Record<string, EvalArchMetrics>;
  k_sweep_summary: Record<string, Record<string, EvalArchMetrics>>;
  bias_report: { by_domain?: Record<string, Record<string, EvalArchMetrics>> };
}
