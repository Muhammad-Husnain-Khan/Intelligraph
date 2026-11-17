export type ThinkingLevel = "quick" | "standard" | "deep";

export const THINKING_LEVELS: Record<ThinkingLevel, { label: string; k: number }> = {
  quick: { label: "Quick", k: 3 },
  standard: { label: "Standard", k: 5 },
  deep: { label: "Deep", k: 10 },
};

export const DEFAULT_THINKING_LEVEL: ThinkingLevel = "standard";

/** Strip markdown characters (*, _, #, `) and leading/trailing punctuation from a heading string. */
export function cleanHeading(h: string | null | undefined): string {
  if (!h) return "";
  return h.replace(/[*_#`]+/g, "").replace(/\s+/g, " ").replace(/^[\s\-–—:]+|[\s\-–—:]+$/g, "");
}

/** Humanise a filename or raw source title without mutating the stored URI. */
export function humaniseTitle(raw: string | null | undefined): string {
  if (!raw) return "Source";
  let s = raw.trim();
  // Drop common document extensions
  s = s.replace(/\.(pdf|docx|txt|md|html?)$/i, "");
  // Strip markdown characters
  s = s.replace(/[*_#`]+/g, " ");
  // Replace hyphens/underscores with spaces
  s = s.replace(/[-_]+/g, " ");
  // Remove trailing copy markers like "(1)" or stray trailing single digits left from "_1_"
  s = s.replace(/\s*\(\d+\)\s*$/, "").replace(/\s+\d+\s*$/, "");
  s = s.replace(/\s+/g, " ").replace(/^[\s.\-–—_()[\]]+|[\s.\-–—_()[\]]+$/g, "");
  return s || raw.trim();
}
