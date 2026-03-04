/**
 * PIL (Persistable Interactive Learning) pipeline — stages 1–4.
 *
 * These are pure transformation functions (no I/O).
 * Stages 5–8 (persist, retrieve, apply, revise) live in store.ts.
 */

import { randomUUID } from "node:crypto";

export type KnowledgeKind =
  | "procedure"   // reusable step-by-step workflows
  | "preference"  // user communication / formatting preferences
  | "convention"  // domain-specific terminology or org norms
  | "judgment"    // values, tradeoffs, success criteria
  | "strategy"    // repeatable problem-solving approaches
  | "fact";       // stable operational facts with access controls

export type KnowledgeArtifact = {
  id: string;
  kind: KnowledgeKind;
  content: string;
  confidence: number;    // 0–1; gates auto-apply (≥0.8) vs. suggest
  provenance: string;    // source session/message reference
  createdAt: string;     // ISO 8601
  revisedAt?: string;    // ISO 8601, set on each revision
  retired?: boolean;     // true when superseded or invalidated
};

// ---------------------------------------------------------------------------
// Stage 1 — Elicit
// ---------------------------------------------------------------------------

const SIGNAL_PATTERNS = [
  /\balways\b/i,
  /\bnever\b/i,
  /\bprefer\b/i,
  /\bi (like|want|need|hate|love)\b/i,
  /\bmake sure\b/i,
  /\bevery time\b/i,
  /\bwhenever\b/i,
  /\bimportant\b/i,
  /\bcritical\b/i,
  /\bmust\b/i,
  /\bplease (don't|avoid|use|always)\b/i,
  /\bdon't\b/i,
  /\bdo not\b/i,
];

/**
 * Stage 1 — Elicit: split input into sentences and return those that contain
 * signal words suggesting persistent, reusable knowledge.
 */
export function elicit(input: string): string[] {
  // Split on sentence-ending punctuation followed by whitespace/end-of-string
  const sentences = input
    .split(/(?<=[.!?])\s+|(?<=[.!?])$/)
    .map((s) => s.trim())
    .filter(Boolean);

  return sentences.filter((sentence) =>
    SIGNAL_PATTERNS.some((pattern) => pattern.test(sentence)),
  );
}

// ---------------------------------------------------------------------------
// Stage 2 — Induce
// ---------------------------------------------------------------------------

type KindRule = { pattern: RegExp; kind: KnowledgeKind };

const KIND_RULES: KindRule[] = [
  { pattern: /\b(prefer|like|want|wish|rather|instead|hate|love)\b/i,       kind: "preference"  },
  { pattern: /\b(step|first[,\s]then|next[,\s]|finally|process|workflow|procedure)\b/i, kind: "procedure" },
  { pattern: /\b(term|means|defined as|called|refers to|known as)\b/i,       kind: "convention"  },
  { pattern: /\b(because|tradeoff|trade-off|value|priority|balance|weight)\b/i, kind: "judgment" },
  { pattern: /\b(approach|strategy|way to|method|technique|pattern|tactic)\b/i, kind: "strategy" },
];

/**
 * Stage 2 — Induce: classify a candidate sentence into a typed KnowledgeArtifact.
 * Returns null for empty input.
 */
export function induce(candidate: string, provenance = "unknown"): KnowledgeArtifact | null {
  const trimmed = candidate.trim();
  if (!trimmed) return null;

  let kind: KnowledgeKind = "fact";
  for (const { pattern, kind: k } of KIND_RULES) {
    if (pattern.test(trimmed)) {
      kind = k;
      break;
    }
  }

  return {
    id: randomUUID(),
    kind,
    content: trimmed,
    confidence: 0.5, // baseline; validate() adjusts this
    provenance,
    createdAt: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Stage 3 — Validate
// ---------------------------------------------------------------------------

const HEDGES      = ["maybe", "sometimes", "might", "could", "possibly", "perhaps", "not sure", "i think", "probably"];
const ASSERTIONS  = ["always", "never", "every", "must", "will", "definitely", "certainly", "without exception"];

/**
 * Stage 3 — Validate: adjust confidence based on linguistic signals and
 * content heuristics.
 */
export function validate(artifact: KnowledgeArtifact): KnowledgeArtifact {
  const lower = artifact.content.toLowerCase();
  let confidence = artifact.confidence;

  for (const h of HEDGES)     { if (lower.includes(h)) confidence -= 0.1; }
  for (const a of ASSERTIONS) { if (lower.includes(a)) confidence += 0.1; }

  // Very short statements are likely too vague to be useful
  const wordCount = artifact.content.split(/\s+/).length;
  if (wordCount < 4)  confidence -= 0.15;
  if (wordCount > 60) confidence -= 0.10; // overly long = noisy

  // Clamp to [0, 1]
  confidence = Math.min(1, Math.max(0, Math.round(confidence * 100) / 100));

  return { ...artifact, confidence };
}

// ---------------------------------------------------------------------------
// Stage 4 — Compact
// ---------------------------------------------------------------------------

/**
 * Stage 4 — Compact: normalise whitespace and punctuation.
 * Deduplication against the existing store happens inside store.persist().
 */
export function compact(artifact: KnowledgeArtifact): KnowledgeArtifact {
  const content = artifact.content
    .trim()
    .replace(/\s+/g, " ")             // collapse runs of whitespace
    .replace(/\s([.,!?;:])/g, "$1");  // remove space before punctuation

  return { ...artifact, content };
}
