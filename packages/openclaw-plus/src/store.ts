/**
 * Knowledge store — PIL stages 5–8.
 *
 * Persistence: JSONL file, one artifact per line.
 * Default location: ~/.openclaw/knowledge/artifacts.jsonl
 * Override via env: KNOWLEDGE_STORE_PATH
 */

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import type { KnowledgeArtifact } from "./pipeline.js";

// ---------------------------------------------------------------------------
// Storage helpers
// ---------------------------------------------------------------------------

function storePath(): string {
  return (
    process.env["KNOWLEDGE_STORE_PATH"] ??
    join(homedir(), ".openclaw", "knowledge", "artifacts.jsonl")
  );
}

async function ensureDir(filePath: string): Promise<void> {
  const dir = dirname(filePath);
  if (!existsSync(dir)) await mkdir(dir, { recursive: true });
}

async function loadAll(): Promise<KnowledgeArtifact[]> {
  const path = storePath();
  await ensureDir(path);
  if (!existsSync(path)) return [];
  const raw = await readFile(path, "utf-8");
  return raw
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as KnowledgeArtifact);
}

async function saveAll(artifacts: KnowledgeArtifact[]): Promise<void> {
  const path = storePath();
  await ensureDir(path);
  const content = artifacts.map((a) => JSON.stringify(a)).join("\n");
  await writeFile(path, content ? content + "\n" : "", "utf-8");
}

// ---------------------------------------------------------------------------
// Similarity (Jaccard on word sets) — used for deduplication in persist()
// ---------------------------------------------------------------------------

function wordSet(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .split(/\W+/)
      .filter((w) => w.length > 2),
  );
}

function jaccard(a: string, b: string): number {
  const sa = wordSet(a);
  const sb = wordSet(b);
  const intersection = [...sa].filter((w) => sb.has(w)).length;
  const union = new Set([...sa, ...sb]).size;
  return union === 0 ? 0 : intersection / union;
}

const DUPLICATE_THRESHOLD = 0.75;

// ---------------------------------------------------------------------------
// Stage 5 — Persist
// ---------------------------------------------------------------------------

/**
 * Stage 5 — Persist: write artifact to the store with provenance.
 *
 * - If an artifact with the same id exists, it is replaced (upsert).
 * - If an active artifact with very similar content exists (Jaccard ≥ 0.75),
 *   the incoming artifact is treated as a duplicate and skipped; the existing
 *   artifact's confidence is nudged up instead.
 */
export async function persist(artifact: KnowledgeArtifact): Promise<void> {
  const all = await loadAll();

  // Upsert by id
  const existingIdx = all.findIndex((a) => a.id === artifact.id);
  if (existingIdx >= 0) {
    all[existingIdx] = artifact;
    await saveAll(all);
    return;
  }

  // Near-duplicate check (active artifacts only, same kind)
  const nearDup = all.find(
    (a) =>
      !a.retired &&
      a.kind === artifact.kind &&
      jaccard(a.content, artifact.content) >= DUPLICATE_THRESHOLD,
  );

  if (nearDup) {
    // Reinforce the existing artifact rather than storing a duplicate
    nearDup.confidence = Math.min(1, Math.round((nearDup.confidence + 0.05) * 100) / 100);
    nearDup.revisedAt = new Date().toISOString();
    await saveAll(all);
    return;
  }

  all.push(artifact);
  await saveAll(all);
}

// ---------------------------------------------------------------------------
// Stage 6 — Retrieve
// ---------------------------------------------------------------------------

/**
 * Stage 6 — Retrieve: recall relevant active artifacts for a query.
 *
 * Ranked by Jaccard similarity against the query; ties broken by confidence.
 * Retired artifacts are excluded.
 */
export async function retrieve(query: string): Promise<KnowledgeArtifact[]> {
  const all = await loadAll();
  const active = all.filter((a) => !a.retired);

  if (!query.trim()) return active;

  const scored = active
    .map((artifact) => ({
      artifact,
      score: jaccard(query, artifact.content) + artifact.confidence * 0.1,
    }))
    .filter(({ score }) => score > 0.05)
    .sort((a, b) => b.score - a.score);

  return scored.map(({ artifact }) => artifact);
}

// ---------------------------------------------------------------------------
// Stage 7 — Apply
// ---------------------------------------------------------------------------

const AUTO_APPLY_THRESHOLD = 0.8;

/**
 * Stage 7 — Apply: confidence-gated decision.
 *
 * confidence ≥ 0.8 → autoApply (inject into context silently)
 * confidence <  0.8 → suggest  (present to user for confirmation)
 */
export async function apply(
  artifact: KnowledgeArtifact,
  _context: string,
): Promise<{ suggestion: string; autoApply: boolean }> {
  const autoApply = artifact.confidence >= AUTO_APPLY_THRESHOLD;
  const label = autoApply ? "[auto-applied]" : "[suggestion]";
  return {
    suggestion: `${label} ${artifact.content}`,
    autoApply,
  };
}

// ---------------------------------------------------------------------------
// Stage 8 — Revise
// ---------------------------------------------------------------------------

/**
 * Stage 8 — Revise: update an artifact in place.
 *
 * If content changes significantly (Jaccard < 0.5 vs. original), the old
 * artifact is retired and the updated one is stored as a fresh entry with a
 * new id, preserving the audit trail.
 */
export async function revise(
  artifact: KnowledgeArtifact,
  update: Partial<KnowledgeArtifact>,
): Promise<KnowledgeArtifact> {
  const { randomUUID } = await import("node:crypto");
  const revised = { ...artifact, ...update, revisedAt: new Date().toISOString() };

  const contentChanged =
    update.content !== undefined &&
    jaccard(artifact.content, update.content) < 0.5;

  if (contentChanged) {
    // Retire original, persist new entry with fresh id
    await persist({ ...artifact, retired: true, revisedAt: revised.revisedAt });
    revised.id = randomUUID();
    revised.createdAt = revised.revisedAt;
    delete revised.revisedAt;
  }

  await persist(revised);
  return revised;
}
