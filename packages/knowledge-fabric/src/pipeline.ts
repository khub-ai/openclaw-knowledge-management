/**
 * PIL pipeline — orchestration layer.
 *
 * Combines extraction (extract.ts) and storage (store.ts) to process a user
 * message end-to-end: extract knowledge → match against store → accumulate
 * or create → determine what is injectable.
 *
 * Backward-compatibility note: the old synchronous elicit / induce / validate /
 * compact functions are retained below (marked @deprecated) so that existing
 * code (e.g., the playground) continues to compile without changes.
 */

import { randomUUID } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import {
  type KnowledgeArtifact,
  type KnowledgeKind,
  type LLMFn,
} from "./types.js";
import {
  type ExtractionCandidate,
  extractFromMessage,
  candidateToArtifact,
} from "./extract.js";
import {
  getActiveTags,
  matchCandidate,
  accumulateEvidence,
  persist,
  getInjectLabel,
  revise,
  detectConflicts,
  type InjectLabel,
} from "./store.js";

// Re-export core types for backward compatibility
export type { KnowledgeKind, KnowledgeArtifact } from "./types.js";
export type { ExtractionCandidate } from "./extract.js";

// ---------------------------------------------------------------------------
// processMessage result types
// ---------------------------------------------------------------------------

export type InjectableArtifact = {
  artifact: KnowledgeArtifact;
  label: InjectLabel;
};

/**
 * Phase 2b — Represents a detected contradiction between a newly created or
 * updated artifact and an existing consolidated artifact.
 *
 * The conflict is stored on both artifacts as a `contradicts` relation edge.
 * The `newArtifact` has its `confidence` penalized by ~30% to signal that the
 * knowledge is disputed until the user resolves the contradiction.
 */
export type ConflictReport = {
  /** The newly created or updated artifact (penalized confidence, has contradicts edge) */
  newArtifact: KnowledgeArtifact;
  /** The existing consolidated artifact it contradicts (also has contradicts back-edge) */
  conflictingArtifact: KnowledgeArtifact;
  /** One-sentence explanation of the contradiction */
  explanation: string;
};

export type ProcessResult = {
  /** Candidates the LLM extracted from the message */
  candidates: ExtractionCandidate[];
  /** Newly created artifacts (novel knowledge not seen before) */
  created: KnowledgeArtifact[];
  /** Updated artifacts (evidence accumulated or newly consolidated) */
  updated: KnowledgeArtifact[];
  /**
   * Artifacts from this message that are injectable into the current session.
   *
   * Note: for retrieving artifacts from *previous* sessions relevant to a
   * query, use retrieve() from store.ts directly.
   */
  injectable: InjectableArtifact[];
  /**
   * Phase 2b — Conflicts detected between new/updated artifacts and existing
   * consolidated knowledge. Empty array when conflict detection is disabled
   * (conflictLlm = null) or when no contradictions are found.
   */
  conflicts: ConflictReport[];
};

// ---------------------------------------------------------------------------
// Main pipeline entrypoint: processMessage
// ---------------------------------------------------------------------------

/**
 * Process a user message through the full PIL pipeline.
 *
 * Stages:
 *   1. Extract   — LLM identifies persistable knowledge candidates
 *   2. Match     — compare each candidate against existing store artifacts
 *   3. Resolve   — novel → create; accumulating → add evidence; confident → update stats
 *   3.5 Conflict — check new/updated artifacts for contradictions (Phase 2b)
 *   4. Decide    — determine which newly-created/updated artifacts are injectable
 *
 * @param message     - User's message in any language
 * @param llm         - LLM adapter function (used for extraction + consolidation)
 * @param provenance  - Session or message reference stored with each artifact
 * @param matchLlm    - Optional separate LLM for Stage 2 semantic matching.
 *                      Defaults to `llm`. Pass `null` to disable semantic matching.
 *                      A cheap/fast model (e.g. haiku) works well here.
 * @param conflictLlm - Optional LLM for Phase 2b conflict detection.
 *                      Defaults to `matchLlm ?? llm`. Pass `null` to disable
 *                      conflict detection entirely (e.g. high-throughput batch
 *                      processing where conflicts are checked separately).
 */
export async function processMessage(
  message: string,
  llm: LLMFn,
  provenance = "unknown",
  matchLlm: LLMFn | null = llm,
  conflictLlm: LLMFn | null = matchLlm ?? llm,
): Promise<ProcessResult> {
  // ── Stage 1: Extract ──────────────────────────────────────────────────────
  const existingTags = await getActiveTags();
  const candidates = await extractFromMessage(message, existingTags, llm);

  const created: KnowledgeArtifact[] = [];
  const updated: KnowledgeArtifact[] = [];

  // ── Stages 2 & 3: Match → Resolve ────────────────────────────────────────
  for (const candidate of candidates) {
    const match = await matchCandidate(candidate, matchLlm ?? undefined);

    if (match === null) {
      // Novel: create new candidate artifact
      const artifact = candidateToArtifact(candidate, provenance);
      await persist(artifact);
      created.push(artifact);
    } else {
      // Accumulating or confident: add evidence, trigger consolidation if ready
      const updated_ = await accumulateEvidence(match, candidate.content, llm);
      updated.push(updated_);
    }
  }

  // ── Stage 3.5: Conflict detection (Phase 2b) ─────────────────────────────
  //
  // For each new or updated artifact, ask a cheap LLM whether it directly
  // contradicts any existing consolidated artifact with overlapping content.
  // On conflict:
  //   - The new artifact's confidence is penalized 30% (disputed knowledge)
  //   - Both artifacts receive a "contradicts" relation edge (bidirectional)
  //   - The penalized version replaces the original in created/updated arrays
  //
  // Stage 4 (injectable decision) runs on the penalized artifacts, so a
  // contradicted claim is less likely to auto-inject as [established].
  const conflicts: ConflictReport[] = [];
  if (conflictLlm !== null && (created.length > 0 || updated.length > 0)) {
    const toCheck = [...created, ...updated];
    for (const artifact of toCheck) {
      const detected = await detectConflicts(artifact, conflictLlm);
      for (const { conflictingArtifact, explanation } of detected) {
        // Penalize confidence: contradicted claim is uncertain until resolved
        const penalized: KnowledgeArtifact = {
          ...artifact,
          confidence: Math.max(
            0.10,
            Math.round(artifact.confidence * 0.70 * 100) / 100,
          ),
          relations: [
            ...(artifact.relations ?? []),
            { type: "contradicts" as const, targetId: conflictingArtifact.id, note: explanation },
          ],
        };
        await persist(penalized);

        // Add reciprocal edge to the existing artifact (deduplicated)
        const updatedRelations = [
          ...(conflictingArtifact.relations ?? []).filter(
            (r) => !(r.type === "contradicts" && r.targetId === artifact.id),
          ),
          { type: "contradicts" as const, targetId: artifact.id, note: explanation },
        ];
        await revise(conflictingArtifact, { relations: updatedRelations });

        // Replace in created/updated so Stage 4 sees the penalized version
        const ci = created.findIndex((a) => a.id === artifact.id);
        if (ci >= 0) created[ci] = penalized;
        const ui = updated.findIndex((a) => a.id === artifact.id);
        if (ui >= 0) updated[ui] = penalized;

        conflicts.push({ newArtifact: penalized, conflictingArtifact, explanation });
      }
    }
  }

  // ── Stage 4: Decide injectable ────────────────────────────────────────────
  const injectable: InjectableArtifact[] = [];
  for (const a of [...created, ...updated]) {
    const label = getInjectLabel(a);
    if (label) injectable.push({ artifact: a, label });
  }

  return { candidates, created, updated, injectable, conflicts };
}

// ---------------------------------------------------------------------------
// Format injectable artifacts for LLM prompt injection
// ---------------------------------------------------------------------------

/**
 * Render a list of injectable artifacts as a formatted string suitable for
 * insertion into an LLM system prompt or context block.
 *
 * Labels:
 *   [established]  — consolidated, high-confidence knowledge; treat as fact
 *   [suggestion]   — consolidated but lower-confidence; apply with judgment
 *   [provisional]  — single strong observation; use with caution
 */
export function formatForInjection(injectables: InjectableArtifact[]): string {
  if (injectables.length === 0) return "";

  const lines = injectables.map(
    ({ artifact, label }) => `${label} [${artifact.kind}] ${artifact.content}`,
  );

  return (
    "Remembered user knowledge (apply as appropriate):\n" + lines.join("\n")
  );
}

// ---------------------------------------------------------------------------
// Compile procedure artifact → executable program
// ---------------------------------------------------------------------------

/**
 * Build a code-generation prompt from a procedure artifact and related
 * supporting knowledge (conventions, facts, etc.).
 *
 * The prompt instructs the LLM to return ONLY source code — no markdown
 * fences, no prose — so the output can be written directly to a file.
 */
function buildCodeGenPrompt(
  procedure: KnowledgeArtifact,
  relatedArtifacts: KnowledgeArtifact[],
  language: string,
): string {
  const related = relatedArtifacts
    .map((a) => `[${a.kind}] ${a.content}`)
    .join("\n");

  return `You are a ${language} programming expert generating a self-contained automation script.

PROCEDURE TO AUTOMATE:
${procedure.content}

SUPPORTING KNOWLEDGE (conventions, facts, and other relevant context):
${related || "(none)"}

Requirements:
- Runnable as \`${language} script.py\` with no interactive prompts
- One clear comment per logical section
- Print progress to stdout as each step executes
- Graceful error handling (try/except or equivalent); exit with non-zero code on failure
- Credentials and site-specific config should be read from environment variables or a
  config file — never hard-coded

Return ONLY the ${language} source code. No markdown fences, no explanation, no preamble.`.trim();
}

/**
 * Derive a filesystem-safe filename for a compiled program.
 *
 * Uses the first two non-generic procedure tags joined with "-".
 * Falls back to "procedure" if no useful tags are present.
 */
function deriveFilename(procedure: KnowledgeArtifact, language: string): string {
  const ext = language === "python" ? "py" : language;
  const skip = new Set(["general", "specific", "procedure", "script", "automation"]);
  const base = (procedure.tags ?? [])
    .filter((t) => !skip.has(t))
    .slice(0, 2)
    .join("-") || "procedure";
  return `${base}.${ext}`;
}

/**
 * Generate an executable program from a procedure artifact's accumulated
 * knowledge and attach it to the artifact.
 *
 * The procedure recipe (content) is always retained as primary documentation
 * and manual fallback. The compiled program is stored in `artifact.program`
 * and optionally written to `saveDir` on disk.
 *
 * @param procedure        - The procedure artifact to compile
 * @param relatedArtifacts - Supporting artifacts (conventions, facts, …)
 * @param language         - Target language, e.g. "python" or "bash"
 * @param llm              - LLM adapter used for code generation
 * @param saveDir          - Directory to save the generated file; if omitted
 *                           the code is stored in the artifact but not on disk
 */
export async function compileToProgram(
  procedure: KnowledgeArtifact,
  relatedArtifacts: KnowledgeArtifact[],
  language: string,
  llm: LLMFn,
  saveDir?: string,
): Promise<KnowledgeArtifact> {
  // ── Generate the code ────────────────────────────────────────────────────
  const prompt = buildCodeGenPrompt(procedure, relatedArtifacts, language);
  const code = (await llm(prompt)).trim();

  // ── Optionally save to disk ───────────────────────────────────────────────
  let filePath: string | undefined;
  if (saveDir) {
    await mkdir(saveDir, { recursive: true });
    const filename = deriveFilename(procedure, language);
    filePath = resolve(join(saveDir, filename));
    await writeFile(filePath, code, "utf-8");
  }

  // ── Attach to the procedure artifact (in-place update via revise) ─────────
  const now = new Date().toISOString();
  const updated = await revise(procedure, {
    program: {
      language,
      code,
      path: filePath,
      generatedAt: now,
    },
  });

  return updated;
}

// ---------------------------------------------------------------------------
// Backward-compatible synchronous stubs (deprecated)
// ---------------------------------------------------------------------------

/** @deprecated Use processMessage() with an LLM adapter instead. */
export function elicit(input: string): string[] {
  // Naive sentence splitter — retained for backward compat only
  return input
    .split(/(?<=[.!?])\s+|(?<=[.!?])$/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/** @deprecated Use processMessage() with an LLM adapter instead. */
export function induce(
  candidate: string,
  provenance = "unknown",
): KnowledgeArtifact | null {
  const trimmed = candidate.trim();
  if (!trimmed) return null;
  return {
    id: randomUUID(),
    kind: "fact" as KnowledgeKind,
    content: trimmed,
    confidence: 0.5,
    provenance,
    createdAt: new Date().toISOString(),
  };
}

/** @deprecated Use processMessage() with an LLM adapter instead. */
export function validate(artifact: KnowledgeArtifact): KnowledgeArtifact {
  return artifact; // No-op; validation now performed by LLM at extraction
}

/** @deprecated Use processMessage() with an LLM adapter instead. */
export function compact(artifact: KnowledgeArtifact): KnowledgeArtifact {
  const content = artifact.content
    .trim()
    .replace(/\s+/g, " ")
    .replace(/\s([.,!?;:])/g, "$1");
  return { ...artifact, content };
}
