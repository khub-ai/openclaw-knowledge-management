/**
 * Benchmark: "lmp" acronym scenario
 *
 * Documents and validates the PIL pipeline's handling of personal shorthands
 * taught through organic conversation rather than explicit statements.
 *
 * The scenario:
 *   1. User types "lmp" — the assistant doesn't know what it means.
 *   2. User clarifies: "it means 'list my preferences'".
 *   3. PIL must learn "lmp = list my preferences" from the *exchange*,
 *      not just the isolated user message (which is a dangling pronoun).
 *   4. In the next session, "lmp" is correctly understood.
 *   5. When a second acronym ("atp") is introduced, semantic matching
 *      groups it with "lmp" rather than creating an orphan artifact.
 *   6. A third acronym ("ds") triggers consolidation into a general rule:
 *      "User defines personal shorthand acronyms for frequently-used commands."
 *
 * This file serves two purposes:
 *   - Automated regression test (run with: pnpm test)
 *   - Instructional documentation of expected PIL behavior
 *
 * See also: docs/benchmarks/10-lmp-acronym.md
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { unlink } from "node:fs/promises";
import { existsSync } from "node:fs";
import { processMessage } from "@khub-ai/knowledge-fabric/pipeline";
import { retrieve, loadAll } from "@khub-ai/knowledge-fabric/store";
import { CONSOLIDATION_THRESHOLD } from "@khub-ai/knowledge-fabric/types";
import {
  createLmpBenchmarkLLM,
  LMP_EXCHANGE_RESPONSE,
  ACRONYM_CONSOLIDATION_RULE,
} from "./mock-llm.js";

// ---------------------------------------------------------------------------
// Test isolation — each test gets its own ephemeral store
// ---------------------------------------------------------------------------

let testStorePath: string;

beforeEach(() => {
  testStorePath = join(
    tmpdir(),
    `pil-lmp-bench-${Date.now()}-${Math.random().toString(36).slice(2)}.jsonl`,
  );
  process.env["KNOWLEDGE_STORE_PATH"] = testStorePath;
});

afterEach(async () => {
  delete process.env["KNOWLEDGE_STORE_PATH"];
  if (existsSync(testStorePath)) await unlink(testStorePath);
});

// ---------------------------------------------------------------------------
// Simulated conversation exchanges
//
// Each exchange is the text passed to the exchange-level PIL pass:
//   "User: <what user typed>\nAssistant: <what the bot replied>"
//
// The assistant's reply is realistic but canned for test stability —
// what matters is that the combined text gives PIL enough context to
// resolve co-references like "it means list my preferences".
// ---------------------------------------------------------------------------

const LMP_EXCHANGE =
  "User: lmp\n" +
  "Assistant: I'm not sure what 'lmp' means. Could you clarify?\n" +
  "User: it means 'list my preferences'";

const ATP_EXCHANGE =
  "User: atp\n" +
  "Assistant: I don't recognize 'atp'. What does it stand for?\n" +
  "User: it means 'add to preferences'";

const DS_EXCHANGE =
  "User: ds\n" +
  "Assistant: What does 'ds' stand for?\n" +
  "User: it means 'debug session'";

// ---------------------------------------------------------------------------
// Phase 1 — Exchange-level extraction
// ---------------------------------------------------------------------------

describe(`Benchmark lmp/Phase 1 — exchange-level extraction`, () => {
  it("extracts the 'lmp' convention from the clarification exchange", async () => {
    // The key insight: PIL processes the FULL exchange (User + Assistant + User),
    // so it sees "lmp" and "list my preferences" together and can connect them.
    // If only the isolated user message "it means 'list my preferences'" were
    // processed, PIL would see a dangling pronoun and extract nothing useful.
    const llm = createLmpBenchmarkLLM();

    const result = await processMessage(LMP_EXCHANGE, llm, "pil-chat:exchange");

    expect(result.candidates).toHaveLength(1);
    expect(result.created).toHaveLength(1);
    expect(result.updated).toHaveLength(0);

    const artifact = result.created[0]!;
    expect(artifact.kind).toBe("convention");
    expect(artifact.stage).toBe("candidate");
    expect(artifact.content).toContain("list my preferences");
    // Definitive certainty → injectable immediately as [provisional]
    expect(result.injectable).toHaveLength(1);
    expect(result.injectable[0]!.label).toBe("[provisional]");
  });
});

// ---------------------------------------------------------------------------
// Phase 2 — Cross-session recall
// ---------------------------------------------------------------------------

describe("Benchmark lmp/Phase 2 — cross-session recall", () => {
  it("retrieves the lmp convention in a simulated next session", async () => {
    // Session 1: teach lmp
    const llm = createLmpBenchmarkLLM();
    await processMessage(LMP_EXCHANGE, llm, "pil-chat:exchange");

    // Session 2 (same store, new process would start here in real usage):
    // The user types "lmp" and the retrieve() call should surface the artifact.
    const results = await retrieve("lmp");

    expect(results.length).toBeGreaterThan(0);
    expect(results[0]!.content).toContain("list my preferences");
  });

  it("lmp convention is injectable in the next session's system prompt", async () => {
    const llm = createLmpBenchmarkLLM();
    await processMessage(LMP_EXCHANGE, llm, "pil-chat:exchange");

    const results = await retrieve("lmp");
    const { getInjectLabel } = await import("@khub-ai/knowledge-fabric/store");
    const injectableResults = results.filter((a) => getInjectLabel(a) !== null);

    // The stored artifact is definitive-certainty → [provisional] injectable
    expect(injectableResults.length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Phase 3a — Without semantic matching (Jaccard-only baseline)
// ---------------------------------------------------------------------------

describe("Benchmark lmp/Phase 3a — WITHOUT semantic matching (Jaccard baseline)", () => {
  it(`creates ${CONSOLIDATION_THRESHOLD} separate orphan artifacts when tags differ`, async () => {
    // This test demonstrates the PROBLEM that semantic matching solves.
    //
    // The mock LLM assigns different tags to each acronym:
    //   lmp → tags: ["lmp", "shorthand-command", "acronym"]
    //   atp → tags: ["atp", "command-alias"]           ← Jaccard = 0 vs lmp
    //   ds  → tags: ["ds", "abbreviation"]             ← Jaccard = 0 vs lmp
    //
    // Without semantic matching (matchLlm = null), each creates a new artifact.
    // No consolidation can occur because all evidenceCount values stay at 1.
    const llm = createLmpBenchmarkLLM();

    // Pass null as matchLlm to use Jaccard-only matching
    await processMessage(LMP_EXCHANGE, llm, "pil-chat:exchange", null);
    await processMessage(ATP_EXCHANGE, llm, "pil-chat:exchange", null);
    await processMessage(DS_EXCHANGE,  llm, "pil-chat:exchange", null);

    const all = await loadAll();
    const active = all.filter((a) => !a.retired);

    // All three are separate artifacts — no grouping occurred
    expect(active).toHaveLength(3);
    // None have reached the consolidation threshold
    expect(active.every((a) => a.stage !== "consolidated")).toBe(true);
    // Each has evidenceCount=1 (isolated, no accumulation)
    expect(active.every((a) => (a.evidenceCount ?? 1) === 1)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Phase 3b — With semantic matching
// ---------------------------------------------------------------------------

describe("Benchmark lmp/Phase 3b — WITH semantic matching", () => {
  it("groups the second acronym (atp) onto the first (lmp) artifact", async () => {
    // With semantic matching enabled, matchCandidate asks the LLM:
    //   "Are 'lmp = list my preferences' and 'atp = add to preferences'
    //    instances of the SAME underlying behavioral pattern?"
    // The LLM answers "1" (yes, both are user-defined acronyms for commands).
    // accumulateEvidence is called → evidenceCount grows to 2.
    const llm = createLmpBenchmarkLLM();

    await processMessage(LMP_EXCHANGE, llm, "pil-chat:exchange", llm);
    const result = await processMessage(ATP_EXCHANGE, llm, "pil-chat:exchange", llm);

    // Second acronym: update (not create)
    expect(result.created).toHaveLength(0);
    expect(result.updated).toHaveLength(1);
    expect(result.updated[0]!.evidenceCount).toBe(2);
    expect(result.updated[0]!.stage).toBe("accumulating");

    // Store still has only ONE artifact (not two orphans)
    const all = await loadAll();
    const active = all.filter((a) => !a.retired);
    expect(active).toHaveLength(1);
  });

  it(`consolidates into a general rule after ${CONSOLIDATION_THRESHOLD} acronym exchanges`, async () => {
    // After three observations on the same artifact, accumulateEvidence triggers
    // the consolidation LLM call, which distills all evidence into a single rule:
    //   "User regularly defines personal shorthand acronyms for frequently-used
    //    commands (e.g., lmp = list preferences, atp = add to preferences, ...)"
    const llm = createLmpBenchmarkLLM();

    await processMessage(LMP_EXCHANGE, llm, "pil-chat:exchange", llm);
    await processMessage(ATP_EXCHANGE, llm, "pil-chat:exchange", llm);
    const result = await processMessage(DS_EXCHANGE,  llm, "pil-chat:exchange", llm);

    // Third exchange triggers consolidation
    expect(result.updated).toHaveLength(1);
    expect(result.updated[0]!.stage).toBe("consolidated");
    expect(result.updated[0]!.content).toBe(ACRONYM_CONSOLIDATION_RULE);
    expect(result.updated[0]!.evidenceCount).toBe(CONSOLIDATION_THRESHOLD);

    // Store has exactly one consolidated artifact
    const all = await loadAll();
    const consolidated = all.filter((a) => a.stage === "consolidated");
    expect(consolidated).toHaveLength(1);

    // The consolidated artifact has higher confidence (boost applied at consolidation)
    expect(consolidated[0]!.confidence).toBeGreaterThan(0.65);
  });

  it("consolidated artifact is injectable as [established] when confidence is high enough", async () => {
    const llm = createLmpBenchmarkLLM();

    // Teach all three acronyms → triggers consolidation
    await processMessage(LMP_EXCHANGE, llm, "pil-chat:exchange", llm);
    await processMessage(ATP_EXCHANGE, llm, "pil-chat:exchange", llm);
    await processMessage(DS_EXCHANGE,  llm, "pil-chat:exchange", llm);

    const all = await loadAll();
    const consolidated = all.find((a) => a.stage === "consolidated")!;
    expect(consolidated).toBeTruthy();

    const { getInjectLabel } = await import("@khub-ai/knowledge-fabric/store");
    const { DEFAULT_AUTO_APPLY_THRESHOLD } = await import("@khub-ai/knowledge-fabric/types");

    const label = getInjectLabel(consolidated);
    if (consolidated.confidence >= DEFAULT_AUTO_APPLY_THRESHOLD) {
      expect(label).toBe("[established]");
    } else {
      expect(label).toBe("[suggestion]");
    }
    // Either way, the consolidated artifact IS injectable
    expect(label).not.toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Semantic match prompt sanity-check
// ---------------------------------------------------------------------------

describe("Benchmark lmp/Semantic match prompt format", () => {
  it("mock LLM correctly identifies the semantic match prompt", async () => {
    // Regression guard: if buildSemanticMatchPrompt changes its wording,
    // the mocks would silently stop working. This test ensures the phrase
    // "SAME underlying behavioral habit" is still present in the prompt.
    //
    // We verify this indirectly: if the mock returns "1" for the ATP exchange
    // (meaning it correctly detected and answered the semantic match prompt),
    // then the phrase must still be there.
    const llm = createLmpBenchmarkLLM();

    await processMessage(LMP_EXCHANGE, llm, "pil-chat:exchange", llm);

    // Teach atp with semantic matching — should UPDATE, not CREATE
    const atpResult = await processMessage(ATP_EXCHANGE, llm, "pil-chat:exchange", llm);

    // If this is 1 (update) rather than 0 (no update), the semantic match fired
    expect(atpResult.updated).toHaveLength(1);
    // If this were 1, the mock didn't catch the prompt → phrase changed
    expect(atpResult.created).toHaveLength(0);
  });

  it("semantic match returning NONE creates a new artifact rather than accumulating", async () => {
    // Verify the semantic mock has correct negative cases: a convention whose
    // content doesn't match any "1" trigger phrase should NOT be grouped with
    // the lmp artifact — it should create its own new artifact.
    //
    // We use a custom mock with a deliberately unique extraction phrase
    // that appears only in the exchange text (not in any semantic match prompt),
    // so the two call sites (extraction vs semantic match) are unambiguously
    // distinguished.
    const llm = createLmpBenchmarkLLM();
    await processMessage(LMP_EXCHANGE, llm, "pil-chat:exchange", llm);

    const { createMultiPatternMockLLM } = await import("./mock-llm.js");

    // An extraction mock for a convention with zero tag overlap with lmp.
    // Uses a highly unique marker phrase so there's no risk of cross-matching.
    const UNRELATED_PHRASE = "zz-unique-bench-marker-xqzt";
    const unrelatedResponse = JSON.stringify({
      candidates: [{
        content: "some unrelated convention zz-unique-bench-marker-xqzt",
        kind: "convention",
        scope: "general",
        certainty: "definitive",
        tags: ["unique-tag-zz", "bench-marker"],  // zero overlap with lmp tags
        rationale: "test fixture — deliberately unrelated",
      }],
    });

    const testLlm = createMultiPatternMockLLM([
      // Extraction: returns the unrelated convention when the unique phrase appears
      { match: UNRELATED_PHRASE, response: unrelatedResponse },
      // Semantic match: always "NONE" — the unique phrase doesn't trigger "1"
      { match: "SAME underlying behavioral habit", response: "NONE" },
    ]);

    const result = await processMessage(
      `User: ${UNRELATED_PHRASE}\nAssistant: Got it.`,
      testLlm,
      "pil-chat:exchange",
      testLlm,
    );

    // Semantic match returned "NONE" → no grouping with lmp → new artifact
    expect(result.created).toHaveLength(1);
    expect(result.updated).toHaveLength(0);

    // Store now has TWO artifacts: lmp + the unrelated one
    const all = await loadAll();
    const active = all.filter((a) => !a.retired);
    expect(active).toHaveLength(2);
  });
});
