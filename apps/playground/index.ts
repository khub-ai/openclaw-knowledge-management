/**
 * Playground — manual harness for exercising the full PIL pipeline
 * without needing a live OpenClaw runtime.
 *
 * Run:   pnpm start          (single run)
 *        pnpm dev            (re-run on file changes)
 *
 * Storage: ~/.openclaw/knowledge/artifacts.jsonl
 *          (override via KNOWLEDGE_STORE_PATH env var)
 */

import { elicit, induce, validate, compact } from "@khub-ai/openclaw-plus/pipeline";
import { persist, retrieve, apply, revise } from "@khub-ai/openclaw-plus/store";

const SAMPLE_INPUT = [
  "I always want bullet-point summaries, no more than five points.",
  "Never use filler phrases like 'in conclusion' or 'to summarise'.",
  "When writing code, I prefer TypeScript with strict mode enabled.",
  "Use async/await rather than raw promises wherever possible.",
].join(" ");

const hr = () => console.log("-".repeat(60));

console.log("=== PIL Playground ===");
console.log(`Input: "${SAMPLE_INPUT}"\n`);

// ── Stage 1: Elicit ───────────────────────────────────────────
hr();
const candidates = elicit(SAMPLE_INPUT);
console.log(`1. Elicit  → ${candidates.length} candidate(s):`);
candidates.forEach((c, i) => console.log(`   [${i}] ${c}`));

// ── Stages 2–4: Induce → Validate → Compact (per candidate) ──
hr();
console.log("2–4. Induce → Validate → Compact\n");

const artifacts = candidates
  .map((c) => induce(c, "playground/sample"))
  .filter(Boolean)
  .map((a) => validate(a!))
  .map((a) => compact(a));

artifacts.forEach((a, i) => {
  console.log(`   [${i}] kind=${a.kind}  confidence=${a.confidence}`);
  console.log(`        "${a.content}"`);
});

// ── Stage 5: Persist ─────────────────────────────────────────
hr();
for (const artifact of artifacts) {
  await persist(artifact);
}
console.log(`5. Persist → ${artifacts.length} artifact(s) written to store`);

// ── Stage 6: Retrieve ─────────────────────────────────────────
hr();
const query = "summary bullet points preference";
const results = await retrieve(query);
console.log(`6. Retrieve → query: "${query}"  →  ${results.length} match(es):`);
results.forEach((r) => {
  console.log(`   [${r.kind} / conf=${r.confidence}] "${r.content}"`);
});

// ── Stage 7: Apply ────────────────────────────────────────────
hr();
console.log("7. Apply:");
for (const result of results.slice(0, 2)) {
  const { suggestion, autoApply } = await apply(result, "summarise this document for me");
  console.log(`   autoApply=${autoApply}  →  ${suggestion}`);
}

// ── Stage 8: Revise ───────────────────────────────────────────
hr();
if (artifacts[0]) {
  const revised = await revise(artifacts[0], {
    content: "I always want concise bullet-point summaries, strictly no more than five points.",
    confidence: 0.95,
  });
  console.log(`8. Revise → id=${revised.id}  confidence=${revised.confidence}`);
  console.log(`   "${revised.content}"`);
}

hr();
console.log("Done.");
