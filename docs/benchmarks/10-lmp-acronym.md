| | |
|---|---|
| **Scenario** | Personal acronym shorthand learning |
| **What it tests** | Exchange-level extraction · Cross-session persistence · Semantic pattern matching (Option A) · Consolidation into general rule |
| **Automated test** | `apps/computer-assistant/src/tests/bench-lmp-acronym.test.ts` |
| **Run with** | `pnpm test` |

## Scenario

A user teaches the chatbot a personal shorthand (`lmp` = "list my preferences") through an organic conversational exchange rather than an explicit statement. The system must:

1. **Learn from the exchange** — understand that "lmp" is defined in the clarification turn, not as an isolated statement
2. **Persist across sessions** — retain the mapping so the next session recognizes "lmp"
3. **Generalize via semantic matching** — when the user introduces a second acronym (`atp` = "add to preferences"), group it with the first rather than creating an independent artifact
4. **Consolidate** — after three acronym exchanges, distill all observations into a general rule

## Why this is hard

### Problem 1 — Co-reference (pronoun resolution)

The user's clarifying message `"it means 'list my preferences'"` contains no explicit subject. Processed in isolation, PIL receives a dangling pronoun and cannot form a complete fact:

```
❌ PIL sees: "it means 'list my preferences'"   → extracts nothing (what is "it"?)
✅ PIL sees: full exchange including "lmp"       → extracts convention: lmp = list my preferences
```

**Fix:** The **exchange pass** (added in pil-chat after each response) processes the full User+Assistant turn together, giving PIL the context it needs to resolve the pronoun.

### Problem 2 — Tag variance (semantic drift)

When PIL learns "lmp" in one exchange and "atp" in another, the extraction LLM may assign slightly different tags to each:

```
lmp → tags: ["lmp", "shorthand-command", "acronym"]
atp → tags: ["atp", "command-alias"]               ← zero overlap!
```

Jaccard tag similarity = 0 → the pipeline creates a second independent artifact instead of accumulating evidence onto the first. Three acronyms create three orphan artifacts; consolidation never fires.

**Fix:** **Semantic matching (Option A)** — when Jaccard finds no match, `matchCandidate` falls back to an LLM call asking:

> *"Are these two observations instances of the SAME underlying behavioral habit — i.e., would combining their evidence yield a useful generalization?"*

For "lmp = list my preferences" and "atp = add to preferences" the answer is YES — both are the same habit (user defines acronyms for commands), so the new observation is grouped with the existing artifact.

A cheaper/faster model can be used for this classification (configure with `--match-model`), since the response is always a single number or "NONE".

### Problem 3 — Consolidation threshold

A single observation of "user uses an acronym" is not enough to generalize. Only after `CONSOLIDATION_THRESHOLD` (default: 3) observations on the same artifact does PIL consolidate them into a reusable rule via an LLM call.

## Full conversation trace

### Session 1

| Turn | Who | Text | PIL action |
|------|-----|------|-----------|
| 1 | User | `lmp` | Pre-response pass: no candidates (opaque string) |
| 1 | Assistant | *"I'm not sure what 'lmp' means. Could you clarify?"* | — |
| 1 | User | *"it means 'list my preferences'"* | — |
| 1 | (exchange pass) | Full turn processed together | Extracts `convention: 'lmp' = 'list my preferences'` → stored as `[provisional]` candidate |

### Session 2

| Turn | Who | Text | PIL action |
|------|-----|------|-----------|
| 1 | User | `lmp` | `retrieve("lmp")` finds the artifact → injected as `[provisional]` into system prompt |
| 1 | Assistant | *"Your current preferences are: bullet points, TypeScript strict mode..."* | — |

### Later — introducing more acronyms (semantic matching active)

| Exchange | Jaccard result | Semantic match result | Store change |
|----------|---------------|----------------------|--------------|
| `atp` = "add to preferences" | 0 (no tag overlap) | LLM: "1" (same pattern) | evidenceCount → 2, stage: `accumulating` |
| `ds` = "debug session" | 0 (no tag overlap) | LLM: "1" (same pattern) | evidenceCount → 3 → **consolidation fires** |

**Consolidated artifact** (after 3 observations):
```
"User regularly defines personal shorthand acronyms for frequently-used commands
 (e.g., lmp = list preferences, atp = add to preferences, ds = debug session)."
```

Stage: `consolidated` · Injectable as `[established]` once confidence ≥ 0.80

## Test phases

The automated test (`bench-lmp-acronym.test.ts`) verifies each step:

| Phase | What is tested |
|-------|---------------|
| **Phase 1** — Exchange extraction | Exchange pass extracts `lmp = list my preferences`; artifact is `[provisional]` injectable |
| **Phase 2** — Cross-session recall | `retrieve("lmp")` finds the artifact in a simulated next session |
| **Phase 3a** — Jaccard baseline | Without semantic matching, 3 acronyms → 3 orphan artifacts (no consolidation) |
| **Phase 3b** — Semantic matching | With semantic matching, 3 acronyms → 1 artifact → consolidation → general rule |
| **Prompt sanity** | Verifies the mock correctly identifies semantic match prompts; regression guard against prompt wording changes |

## Configuration

```bash
# Basic (semantic matching uses same model as chat)
pnpm chat -- --verbose --log /tmp/session.log

# With a cheap model for semantic matching
pnpm chat -- --model claude-sonnet-4-6 --match-model claude-3-5-haiku-20241022 --verbose
```

The semantic match call expects a response of just `1`, `2`, …, or `NONE` — a very cheap classification. A small/fast model works well and keeps costs low even in active sessions.
