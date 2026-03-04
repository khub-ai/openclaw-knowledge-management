# OpenClaw Knowledge Management

**OpenClaw Knowledge Management** is an additive extension layer for OpenClaw focused on **acquiring user knowledge through persistable interactive learning**—and then making that knowledge **reusable, revisable, and scalable** across sessions.

Instead of treating conversations as ephemeral, this project turns interaction into a **knowledge acquisition pipeline**: the agent can learn *what matters to the user*, *how the user reasons*, *what procedures the user follows*, and *what judgments/values the user endorses*—then persist those learnings as compact, structured artifacts that can be recalled and applied later with confidence gating.

## Why this exists

Modern LLM agents can solve tasks, but they often fail to **accumulate durable knowledge** from ongoing use. When context resets, hard-won insights disappear—or require repeated prompting. This repo explores a broader solution: **interactive learning that produces persistable knowledge**, so the system improves over time without fine-tuning the base model.

In other words, the goal is not just “memory,” but **knowledge management**:
- learn from interaction,
- compact and index what was learned,
- retrieve it when relevant,
- apply it safely,
- and revise it as the user or environment changes.

## What counts as “user knowledge”?

This project targets multiple knowledge types learned through interaction, including:

- **Generalized patterns** (reusable procedures, heuristics, checklists, workflows)
- **Preferences and style** (communication style, formatting, level of detail, decision criteria)
- **Domain conventions** (user’s private terminology, org-specific practices, implicit constraints)
- **Judgments and values** (how the user weighs tradeoffs, what “good” means in their context)
- **Task strategies** (repeatable approaches to common problems; “skills” learned from dialogue)
- **Operational facts** (stable facts the user wants the agent to remember, with appropriate controls)

Not all learnings are equal. Some should be ephemeral. Some should be stored only with explicit permission. Some should decay or require periodic re-validation. This repo is about building the mechanisms to handle those distinctions.

## Core idea: Persistable Interactive Learning (PIL)

The system treats dialogue as a learning substrate and produces durable knowledge artifacts.

A typical lifecycle:

1. **Elicit**: ask targeted questions or observe repeated behaviors to surface candidate knowledge  
2. **Induce**: convert raw interaction into a candidate representation (pattern, rule, rubric, preference model, etc.)  
3. **Validate**: sanity-check, ask for confirmation when needed, estimate confidence and scope of applicability  
4. **Compact**: compress into a minimal but useful artifact (schema + summary + examples + constraints)  
5. **Persist**: store with metadata (provenance, timestamps, confidence, versioning, privacy controls)  
6. **Retrieve**: recall by relevance and context, not just keyword matching  
7. **Apply**: use confidence-gated application (suggest, partially apply, or automatically apply based on risk)  
8. **Revise**: update or retire artifacts when contradicted, outdated, or superseded

This pipeline is designed to improve **scalability** (less repeated prompting, smaller required context) and **personalization** (learned behavior without model fine-tuning).

## Design goals

- **Additive layer**: users can install/upgrade official OpenClaw normally; this layer adds capabilities without forking upstream.
- **Confidence-gated reuse**: learned knowledge can be *suggested*, *auto-applied*, or *held back* depending on risk and certainty.
- **Versioned knowledge artifacts**: changes are tracked; revisions are first-class.
- **Minimal context dependency**: patterns can be applied with tiny prompts when confidence is high.
- **Practical safety controls**: explicit user consent where appropriate; scoped applicability; easy inspection and deletion.
- **Portable representations**: knowledge artifacts should be exportable and auditable (e.g., JSON/YAML + human-readable summaries).

## Repository structure

```
openclaw-knowledge-management/
├── packages/
│   ├── openclaw-plus/          # Core PIL extension (OpenClaw plugin)
│   │   ├── index.ts            # Plugin entry point; registers tools with OpenClaw
│   │   ├── openclaw.plugin.json
│   │   └── src/
│   │       ├── pipeline.ts     # Stages 1–4: elicit, induce, validate, compact
│   │       ├── store.ts        # Stages 5–8: persist, retrieve, apply, revise
│   │       └── tools.ts        # knowledge_search tool registered via plugin SDK
│   └── skills-foo/             # Example skill demonstrating PIL-aware patterns
│       └── SKILL.md
└── apps/
    └── playground/             # Dev harness — runs the full pipeline without OpenClaw
        └── index.ts
```

## Getting started (development)

Requires Node.js ≥ 18 and pnpm.

```bash
git clone https://github.com/khub-ai/openclaw-knowledge-management
cd openclaw-knowledge-management
pnpm install
cd apps/playground
pnpm start        # runs all 8 PIL stages against sample input
pnpm dev          # re-runs on file changes
```

Artifacts are stored at `~/.openclaw/knowledge/artifacts.jsonl`.
Override with `KNOWLEDGE_STORE_PATH=/your/path pnpm start`.

## Implementation status

### Done
| Stage | Module | Description |
|---|---|---|
| 1 Elicit | `pipeline.ts` | Signal-word heuristics extract candidate sentences from raw input |
| 2 Induce | `pipeline.ts` | Regex rules classify each candidate into a `KnowledgeKind` with baseline confidence |
| 3 Validate | `pipeline.ts` | Adjusts confidence up/down based on assertive vs. hedging language and length |
| 4 Compact | `pipeline.ts` | Normalises whitespace; deduplication runs in Persist |
| 5 Persist | `store.ts` | JSONL-backed upsert; near-duplicate detection via Jaccard similarity (≥0.75 threshold) |
| 6 Retrieve | `store.ts` | Keyword search scored by Jaccard + confidence; retired artifacts excluded |
| 7 Apply | `store.ts` | confidence ≥ 0.8 → auto-apply; below → suggest; both labelled in output |
| 8 Revise | `store.ts` | Minor edits update in place; significant content change retires old entry, preserving audit trail |
| Plugin wiring | `index.ts` / `tools.ts` | `knowledge_search` tool registered with OpenClaw via plugin SDK |

### Planned
- LLM-backed induction (replace regex heuristics with a structured extraction call)
- Confidence calibration from user feedback signals
- Vector/semantic retrieval (replacing or augmenting keyword Jaccard)
- Consent and privacy controls per artifact kind
- Decay / re-validation scheduling for time-sensitive facts
- CLI commands for inspecting, editing, and deleting stored artifacts
- Integration tests against a live OpenClaw workspace

## Relationship to “Saving Learned Generalized Patterns”

The thread “Saving Learned Generalized Patterns” motivates one key slice of this work: learning reusable patterns from dialogue and persisting them as compact artifacts that can later be recalled and applied. This repo generalizes that concept into a broader **knowledge acquisition and management** framework spanning multiple knowledge types and governance controls.

## Non-goals (for clarity)

- Not trying to fine-tune base LLM weights.
- Not trying to replace OpenClaw’s core; this is an extension layer.
- Not treating “memory” as a single bucket—knowledge types differ and require different controls.

## Status

Early-stage / experimental. Core PIL pipeline (stages 1–8) is implemented and runnable via the playground. OpenClaw plugin wiring is in place. The current heuristic implementations are intentionally simple — designed to be replaced with LLM-backed logic incrementally. Expect schema evolution and rapid iteration.

## Contributing

Contributions are welcome—especially around:
- knowledge schemas and evaluation
- retrieval/ranking methods for learned artifacts
- safe application policies and conflict resolution
- tooling for inspection, export, and deletion

---

**Working thesis:** Agents become genuinely useful when they can **learn interactively**, **store what they learn**, and **reliably reuse it**—all without repeatedly re-prompting the user or retraining the model.

