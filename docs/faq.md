# Frequently Asked Questions

---

### Is a Python API available for KHUB-PIL?

Not yet. The current implementation is TypeScript-only. Here is what is available today and what is planned.

**What Python callers can do today:**

The artifact store is plain JSONL at `~/.openclaw/knowledge/artifacts.jsonl` — one JSON object per line. Python can read, filter, and write artifacts directly without going through the TypeScript pipeline. The schema is documented in plain tables at [`docs/architecture.md`](architecture.md#knowledge-artifact-schema) (the TypeScript type definitions are at `packages/knowledge-fabric/src/types.ts` for reference). This covers tooling use cases: artifact inspection, migration scripts, reporting, and external injection.

**What is planned:**

A native Python library — a port of the PIL pipeline to Python. PIL is a library, not a service: a Python agent should call the Python PIL library from within the same process, the same way the TypeScript agent calls the TypeScript library. A REST API would require running a TypeScript daemon alongside the Python agent, which contradicts the local-first, zero-extra-infrastructure design.

The Python port is architecturally straightforward: the LLM adapter is a plain callable `(prompt: str) -> str`, the store is plain JSON, and the pipeline has no framework dependencies. It is planned for after Phase 5 (Portability), when the artifact schema is stable enough that a Python implementation can commit to full compatibility.

---

### Does PIL require OpenClaw, or can I use it as a standalone library?

The core pipeline — `processMessage`, `retrieve`, `apply`, `revise` — is a plain TypeScript library with no hard dependency on OpenClaw. It can be instantiated in any Node.js environment by supplying an `LLMFn` adapter (a function that takes a prompt string and returns a promise of the LLM's response).

The [`apps/playground`](../apps/playground/index.ts) demonstrates this: it runs all eight PIL stages against sample input using an Anthropic SDK adapter, with no OpenClaw process involved.

OpenClaw integration adds:
- **Passive elicitation** — the `message_received` hook allows PIL to observe every conversation turn automatically, without requiring the user to issue an explicit "remember this" instruction.
- **Automatic injection** — the `before_prompt_build` hook injects relevant artifacts into every prompt without manual invocation.
- **Multi-channel reach** — knowledge learned in one channel (Slack, email, etc.) is available across all 24+ platforms OpenClaw supports.

For development, evaluation, and non-OpenClaw deployments, the core library is usable standalone.

---

### How is PIL different from RAG (Retrieval-Augmented Generation)?

RAG and PIL are complementary rather than competitive, but they operate on fundamentally different material:

| | RAG | PIL |
|---|---|---|
| **What is stored** | Raw source documents | Distilled, typed, confidence-scored knowledge artifacts |
| **What is retrieved** | Relevant document chunks | Relevant conclusions already extracted and generalized |
| **Re-derivation at retrieval** | Yes — LLM reads raw text and re-derives relevance each call | No — artifact content is already generalized; retrieval is a lookup |
| **Provenance** | Source document reference | Full lifecycle provenance from creation through revision |
| **User can edit** | Rarely (depends on pipeline) | Always — artifacts are plain text files |
| **Learns from interaction** | No — source documents must be manually added | Yes — artifacts emerge from conversation |

In short, RAG is a retrieval mechanism for existing document knowledge. PIL is a learning mechanism for interactively acquired behavioral knowledge. An enterprise deployment might reasonably use both: RAG over document repositories for factual lookup, PIL over interaction history for behavioral adaptation.

---

### What is Expert-to-Agent Dialogic Learning?

Expert-to-agent dialogic learning is a learning pattern in which an agent acquires deep, reusable knowledge from a domain expert through structured back-and-forth dialogue — not by passively recording conversation, but by asking purposeful questions that surface procedures, judgment rules, boundary conditions, and revision triggers.

The distinction from ordinary memory is important: the goal is not recall of what was said, but distillation of how the expert thinks. A domain expert knows not just facts but methods — which signals to prioritize, when a rule stops applying, what evidence should change a conclusion, and which past mistakes to avoid repeating. That tacit knowledge is hard to write down in advance but can be drawn out through a well-structured dialogue.

The agent plays an active role: proposing tentative generalizations for the expert to correct, asking for failure cases and exceptions, and testing whether a rule still holds in a difficult case. The result is a set of structured artifacts — procedures, judgment rules, boundary conditions, failure cases — that the agent can reuse in future interactions, not a transcript.

This pattern is most valuable when:
- the important knowledge is partly tacit and not captured in existing documents
- the user wants the agent to learn a *method*, not just remember facts
- the domain requires caveats and revision conditions rather than simple answers
- the knowledge should remain inspectable and editable after it is learned

→ *[Expert-to-Agent Dialogic Learning spec](../specs/expert-to-agent-dialogic-learning.md)* · *[Worked example: investing domain](../specs/expert-to-agent-dialogic-learning-example-investing.md)* · *[Landscape positioning](./dialogic-learning-positioning.md)*

---

### How is PIL different from fine-tuning?

Fine-tuning adjusts model weights on domain-specific training data and is appropriate for *capability expansion* — teaching a new vocabulary, a new code style, a specialized domain. For *behavioral adaptation to a specific user, team, or operational context*, fine-tuning is ill-matched: it is expensive, requires batch training cycles, cannot practically adapt to an individual, and has no mechanism for correcting a learned behavior without risking collateral damage to others.

PIL is a complementary mechanism that addresses behavioral adaptation where fine-tuning cannot:

- **Incremental** — every interaction is a potential learning event
- **Inspectable** — knowledge is stored as human-readable artifacts, not as opaque weight updates
- **Correctable in real time** — a wrong artifact can be revised or retired immediately
- **Individual-specific** — learning accumulates for one person's preferences, not a population average

→ *[Full analysis with analogy](design-decisions.md#pil-as-a-complement-to-fine-tuning)*

---

### Who owns the knowledge artifacts? Can a vendor access them?

The user owns them entirely. Artifacts are stored as plain files on the user's own machine (default: `~/.openclaw/knowledge/artifacts.jsonl`). There is no telemetry, no server-side component, and no network call associated with storage or retrieval. The only network calls the pipeline makes are to the LLM of your choice, for extraction and consolidation.

A vendor can access artifacts only if you explicitly export and share them. The local-first, file-based storage model is a deliberate design choice, not an implementation convenience.

---

### What LLMs does PIL work with?

Any sufficiently capable LLM can be used. The pipeline depends on a single injected LLM adapter — a function that accepts a prompt string and returns the model's text response — with no hard dependency on any specific SDK or provider. The caller provides the implementation. In the TypeScript reference implementation this adapter is typed as `LLMFn: (prompt: string) => Promise<string>`; a Python implementation would express the same contract as a callable `(prompt: str) -> str`.

In practice, leading models from Anthropic, OpenAI, Google, and open-source providers all work. The quality of extraction and consolidation scales with the model's reasoning capability — smaller or older models may produce lower-quality generalizations, but the pipeline degrades gracefully (lower-confidence candidates rather than failures).

The reference implementation in `apps/playground` uses Anthropic's Claude via the `@anthropic-ai/sdk`. Swapping to a different provider requires only replacing the `LLMFn` implementation.

---

### What happens to my artifacts if I switch LLM providers?

Nothing. Artifacts are model-agnostic text — they contain no embeddings, no token IDs, and no representations tied to any specific model or vendor. The same artifact file works with Claude, GPT, Gemini, Llama, or any other LLM capable of reading text.

This is a deliberate design constraint: knowledge accumulated under one LLM vendor must survive a vendor change without conversion or migration. Platform lock-in should not extend to accumulated knowledge.

---

### Can I inspect or edit what my agent has learned?

Yes. Artifacts are stored as plain JSON at `~/.openclaw/knowledge/artifacts.jsonl` — one artifact per line, human-readable, editable with any text editor. You can:

- Open the file and read every artifact your agent has accumulated
- Edit an artifact's content, confidence, or tags directly
- Delete individual lines to retire specific artifacts
- Copy the file to share artifacts with colleagues or move them to another machine
- Version-control the file with git for a full audit trail of changes

A dedicated CLI for browsing, editing, and retiring artifacts is on the near-term roadmap and will make these operations more ergonomic than manual file editing.

---

### Will retrieval get slower as the knowledge store grows?

Not significantly, by design. The retrieval system is tiered:

- **Tier 1 (no LLM cost)**: High-confidence artifacts are indexed by tags in an in-memory hash table. Lookup is a set-intersection operation — effectively O(1) relative to store size.
- **Tier 2 (minimal LLM cost)**: Only partial or ambiguous matches trigger a lightweight LLM reasoning call over artifact *summaries*, not full content.
- **Tier 3**: The primary model reasons over already-retrieved, already-injected artifacts.

As the knowledge base matures, an increasing proportion of interactions are handled at Tier 1. A larger, more mature knowledge base is therefore cheaper per interaction, not slower — the inverse of naive history-injection approaches.

The practical limit is Tier 2 disambiguation quality at very large stores (thousands of artifacts with overlapping tags). The full inverted-index implementation addressing this is targeted for Milestone 1d.

---

### Is PIL ready for production use?

The pipeline is functional and tested — extraction, accumulation, consolidation, retrieval, apply, and revise all work end-to-end, covered by 112 unit and scenario tests. All Phase 1 milestones (1a–1d) are complete: passive elicitation, evidence consolidation, and automatic retrieval injection all function. Specific caveats for production deployments:

- The artifact schema and API surface may evolve in breaking ways before Phase 5 (Portability).
- Expert-to-Agent Dialogic Learning (Phase 4) is fully specified but not yet implemented.
- Enterprise governance features (tiered stores, RBAC, audit trails) are Phase 6.
- There is no dedicated CLI for artifact management yet.
- File locking and concurrent-write safety are known gaps for multi-process deployments.

For production use cases, the current recommendation is to deploy the pipeline behind a controlled integration layer, monitor artifact quality, and expect to update as the API stabilizes through Phase 2. For evaluation, experimentation, and development use, the pipeline is ready to use now.

→ *[Current implementation status](../README.md#implementation-status)*
