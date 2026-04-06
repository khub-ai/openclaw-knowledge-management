# KHUB Knowledge Fabric

**A knowledge store that learns from your conversations, persists across sessions and agents, and stays on your machine Ã¢â‚¬â€ inspectable and portable by design.**

**[Knowledge Fabric (KF)](docs/what-is-kf.md)** is the broader runtime knowledge layer in this repository. It wraps one or more LLMs, VLMs, or multimodal models and gives the overall system capabilities the base models do not natively provide on their own: runtime learning, persistent reusable knowledge, explicit revision, and portable human-readable artifacts.

The core learning pattern inside KF is **[PIL (Persistable Interactive Learning)](docs/glossary.md#pil-persistable-interactive-learning)**. PIL is the mechanism by which the system learns from interaction, distils what it learns into persistent [knowledge artifacts](docs/glossary.md#knowledge-artifact), and makes those artifacts available across sessions, tasks, and agents.

## What KF Is In 60 Seconds

- **KF is not a standalone model.** It is a runtime knowledge layer that works with one or more underlying models.
- **KF learns outside the model.** Useful knowledge is captured as persistent artifacts rather than buried in model weights.
- **KF improves systems incrementally.** A correction, rule, procedure, or judgment can change behavior immediately without fine-tuning.
- **KF keeps knowledge governable.** What the system has learned remains inspectable, editable, portable, and auditable.
- **KF goes beyond memory retrieval.** It is about capturing, structuring, applying, revising, and sometimes compiling knowledge into more reliable tools.

If you are new to the project, start with **[What Knowledge Fabric Is](docs/what-is-kf.md)**.

## Inductive learning: a new way to improve AI systems

Knowledge Fabric is built on a simple but far-reaching idea: many of the improvements we want from AI do not require changing the model itself. They require a system that can **learn from interaction**, extract what matters, and keep that knowledge in a form it can reuse.

This is the role of **inductive learning** in KF. Instead of burying improvements inside model weights, KF draws out reusable knowledge from **LLMs, VLMs, and human experts**: rules, procedures, judgments, boundary conditions, and corrective insights. It then turns those into persistent artifacts that can be applied later to patch the weaknesses of the current model in use.

In that sense, KF offers a new path beyond traditional fine-tuning. It improves behavior at the **system layer** rather than the model layer.

That has important consequences:

- **Improvement becomes immediate** â€” a useful insight can change system behavior without retraining
- **Knowledge stays visible** â€” what was learned is inspectable, editable, and auditable
- **Knowledge becomes portable** â€” improvements can move across models, agents, and platforms
- **Expertise becomes cumulative** â€” human and model-derived insights can compound over time
- **Correction becomes governable** â€” bad rules can be revised, narrowed, or retired without touching the base model

The long-term implication is significant: instead of treating every model as a largely sealed intelligence that must be retrained to improve, KF treats intelligence as something that can be **continuously extended through explicit, reusable knowledge**.

That is why we see inductive learning as more than a memory feature. It is a candidate for a new layer in the AI stack: one that lets agentic systems become progressively more capable, more aligned to their domains, and more valuable over time without locking those gains inside a vendor model.

## Who this is for

**Agent developers** Ã¢â‚¬â€ if you are building an user-facing agent or AI assistant, PIL gives it a long-term memory that grows with each user interaction. The agent learns patterns, preferences, and workflows naturally from conversation Ã¢â‚¬â€ without requiring the user to say "remember this." Knowledge is stored as lightweight, inspectable artifacts on the user's machine, and applied in future sessions only when confidence warrants it. PIL integrates into any conversation loop, works with any LLM, and is completely platform-agnostic. A complete worked example is available in `apps/computer-assistant/`.

**OpenClaw users** Ã¢â‚¬â€ PIL is packaged as an OpenClaw extension, so your OpenClaw instance can learn the patterns of how you work and become progressively more efficient without you having to repeat yourself. It can also turn a procedure you perform repeatedly into an executable script Ã¢â‚¬â€ future runs become faster, cheaper, and fully reliable.

**Enterprise AI adopters** Ã¢â‚¬â€ deploying AI agents at organizational scale surfaces four problems that no single framing captures:

- **Scalability and cost**: Naive agent memory forces a choice between losing context across sessions or injecting growing conversation histories into every context window Ã¢â‚¬â€ a cost that scales linearly with accumulated history. Structured local artifacts break this: once a preference, convention, or procedure is consolidated, it is applied through an in-memory index lookup at zero LLM cost. The more the system knows, the cheaper each interaction becomes, not the reverse.

- **Knowledge continuity**: When an expert leaves, their judgment Ã¢â‚¬â€ which arguments hold up, which exceptions to flag, which edge cases to handle a particular way Ã¢â‚¬â€ leaves with them. PIL captures that judgment incrementally as active artifacts that a successor's agent inherits from day one, without requiring any explicit documentation effort from the departing employee.

- **Knowledge as an organizational asset**: Structured, versioned, provenance-bearing artifacts are organizational property, not configuration. They survive platform migrations (model-agnostic text, no vendor lock-in), enable coherent M&A knowledge reconciliation at the artifact level, and Ã¢â‚¬â€ as the format matures Ã¢â‚¬â€ can be certified, licensed, or traded as expert knowledge packages.

- **Governance**: Every artifact carries a structured provenance record from creation through retirement Ã¢â‚¬â€ who created it, from which conversation, who approved it for team or org use, when it was revised, and when it was superseded. For regulated industries, this answers the question auditors will eventually ask: *what did the agent know at the time of this recommendation, and who signed off on it?*

**AI ecosystem builders and strategists** Ã¢â‚¬â€ if you are tracking where durable value accumulates in the AI stack, PIL proposes a new asset class: portable, typed, user-owned knowledge artifacts. The artifact format, if it achieves adoption, defines a coordination layer Ã¢â‚¬â€ analogous to what npm did for packages or OpenAPI did for APIs Ã¢â‚¬â€ around which expert knowledge marketplaces, org custody services, and certification businesses can form.

The economic logic is structural, not speculative. Previous attempts at expert knowledge capture Ã¢â‚¬â€ expert systems in the 1980s, knowledge management platforms in the 2000s Ã¢â‚¬â€ failed because experts were asked to invest significant effort in exchange for nothing. PIL changes this in two stages. First and most importantly: the expert gets a better agent for their own work. An investment analyst who elicits their own judgment framework into an agent gets an agent that pre-screens opportunities using their own criteria and applies their standards without being retaught Ã¢â‚¬â€ immediate personal ROI, no marketplace required. Second: that same knowledge package can be distributed or sold, adding further upside once the ecosystem matures.

Think of it as the Excel moment for expert knowledge. Analysts did not build financial models in order to sell them Ã¢â‚¬â€ they built them because they used those models every day to do their own work better. The ability to distribute models was a bonus. PIL works the same way: experts author knowledge packages primarily because a better-trained agent makes them more productive, and distribution is the additional layer that a knowledge marketplace unlocks.

**AI platforms and potential partners** Ã¢â‚¬â€ PIL's default integration is with OpenClaw, but the artifact format and core library are platform-agnostic. A platform that supports PIL artifact import/export gains interoperability with an emerging knowledge ecosystem and a credible story for user data portability and governance.

Ã¢â€ â€™ *[Detailed enterprise and investment case](docs/enterprise-vision.md)* Ã‚Â· *[Security threat model](docs/security.md)*

## Developer documentation

| Document | What it covers |
|---|---|
| [docs/what-is-kf.md](docs/what-is-kf.md) | **Start here if you are new to the repo.** Quick explanation of what KF is, how it works with LLMs/VLMs, and how it differs from adjacent approaches |
| [docs/ensemble-pipeline.md](docs/ensemble-pipeline.md) | **Start here for development.** 4-round ensemble pipeline, core class APIs (RuleEngine, ToolRegistry, StateManager, GoalManager, call_agent), domain specializations (ARC-AGI-2, ARC-AGI-3, UC200 image classification), extension guide |
| [docs/architecture.md](docs/architecture.md) | Knowledge artifact schema, storage, tiered retrieval, and OpenClaw plugin integration |
| [docs/design-decisions.md](docs/design-decisions.md) | How KF differs from other agent memory systems (Letta, platform memory, fine-tuning) |
| [docs/glossary.md](docs/glossary.md) | Canonical definitions for all KF terms |
| [docs/roadmap.md](docs/roadmap.md) | Planned benchmarks and future use cases |

## Why this exists

If you use an AI agent daily you have likely found yourself re-explaining your preferences, correcting the same mistakes, or re-teaching your workflow every session. The knowledge the agent should have accumulated is simply not there Ã¢â‚¬â€ because most agents today do not have a durable, user-owned place to put it.

The deeper problem: the knowledge AI agents do accumulate Ã¢â‚¬â€ through context windows, compacted session memory, or fine-tuning Ã¢â‚¬â€ remains **server-side, opaque, and platform-locked**. The user cannot inspect, edit, govern, or move it. Switch platforms and it disappears entirely.

PIL takes a different approach: knowledge is extracted from interaction, stored locally as human-readable text, and owned entirely by the user. The artifacts are **model-agnostic** Ã¢â‚¬â€ any sufficiently capable LLM can consume them. They can be edited with a text editor, version-controlled with git, shared with colleagues, or imported into a different AI assistant.

Ã¢â€ â€™ *[How this differs from existing agent memory](docs/design-decisions.md#how-this-differs-from-existing-agent-memory)*

## What the agent learns

The project is built around **four types of knowledge**, with emphasis on their generalized forms:

| Memory type | What it captures | Generalized into |
|---|---|---|
| **Episodic** | What happened in conversation | Raw material for generalization of other types |
| **Semantic** | Facts, preferences, conventions | General rules applicable across contexts |
| **Procedural** | How to do things | Structured recipes, optionally executable programs |
| **Evaluative** | What counts as "good" | Judgment heuristics and value frameworks |

**Generalization is the key.** We don't just record that the user corrected a summary format five times Ã¢â‚¬â€ we distill it into a rule ("always use bullet points") that the agent applies proactively in new situations. For procedures, a structured recipe can optionally be compiled into a deterministic program for tasks that require perfect repeatability. For evaluative knowledge, patterns of preference become judgment frameworks the agent uses to make good choices in novel situations Ã¢â‚¬â€ the closest analogue to what we colloquially call "wisdom."

Ã¢â€ â€™ *[Full memory taxonomy and evaluative knowledge](docs/memory-taxonomy.md)*

## See it in action

A condensed example of how the agent learns progressively over multiple sessions:

> **Session 2** Ã¢â‚¬â€ Agent notices the user always names financial statements as `YYYY-MM-institution-account.pdf`. It proposes a convention. User confirms. Ã¢â€ â€™ *Semantic artifact created.*
>
> **Session 3** Ã¢â‚¬â€ Agent learns the user downloads from Chase, Fidelity, and Amex monthly. It proposes a checklist procedure. Ã¢â€ â€™ *Procedural artifact created.*
>
> **Session 4** Ã¢â‚¬â€ After three identical runs, the agent offers to compile the checklist into a script. Ã¢â€ â€™ *Recipe retained; executable program linked as optional optimised form.*
>
> **Session 5** Ã¢â‚¬â€ User closes an account and opens a new one. Agent revises the institution list, the procedure, and the script together. Ã¢â€ â€™ *Coherent revision across artifact types.*

Ã¢â€ â€™ *[Full example with dialogue](docs/example-learning-in-action.md)*

## Core idea: Persistable Interactive Learning (PIL)

The system treats dialogue as a learning substrate and produces durable knowledge artifacts through an 8-stage pipeline:

1. **Elicit** Ã¢â‚¬â€ surface candidate knowledge from interaction
2. **Induce** Ã¢â‚¬â€ classify into a typed artifact (semantic, procedural, evaluative, etc.)
3. **Validate** Ã¢â‚¬â€ estimate confidence and scope
4. **Compact** Ã¢â‚¬â€ compress into minimal form; deduplicate
5. **Persist** Ã¢â‚¬â€ store locally with provenance and versioning
6. **Retrieve** Ã¢â‚¬â€ recall by relevance and context
7. **Apply** Ã¢â‚¬â€ confidence-gated: suggest, auto-apply, or hold back
8. **Revise** Ã¢â‚¬â€ update or retire when contradicted or outdated

## Architecture at a glance

```mermaid
graph TB
    subgraph machine["Ã°Å¸â€“Â¥Ã¯Â¸Â  Your Machine"]
        subgraph proc["Agent Process"]
            A["Your Agent"] -->|"processMessage Ã‚Â· retrieve Ã‚Â· apply Ã‚Â· revise"| P["PIL Library  (embedded Ã¢â‚¬â€ no daemon needed)"]
        end
        subgraph storage["~/.openclaw/knowledge/  Ã¢â‚¬â€  plain JSON, user-owned"]
            J["artifacts.jsonl Ã¢â‚¬â€ preferences, rules, procedures, judgments"]
            S["sessions/ Ã¢â‚¬â€ dialogic learning sessions"]
            C["communication-profile.json Ã¢â‚¬â€ dialogue preferences"]
        end
        P -->|"read / write"| J & S & C
    end

    subgraph cloud["Ã¢ËœÂÃ¯Â¸Â  LLM Provider"]
        L["Anthropic Ã‚Â· OpenAI Ã‚Â· Google Ã‚Â· open-source Ã‚Â· local"]
    end

    P -->|"extract Ã‚Â· consolidate Ã¢â‚¬â€ nothing stored remotely"| L
    L -->|"structured response"| P
```

PIL runs entirely inside your agent process Ã¢â‚¬â€ there is no server, no daemon, and no extra infrastructure to start. Knowledge files are plain JSON on your machine, editable with any text editor. The LLM is called only for processing (extraction, consolidation, response generation) and never stores knowledge on your behalf. Any LLM provider, or a local model, can be substituted by replacing a single adapter function.

## Design goals

- **User-owned and local** Ã¢â‚¬â€ knowledge lives on your machine, not a vendor's server
- **Model-agnostic portability** Ã¢â‚¬â€ artifacts are text; any capable LLM can consume them
- **Platform-agnostic** Ã¢â‚¬â€ works standalone or as an extension; default integration with OpenClaw
- **Confidence-gated reuse** Ã¢â‚¬â€ learned knowledge is *suggested*, *auto-applied*, or *held back* based on certainty
- **Free-form artifacts** Ã¢â‚¬â€ lightweight conventions, not rigid schemas; human-readable and editable
- **Versioned and auditable** Ã¢â‚¬â€ changes are tracked; revisions are first-class

Ã¢â€ â€™ *[Design decisions and rationale](docs/design-decisions.md)* Ã‚Â· *[Architecture (tiered triggering, knowledge graph, artifact schema)](docs/architecture.md)* Ã‚Â· *[Security threat model](docs/security.md)* Ã‚Â· *[FAQ](docs/faq.md)*

## Roadmap

### Near-term (Phase 1 milestones)

| Milestone | What it delivers | Status |
|---|---|---|
| **1a Ã¢â‚¬â€ Scaffolding** | Pipeline architecture with placeholder heuristics, playground | Ã¢Å“â€¦ Done |
| **1b Ã¢â‚¬â€ Explicit "remember"** | User says "remember this" Ã¢â€ â€™ LLM-backed artifact creation Ã¢â€ â€™ retrieval in future sessions | Ã¢Å“â€¦ Done |
| **1c Ã¢â‚¬â€ Passive elicitation** | Agent observes conversation via hooks and proposes knowledge without explicit instruction | Ã¢Å“â€¦ Done |
| **1d Ã¢â‚¬â€ Tier 1 triggering** | Keyword index enables zero-cost retrieval on every message | Ã¢Å“â€¦ Done |

### Long-term (Phases 2Ã¢â‚¬â€œ6)

| Phase | Focus | Status |
|---|---|---|
| **[2 Ã¢â‚¬â€ Generalization Engine](docs/roadmap.md#phase-2--generalization-engine)** | Episodic Ã¢â€ â€™ semantic/evaluative generalization, Tier 2 triggering, decay, feedback | Planned |
| **[3 Ã¢â‚¬â€ Procedural Memory & Code Synthesis](docs/roadmap.md#phase-3--procedural-memory-and-code-synthesis)** | Structured recipes, optional program compilation, tool library | Planned |
| **[4 Ã¢â‚¬â€ Expert-to-Agent Dialogic Learning](docs/roadmap.md#phase-4--expert-to-agent-dialogic-learning)** | Active expert elicitation through structured dialogue; produces procedures, judgments, boundary conditions, and revision triggers Ã¢â‚¬â€ see [spec](specs/expert-to-agent-dialogic-learning.md) Ã‚Â· [demo](docs/demo-dialogic-learning.md) | Ã¢Å“â€¦ Done |
| **[5 Ã¢â‚¬â€ Portability](docs/roadmap.md#phase-5--portability-and-cross-agent-compatibility)** | Standard artifact format, import/export, cross-agent compatibility | Planned |
| **[6 Ã¢â‚¬â€ Governance & Ecosystem](docs/roadmap.md#phase-6--governance-and-ecosystem-long-term)** | Team/org knowledge tiers, access controls, compliance audit trails, knowledge ecosystem and new business models | Long-term |

Ã¢â€ â€™ *[Detailed roadmap with milestones](docs/roadmap.md)* Ã‚Â· *[Enterprise vision and investment thesis](docs/enterprise-vision.md)*

## Repository structure

```
khub-knowledge-fabric/
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ packages/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ knowledge-fabric/       # Core PIL library (optional OpenClaw plugin)
Ã¢â€â€š   Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ index.ts            # Plugin entry point (when used with OpenClaw)
Ã¢â€â€š   Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ openclaw.plugin.json
Ã¢â€â€š   Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ src/
Ã¢â€â€š   Ã¢â€â€š       Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ pipeline.ts     # Stages 1Ã¢â‚¬â€œ4: elicit, induce, validate, compact
Ã¢â€â€š   Ã¢â€â€š       Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ store.ts        # Stages 5Ã¢â‚¬â€œ8: persist, retrieve, apply, revise
Ã¢â€â€š   Ã¢â€â€š       Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ tools.ts        # knowledge_search tool via OpenClaw plugin SDK
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ skills-foo/             # Example PIL-aware skill
Ã¢â€â€š       Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ SKILL.md
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ apps/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ computer-assistant/     # PIL-powered REPL demo (learns from real interaction)
Ã¢â€â€š   Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ src/
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ playground/             # Dev harness Ã¢â‚¬â€ runs the pipeline without OpenClaw
Ã¢â€â€š       Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ index.ts
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ tools/
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ pil-chat/               # Interactive CLI chatbot for testing PIL
Ã¢â€â€š       Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ index.ts
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ specs/                      # Design and mechanism specifications
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ expert-to-agent-dialogic-learning.md            # Spec for learning from experts via dialogue
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ expert-to-agent-dialogic-learning-example-investing.md  # Worked example (investing domain)
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ learnable-procedural-primitives-runtime.md      # Spec for LLM-centered procedural learning runtime
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ docs/                       # Design documents
    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ memory-taxonomy.md      # Four memory types, generalization, cognitive mechanisms
    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ architecture.md         # Tiered triggering, knowledge graph, artifact schema
    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ example-learning-in-action.md
    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ roadmap.md              # Phased roadmap with near-term milestones
    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ design-decisions.md     # Rationale, differentiators, forward-compatibility
    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ security.md             # Threat model, risks, and mitigations by phase
    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ enterprise-vision.md    # Scalability, institutional knowledge, tradeable artifacts, governance, investment thesis
    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ faq.md                  # Frequently asked questions
    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ glossary.md             # Definitions for all key terms used across the project
    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ dialogic-learning-positioning.md  # Landscape comparison: expert-to-agent dialogic learning
    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ positioning-doc.md      # AI-assisted strategic positioning analysis
    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ openclaw-plugin-setup.md # Installing and running PIL inside OpenClaw
    Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ benchmarks/             # Annotated walkthroughs of runnable test programs
```

## Getting started

**Prerequisites:** Node.js Ã¢â€°Â¥ 18, pnpm, and an Anthropic API key.

The reference implementation uses [Anthropic Claude](https://console.anthropic.com/) as the LLM backend. Any LLM can be substituted by providing a different `LLMFn` adapter Ã¢â‚¬â€ see [`apps/playground/index.ts`](apps/playground/index.ts) for the adapter pattern.

```bash
git clone https://github.com/khub-ai/khub-knowledge-fabric
cd khub-knowledge-fabric
pnpm install

# Set your Anthropic API key (obtain from https://console.anthropic.com/)
export ANTHROPIC_API_KEY=sk-ant-...        # macOS / Linux / WSL
# $env:ANTHROPIC_API_KEY="sk-ant-..."     # Windows PowerShell

pnpm start        # runs all 8 PIL stages against sample input
pnpm dev          # re-runs on file changes (watch mode)
pnpm test         # runs the full test suite (no API key required)
pnpm chat         # interactive PIL chatbot (see tools/pil-chat/)
pnpm chat -- --fresh   # start with a clean store

# Teach the agent something via structured dialogue (Phase 4):
# /teach <domain> "<what to learn>"
# Then answer the agent's follow-up questions until the rule is synthesised.
```

Artifacts are stored at `~/.openclaw/knowledge/artifacts.jsonl`.
Override with `KNOWLEDGE_STORE_PATH=/your/path pnpm start`.

> **Windows PowerShell users:** if `pnpm` fails with a script execution policy
> error (`npm.ps1 cannot be loaded ... not digitally signed`), run this once to
> allow locally-installed package manager scripts:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Alternatively, bypass pnpm entirely and run pil-chat directly with Node:
> ```powershell
> node --loader ts-node/esm/transpile-only --no-warnings tools/pil-chat/index.ts -- --fresh
> ```

To run the library as a plugin inside a live OpenClaw instance, see **[docs/openclaw-plugin-setup.md](docs/openclaw-plugin-setup.md)**.

## Implementation status

Phases 1 and 4 are implemented. The passive learning pipeline (1aÃ¢â‚¬â€œ1d) and the dialogic learning engine (Phase 4) are both functional.

### What's implemented

| Component | Module | Status |
|---|---|---|
| **Artifact schema** | `src/types.ts` | Full schema: kind, certainty, scope, stage, tags, evidence, relations, salience, lifecycle fields |
| **LLM extraction** | `src/extract.ts` | `extractFromMessage()` Ã¢â‚¬â€ language-agnostic, single LLM call per message |
| **Evidence consolidation** | `src/extract.ts` | `consolidateEvidence()` Ã¢â‚¬â€ distills N observations into a generalized rule |
| **Pipeline orchestration** | `src/pipeline.ts` | `processMessage()` Ã¢â‚¬â€ extract Ã¢â€ â€™ match Ã¢â€ â€™ accumulate Ã¢â€ â€™ inject decision |
| **Three-stage model** | `src/store.ts` | candidate Ã¢â€ â€™ accumulating Ã¢â€ â€™ consolidated; auto-promotes at threshold (default: 3) |
| **Tag-based retrieval** | `src/store.ts` | `retrieve()` Ã¢â‚¬â€ tag overlap scoring (Tier 1) + content fallback |
| **Inject label logic** | `src/store.ts` | `[established]` / `[suggestion]` / `[provisional]` gating |
| **Feedback tracking** | `src/store.ts` | `recordAccepted()` / `recordRejected()` Ã¢â‚¬â€ nudge confidence from user signals |
| **Plugin wiring** | `index.ts` / `src/hooks.ts` / `tools.ts` | `knowledge_search` tool registered; `message_received` + `before_prompt_build` hooks implemented (Milestones 1c/1d) |
| **Computer-assistant demo** | `apps/computer-assistant/` | REPL, Anthropic LLM adapter, OS actions, PIL-aware agent |
| **Test suite** | `apps/computer-assistant/src/tests/` | 112 tests covering extraction, store, pipeline, scenarios, and benchmarks |
| **Benchmark suite** | `apps/computer-assistant/benchmarks/` | Extraction precision/recall/F1; retrieval hit rate; 18+ scenarios |
| **Session management** | `src/session.ts` | `createSession`, `loadSession`, `saveSession`, `listSessionsByDomain`, `promoteSession` (idempotent) |
| **Dialogic engine** | `src/dialogue.ts` | Gap-driven question ladder, LLM synthesis, correction parsing, `processTurn` entrypoint |
| **pil-chat teach mode** | `tools/pil-chat/index.ts` | `/teach <domain> "<objective>"`, gap-status bar, `/endteach`; see [demo walkthrough](docs/demo-dialogic-learning.md) |

### Phases 1 and 4 complete Ã¢Å“â€¦

**Phase 1** Ã¢â‚¬â€ passive learning: knowledge is extracted from every inbound message, accumulated across interactions, consolidated into generalized rules, and injected into future prompts automatically via the `before_prompt_build` hook.

**Phase 4** Ã¢â‚¬â€ dialogic learning: the expert runs `/teach <domain> "<objective>"` in pil-chat. The agent asks five targeted follow-up questions (one per consolidation gap), proposes a synthesis when all gaps are closed, parses the expert's correction, and promotes the confirmed rule to `artifacts.jsonl` with session provenance. A subsequent session in the same domain retrieves and injects the rule automatically. See the [full demo walkthrough](docs/demo-dialogic-learning.md).

### Phases 2, 3, 5, 6 Ã¢â‚¬â€ planned

- True Tier-2 triggering: cheap LLM disambiguation of partial tag matches
- Decay: effective confidence decreases for unretrieved, unreinforced artifacts
- Semantic/vector retrieval
- Evaluative knowledge generalization (judgment heuristics, value frameworks)
- Procedural recipe compilation to executable programs ([Phase 3](docs/roadmap.md#phase-3--procedural-memory-and-code-synthesis))
- Import/export and cross-platform portability ([Phase 5](docs/roadmap.md#phase-5--portability-and-cross-agent-compatibility))
- CLI for inspecting, editing, and deleting artifacts

## Non-goals

- Not fine-tuning base LLM weights
- Not a replacement for any specific AI platform Ã¢â‚¬â€ this is a portable knowledge layer
- Not treating "memory" as a single bucket Ã¢â‚¬â€ knowledge types differ and require different controls
- Not prescribing a rigid schema Ã¢â‚¬â€ artifacts are free-form text with conventions

## Status

Phases 1 and 4 implemented. Passive learning (Milestones 1aÃ¢â‚¬â€œ1d) extracts knowledge from user messages, accumulates evidence across interactions, and injects consolidated rules into future prompts automatically. Phase 4 adds active expert elicitation via `/teach` in pil-chat Ã¢â‚¬â€ the agent asks targeted follow-up questions, synthesises a rule when all five consolidation gaps are closed, and promotes it to the knowledge store with full session provenance. 112 tests pass with no API key required.

## License

KHUB Knowledge Fabric is released under the **[PolyForm Noncommercial License 1.0.0](LICENSE)**.

- Noncommercial use is permitted, including personal use, research, education, government, and other noncommercial organizational use.
- The repo is **source-available**, not open-source.
- Commercial use requires a separate commercial license.

If you are evaluating KHUB Knowledge Fabric for a potential commercial deployment, keep that distinction in mind before redistributing, bundling, or offering it as part of a paid product or service.

## Contributing

Contributions are welcome Ã¢â‚¬â€ especially around:
- Knowledge artifact conventions and evaluation
- Retrieval and ranking methods for learned artifacts
- Evaluative knowledge representation
- Procedural knowledge and code synthesis
- Safe application policies and conflict resolution
- Tooling for inspection, export, and deletion

---

**Working thesis:** Agents become genuinely useful when they can **learn interactively**, **persist what they learn as user-owned artifacts**, and **reliably reuse that knowledge** Ã¢â‚¬â€ all without repeatedly re-prompting the user, retraining the model, or locking knowledge into a vendor's platform.
