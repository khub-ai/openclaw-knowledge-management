# Phase 2 Close-Inspection Example

## Purpose

This document gives developers a small but information-rich scenario they can trace through the current Phase 2 implementation, while also showing users what to expect from the system's learning behavior.

The example intentionally stays simple on the surface, but it exercises the main Phase 2 lifecycle:

- extraction from a user message
- candidate creation
- same-kind matching
- evidence accumulation
- consolidation into a generalized artifact
- later retrieval and application
- explicit revision when the preference changes

## Why This Example

The scenario is a recurring formatting preference: the user wants summaries in bullet points, kept short. This is easy to follow, but it still reveals most of the important inner workings without domain-specific complexity.

Developers reading this example should keep these files open:

- [extract.ts](C:/_backup/openclaw/khub-knowledge-fabric/packages/knowledge-fabric/src/extract.ts)
- [store.ts](C:/_backup/openclaw/khub-knowledge-fabric/packages/knowledge-fabric/src/store.ts)
- [pipeline.ts](C:/_backup/openclaw/khub-knowledge-fabric/packages/knowledge-fabric/src/pipeline.ts)
- [types.ts](C:/_backup/openclaw/khub-knowledge-fabric/packages/knowledge-fabric/src/types.ts)

Important current constants:

- `CONSOLIDATION_THRESHOLD = 3`
- definitive confidence seed: `0.65`
- tentative confidence seed: `0.35`
- uncertain confidence seed: `0.15`
- consolidation confidence bump: `+0.20`, capped at `0.92`
- default auto-apply threshold with no salience: `0.80`

## Walkthrough

### Turn 1: Initial Preference

User message:

```text
I always want summaries in bullet points.
```

`processMessage()` calls `extractFromMessage()`, which may produce a candidate like:

```json
{
 "content": "The user prefers summaries in bullet-point format.",
 "kind": "preference",
 "scope": "general",
 "certainty": "definitive",
 "tags": ["summary-format", "bullet-points", "formatting"],
 "rationale": "The user stated a general formatting preference using explicit language ('always')."
}
```

`candidateToArtifact()` turns that into an initial artifact:

```json
{
 "id": "artifact_1",
 "kind": "preference",
 "content": "The user prefers summaries in bullet-point format.",
 "confidence": 0.65,
 "stage": "candidate",
 "evidenceCount": 1
}
```

What this means:

- the system has noticed a reusable preference
- it is stored cautiously as a `candidate`
- one statement is not yet enough for strong automation

### Turn 2: Repeated Evidence

User message:

```text
Please use bullet points for all summaries going forward.
```

A new `preference` candidate is extracted. `matchCandidate()` compares only against artifacts of the same `kind`, first using tag Jaccard overlap and then semantic fallback if needed. Here, the tag overlap is already enough to match `artifact_1`.

Instead of creating a duplicate, `accumulateEvidence()` updates the same artifact:

```json
{
 "id": "artifact_1",
 "stage": "accumulating",
 "confidence": 0.65,
 "evidenceCount": 2
}
```

This is the first big Phase 2 idea: the system starts treating repeated statements as evidence for one underlying rule, rather than as disconnected memories.

Important implementation note: the current code marks an `accumulating` artifact as injectable with a `[suggestion]` label. That is the behavior in [store.ts](C:/_backup/openclaw/khub-knowledge-fabric/packages/knowledge-fabric/src/store.ts), even though some benchmark notes imply stricter gating. For close inspection, developers should trust the code path.

### Turn 3: Consolidation

User message:

```text
Always keep those bullet-point summaries short, ideally no more than 5 bullets.
```

The third candidate still matches the same artifact, so `accumulateEvidence()` runs again. Now the artifact reaches the consolidation threshold from [types.ts](C:/_backup/openclaw/khub-knowledge-fabric/packages/knowledge-fabric/src/types.ts): `3`. That causes `consolidateEvidence()` to run.

Representative consolidated artifact:

```json
{
 "id": "artifact_1",
 "kind": "preference",
 "content": "The user generally prefers summaries in concise bullet-point form, usually with no more than 5 bullets.",
 "confidence": 0.85,
 "stage": "consolidated",
 "evidenceCount": 3
}
```

Why this matters:

- confidence rises from `0.65` to `0.85` under the current `+0.20` consolidation bump
- the content becomes more general than any single utterance
- the artifact is now much closer to stable reusable knowledge

### Later Retrieval And Apply

Much later, the user asks:

```text
Please summarize this article.
```

`retrieve()` scores candidate artifacts using tag overlap, content overlap, effective confidence, a consolidated bonus, and recency. Because this artifact has strong summary-format tags, high confidence, and consolidated stage, it is likely to rank well.

Representative retrieval result:

```json
{
 "id": "artifact_1",
 "content": "The user generally prefers summaries in concise bullet-point form, usually with no more than 5 bullets.",
 "score": 0.87,
 "stage": "consolidated"
}
```

After retrieval, `apply()` can inject the artifact. Because the artifact is consolidated and above the default `0.80` threshold, `getInjectLabel()` returns `[established]`.

Representative apply output:

```json
{
 "injected": [
 "[established] The user generally prefers summaries in concise bullet-point form, usually with no more than 5 bullets."
 ],
 "autoApply": true
}
```

This is the user-visible payoff of the earlier accumulation process: the system can now shape output proactively instead of waiting for the user to restate the preference again.

### Optional Revision

If the user later says, `Actually, keep summaries to at most 4 bullets, not 5.`, that is no longer repeated evidence. It is a correction. An explicit `revise()` call can update the artifact in place if the change is similar enough, or retire the old artifact and create a successor if the change is large.

Representative revised artifact:

```json
{
 "id": "artifact_1",
 "content": "The user generally prefers summaries in concise bullet-point form, usually with no more than 4 bullets.",
 "stage": "consolidated"
}
```

## What Developers Should Inspect

1. In [extract.ts](C:/_backup/openclaw/khub-knowledge-fabric/packages/knowledge-fabric/src/extract.ts): the `ExtractionCandidate` shape, the extraction prompt, and how `candidateToArtifact()` seeds confidence and initializes evidence.

2. In [store.ts](C:/_backup/openclaw/khub-knowledge-fabric/packages/knowledge-fabric/src/store.ts): `matchCandidate()`, the tag-based Jaccard fast path, same-kind filtering, semantic fallback, and `accumulateEvidence()`.

3. Also in [store.ts](C:/_backup/openclaw/khub-knowledge-fabric/packages/knowledge-fabric/src/store.ts): when `consolidateEvidence()` is called, how confidence is bumped, and how `retrieve()`, `getInjectLabel()`, and `apply()` turn a learned artifact into operational behavior.

4. In [pipeline.ts](C:/_backup/openclaw/khub-knowledge-fabric/packages/knowledge-fabric/src/pipeline.ts): how `processMessage()` stitches extraction, matching, accumulation, and injection together into one lifecycle.

## What Users Should Expect

Users should expect a system that can notice reusable preferences from natural conversation, avoid overcommitting after a single mention, become more confident after repeated evidence, later apply what it has learned without being reminded every time, and still remain revisable when the user changes the rule.

This is a good mental model for the current Phase 2 implementation. It is not yet deep expert learning, but it is already more than plain chat memory: it is a concrete pipeline for gradually turning repeated conversational signals into reusable, confidence-gated knowledge artifacts.

## Why This Example Is Useful

For developers, this example is small enough to step through with logs, breakpoints, or unit tests. For users, it sets healthy expectations about how the system learns: gradually, with confidence signals, with stabilization through repetition, and with room for later correction.
