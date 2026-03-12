# Phase 2 Close-Inspection Example: Investment Workflow

## Purpose

This companion example is more domain-rich than the summary-format example. It shows how the current Phase 2 implementation can accumulate and consolidate a reusable workflow in a domain that involves judgment, sequencing, and boundaries.

The domain here is investment analysis, but the point is not to teach the system financial truth or investment advice. The point is to show how the current implementation can learn a repeatable analysis procedure from the user's repeated instructions and then apply that procedure later.

That distinction matters:

- Phase 2 can learn a reusable workflow that the user endorses
- Phase 2 does not yet independently validate whether the workflow is correct in the world
- the stored artifact is therefore best understood as a learned operating procedure, not a proven market edge

## Why This Example

The first Phase 2 example focused on a formatting preference. That is useful for learning the mechanics, but it does not show how the system behaves when the learned artifact is more like a compact expert checklist.

This example does that. It stays faithful to the current code paths, but the learned artifact is now a `procedure` rather than a simple `preference`.

Developers reading this example should keep these files open:

- [extract.ts](C:/_backup/openclaw/khub-knowledge-fabric/packages/knowledge-fabric/src/extract.ts)
- [store.ts](C:/_backup/openclaw/khub-knowledge-fabric/packages/knowledge-fabric/src/store.ts)
- [pipeline.ts](C:/_backup/openclaw/khub-knowledge-fabric/packages/knowledge-fabric/src/pipeline.ts)
- [types.ts](C:/_backup/openclaw/khub-knowledge-fabric/packages/knowledge-fabric/src/types.ts)

The same Phase 2 constants still matter here:

- `CONSOLIDATION_THRESHOLD = 3`
- definitive confidence seed: `0.65`
- consolidation confidence bump: `+0.20`, capped at `0.92`
- default auto-apply threshold with no salience: `0.80`

## Scenario

The user wants the system to help review small public software companies. Over several interactions, the user repeatedly explains the same evaluation sequence:

1. check whether the balance sheet can survive a weak year
2. check for customer concentration risk
3. check whether management allocates capital sensibly
4. only then look at valuation

This is exactly the sort of reusable workflow that Phase 2 can store as a `procedure` artifact.

## Walkthrough

### Turn 1: First Statement Of Method

User message:

```text
When I ask you to review a small software company, first check whether it can survive a weak year without raising capital.
```

`extractFromMessage()` may produce a candidate like:

```json
{
 "content": "When reviewing a small software company, first check whether the company can survive a weak year without raising capital.",
 "kind": "procedure",
 "scope": "general",
 "certainty": "definitive",
 "tags": ["investment-review", "software-companies", "balance-sheet", "survivability"],
 "rationale": "The user is stating a reusable first-step workflow for a recurring analysis task."
}
```

`candidateToArtifact()` then creates an initial artifact:

```json
{
 "id": "artifact_proc_1",
 "kind": "procedure",
 "content": "When reviewing a small software company, first check whether the company can survive a weak year without raising capital.",
 "confidence": 0.65,
 "stage": "candidate",
 "evidenceCount": 1
}
```

### Turn 2: More Of The Same Procedure

User message:

```text
After that, check whether one customer is too large a share of revenue. A cheap stock is not good enough if concentration risk can break the business.
```

A second `procedure` candidate is extracted. Because it shares kind and overlapping tags with the first artifact, `matchCandidate()` can treat it as evidence for the same workflow rather than an unrelated memory.

Representative state after accumulation:

```json
{
 "id": "artifact_proc_1",
 "kind": "procedure",
 "stage": "accumulating",
 "confidence": 0.65,
 "evidenceCount": 2
}
```

This is already more domain-rich than the first example, but the mechanics are unchanged. Phase 2 is still doing the same things:

- extracting a reusable artifact from natural language
- matching against prior evidence of the same kind
- accumulating support for one underlying generalized rule

### Turn 3: Full Workflow Emerges

User message:

```text
Then check whether management allocates capital sensibly. Only after those checks should we spend time on valuation.
```

A third matching candidate arrives. The artifact now reaches the consolidation threshold and `consolidateEvidence()` synthesizes a more compact workflow.

Representative consolidated artifact:

```json
{
 "id": "artifact_proc_1",
 "kind": "procedure",
 "content": "When reviewing a small software company, first check balance-sheet survivability, then customer concentration risk, then management capital allocation, and only then examine valuation.",
 "confidence": 0.85,
 "stage": "consolidated",
 "evidenceCount": 3
}
```

What changed here is important:

- the artifact is no longer just one sentence remembered from the user
- it is now a generalized sequence synthesized from repeated evidence
- the confidence has crossed the default threshold for stronger reuse

## Later Application

Later, the user says:

```text
Please help me review TinySoft Systems.
```

`retrieve()` now has a much better chance of surfacing the procedure because the query overlaps with investment-review language and the artifact now carries the advantages of higher confidence and consolidated stage.

Representative retrieval result:

```json
{
 "id": "artifact_proc_1",
 "content": "When reviewing a small software company, first check balance-sheet survivability, then customer concentration risk, then management capital allocation, and only then examine valuation.",
 "score": 0.84,
 "stage": "consolidated"
}
```

Representative apply output:

```json
{
 "injected": [
 "[established] When reviewing a small software company, first check balance-sheet survivability, then customer concentration risk, then management capital allocation, and only then examine valuation."
 ],
 "autoApply": true
}
```

## What This Example Shows Clearly

This example helps developers and users inspect an important boundary of the current system.

What Phase 2 can do today:

- retain a reusable domain workflow in artifact form
- generalize repeated user guidance into a compact procedural summary
- retrieve and apply that workflow later with confidence gating
- revise the workflow when the user changes it

What Phase 2 does not yet do by itself:

- prove that the workflow is objectively correct
- test the workflow against external investment outcomes
- distinguish a merely repeated instruction from a truly superior expert method

That boundary should be made explicit in any close inspection of the current implementation. The learning here is real, but it is still learning a user-endorsed procedure, not independently validated domain truth.

## What Developers Should Inspect

1. In [extract.ts](C:/_backup/openclaw/khub-knowledge-fabric/packages/knowledge-fabric/src/extract.ts), inspect whether the extraction prompt reliably recognizes procedural knowledge as distinct from facts and preferences.

2. In [store.ts](C:/_backup/openclaw/khub-knowledge-fabric/packages/knowledge-fabric/src/store.ts), inspect whether procedure artifacts match and consolidate in the same robust way as simpler preference artifacts.

3. In [pipeline.ts](C:/_backup/openclaw/khub-knowledge-fabric/packages/knowledge-fabric/src/pipeline.ts), inspect how the resulting procedure is surfaced later as an operational hint during a new task.

## What Users Should Expect

Users should expect that repeated explanation of a workflow can become a reusable operating procedure inside the system. They should not yet expect the system to independently certify the wisdom of that workflow. In the current Phase 2 design, the gain is continuity, reuse, and generalization of the user's method.

## Relation To The First Example

The first example shows Phase 2 learning a simple preference. This example shows the same implementation pattern operating on a richer artifact: a compact domain workflow. Together, the two examples show that the current Phase 2 design can already handle both low-complexity personal preferences and richer reusable procedures without any change to the core pipeline.
