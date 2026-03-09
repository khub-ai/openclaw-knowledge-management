# Spec: Learnable Procedural Primitives Runtime

## Purpose

This spec defines an optional PIL extension for learning and reusing executable procedures as programs over composable primitives.

It is intentionally broader than ARC. ARC-v2 and ARC-v3 are optional adapters that validate the design, not the reason for the design.

## Core Rules

- Preserve PIL principles: local, inspectable, portable, artifact-based knowledge.
- Do not change default repo behavior when the runtime is not configured.
- Load ARC adapters only when explicitly requested.
- Let humans tune penalties, promotion decisions, and generalization pressure.
- Avoid benchmark-specific tricks in core abstractions.
- Treat the canonical model as language-independent. TypeScript, Python, or other language bindings are implementations of the model, not the model itself.

## Architecture

1. Core PIL handles persistence, provenance, retrieval, lifecycle, revision, confidence, and feedback.
2. Procedural Runtime handles generic program induction and deterministic execution over primitives.
3. Domain Adapters parse inputs, extract features, register primitives, generate candidates, and verify outputs.
4. Benchmark Harness evaluates adapters but does not define the runtime model.

## Canonical Definition Style

All definitions in this runtime should be specified as language-neutral records, enums, and lists.
Canonical definitions should describe field names, required versus optional status, semantic meaning, allowed values, and invariants.
Canonical definitions should not depend on TypeScript syntax, Python syntax, implementation-specific class hierarchies, or a specific serialization library.

The preferred long-term direction is:
1. maintain a language-neutral schema definition
2. generate or hand-maintain TypeScript bindings from that schema
3. generate or hand-maintain Python bindings from that schema

## Program Model

Programs use a benchmark-neutral intermediate representation built from typed primitive calls with named inputs, outputs, and ordered steps.
The representation must be plain-text serializable, executable without an LLM, inspectable, replayable, composable, portable across domains, and stable enough to persist in artifacts.

## Canonical Records

This section defines the minimum language-independent records required by the runtime.

### Program
Required fields: `language`, `version`, `inputs`, `output`, `steps`.

### Primitive Invocation
Required fields: `op`, `args`.
Optional fields: `bind`, `when`, `meta`.

### Primitive Specification
Required fields: `name`, `summary`, `input_shape`, `output_shape`, `deterministic`.
Optional fields: `constraints`, `cost_hint`, `tags`.

### Hypothesis
Required fields: `id`, `adapter`, `program`, `status`, `created_at`.
Optional fields: `task_ref`, `score`, `rationale`, `trace_ids`, `counterexample_ids`.

### Transform Rule
Required fields: `id`, `name`, `program`, `scope`.
Optional fields: `preconditions`, `success_history`, `derived_from`.

### Macro
Required fields: `id`, `name`, `program_fragment`, `input_shape`, `output_shape`, `promoted_at`.
Optional fields: `promotion_basis`, `source_hypothesis_ids`, `stability_score`.

### Execution Trace
Required fields: `id`, `program_id`, `adapter`, `started_at`, `result`.
Optional fields: `task_ref`, `step_results`, `score`, `error_summary`.

### Counterexample
Required fields: `id`, `target_id`, `adapter`, `observed_failure`, `created_at`.
Optional fields: `task_ref`, `expected_summary`, `actual_summary`.

### Penalty Profile
Required fields: `id`, `name`, `complexity_weight`, `instability_weight`, `overfit_weight`, `runtime_weight`.
Optional fields: `adapter`, `notes`, `author`.

### Penalty Decision
Required fields: `id`, `profile_id`, `decision_type`, `author`, `created_at`.
Optional fields: `target_id`, `delta_complexity`, `delta_instability`, `delta_overfit`, `delta_runtime`, `delta_total`, `rationale`.

### Reasoning Score
Required fields: `fit_score`, `complexity_penalty`, `instability_penalty`, `overfit_penalty`, `runtime_penalty`, `human_adjustment`, `total`.
Optional fields: `exact_match_count`, `partial_match_score`, `notes`.

### Task Reference
Required fields: `adapter`.
Optional fields: `task_id`, `split`, `variant`.

## Learnable Procedural Primitives

The base primitive set should stay small and generic. Expressivity should grow mainly through learned macros promoted from successful recurring subprograms.
Macro names must describe general behavior, for example `compress_repeated_motif`, not benchmark-specific task IDs.
Recommended primitive families are structural, selection, transformation, propagation, inference, and control.
Example primitive names within those families: `extract_objects`, `connected_components`, `detect_symmetry`, `detect_periodicity`, `filter_items`, `select_best`, `choose_consensus`, `crop`, `translate`, `rotate`, `reflect`, `recolor`, `replace_if`, `trace_path`, `fill_region`, `propagate_until_blocked`, `infer_mapping`, `infer_template`, `majority_merge`, `if`, `for_each`, `map`, `reduce`, `return`.

## Artifact Integration

Extend the existing artifact model rather than forking it. A reasoning payload may include adapter ID, task reference, program, extracted features, score, trace references, and counterexample references.
Recommended reasoning artifact kinds are `hypothesis`, `transform-rule`, `counterexample`, `execution-trace`, `macro`, `penalty-profile`, and `penalty-decision`.

## Search And Retrieval

The runtime should search over programs, not benchmark templates. The loop is: parse task, extract features, retrieve related programs and counterexamples, generate candidates, execute, score, retain top candidates, accept human feedback, and promote stable subprograms.
Structured retrieval should coexist with the current text-based store and use keys such as input-output signature, object-count changes, symmetry profile, periodicity profile, graph motifs, and prior macro success.

## Human-Tunable Penalties

Candidate score should combine fit with penalties for complexity, instability, overfitting, and runtime cost, then apply an explicit human adjustment term.
Human input must work at three levels: global profile, adapter-specific profile, and per-candidate override.
A penalty profile should minimally store weights for complexity, instability, overfitting, and runtime, plus optional overrides and reviewer notes.
A penalty decision should record exactly what was changed, why it was changed, who changed it, and which target was affected.

## Optional ARC Adapters

ARC-v2 and ARC-v3 adapters remain optional packages. They must not introduce core dependencies, core prompts, or core artifact assumptions.

## Success Criteria

- Existing PIL flows remain unchanged by default.
- Induced programs are inspectable and replayable.
- Repeated subprograms become reusable macros.
- Humans can tune penalty values and promotion decisions directly.
- Future domains can adopt the runtime through adapters with limited friction.
