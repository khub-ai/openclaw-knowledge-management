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

## Architecture
1. Core PIL handles persistence, provenance, retrieval, lifecycle, revision, confidence, and feedback.
2. Procedural Runtime handles generic program induction and deterministic execution over primitives.
3. Domain Adapters parse inputs, extract features, register primitives, generate candidates, and verify outputs.
4. Benchmark Harness evaluates adapters but does not define the runtime model.

## Program Model
Programs use a benchmark-neutral intermediate representation built from typed primitive calls with named inputs, outputs, and ordered steps.
The representation must be plain-text serializable, executable without an LLM, inspectable, replayable, composable, and portable across domains.

## Learnable Procedural Primitives
The base primitive set should stay small and generic. Expressivity should grow mainly through learned macros promoted from successful recurring subprograms.
Macro names must describe general behavior, for example compress_repeated_motif, not benchmark-specific task IDs.
Recommended primitive families are structural, selection, transformation, propagation, inference, and control.

## Artifact Integration
Extend the existing artifact model rather than forking it. A reasoning payload may include adapter ID, task reference, program IR, extracted features, score, and trace references.
Recommended reasoning artifact kinds are hypothesis, transform-rule, counterexample, execution-trace, macro, and penalty-profile.

## Search And Retrieval
The runtime should search over programs, not benchmark templates. The loop is: parse task, extract features, retrieve related programs and counterexamples, generate candidates, execute, score, retain top candidates, accept human feedback, and promote stable subprograms.
Structured retrieval should coexist with the current text-based store and use keys such as input-output signature, object-count changes, symmetry profile, periodicity profile, graph motifs, and prior macro success.

## Human-Tunable Penalties
Candidate score should combine fit with penalties for complexity, instability, overfitting, and runtime cost, then apply an explicit human adjustment term.
Human input must work at three levels: global profile, adapter-specific profile, and per-candidate override.
A penalty profile should minimally store weights for complexity, instability, overfitting, and runtime, plus optional overrides and reviewer notes.

## Optional ARC Adapters
ARC-v2 and ARC-v3 adapters remain optional packages. They must not introduce core dependencies, core prompts, or core artifact assumptions.

## Success Criteria
- Existing PIL flows remain unchanged by default.
- Induced programs are inspectable and replayable.
- Repeated subprograms become reusable macros.
- Humans can tune penalty values and promotion decisions directly.
- Future domains can adopt the runtime through adapters with limited friction.
