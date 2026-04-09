 
# Glossary

This is the canonical terminology page for the repository. When a term such as
`knowledge artifact`, `knowledge patch`, or `Knowledge Fabric` appears elsewhere
in the docs, this page is the intended reference point.

If you want a quick first explanation rather than term definitions, start with
[what-is-kf.md](what-is-kf.md).

## PIL (Persistable Interactive Learning)

PIL is the core learning pattern behind this repository: an agent learns from
interaction, turns what it learns into reusable knowledge artifacts, and
applies that knowledge later under confidence-aware controls.

## Knowledge Fabric (KF)

Knowledge Fabric, or KF, is the broader runtime knowledge layer built around
PIL. In plain terms, it is the mechanism by which knowledge is elicited,
generalized, stored, revised, retrieved, and reused across sessions, tasks, or
agents.

KF is typically used alongside one or more LLMs, VLMs, or multimodal models.
It is not itself the base model; it is the explicit knowledge layer around
those models.

## Knowledge Artifact

A knowledge artifact is the basic stored unit of reusable knowledge in this
project.

It is:

- **stored** so it persists across sessions
- **human-readable** so users can inspect and edit it
- **reusable** so it can be retrieved and applied in future situations
- **revisable** so it can be corrected, superseded, or retired

A knowledge artifact may capture a fact, preference, procedure, judgment,
strategy, boundary condition, revision rule, or failure case.

Importantly, a knowledge artifact is **not** just any remembered text snippet.
It is a curated or structured unit that the system can reason about, retrieve,
and apply intentionally.

## Generalized Knowledge Artifact

A generalized knowledge artifact is an artifact that captures a pattern, rule,
or heuristic extending beyond a single episode. It is more reusable than a raw
transcript excerpt or one-off memory.

## Knowledge Patch

A knowledge patch is a bounded application of one or more knowledge artifacts to
improve system behavior in a specific context.

In other words:

- an **artifact** is the stored knowledge unit
- a **patch** is that knowledge being applied to a concrete task or failure mode

A patch may consist of a single artifact or a small set of coordinated
artifacts.

## Rule

A rule is a specific claim, instruction, or decision criterion contained within
an artifact or patch.

Examples:

- "Always use bullet points for executive summaries."
- "If bare red skin extends around the eye, it is a Red-faced Cormorant."

An artifact can contain one rule or many related rules.

## Memory

Memory is the broader umbrella term for retained information from prior
interaction. Not all memory is yet a knowledge artifact.

For example:

- a raw transcript excerpt or event log is memory
- a distilled, reusable rule derived from repeated observations is a knowledge artifact

The project focuses on transforming useful memory into explicit artifacts.

## Procedure Artifact

A procedure artifact describes how to do something in a repeatable way. In some
contexts this may later be compiled into code, but the canonical form remains
human-readable.

## Judgment Artifact

A judgment artifact captures an evaluative principle, such as what counts as
good evidence, a dangerous pattern, or an attractive opportunity.

## Strategy Artifact

A strategy artifact captures a general approach to a class of problems rather
than a single procedure for one task.

## Boundary Artifact

A boundary artifact describes when a rule, strategy, or procedure should not be
applied, or when its confidence should be reduced.

## Revision Trigger

A revision trigger is a condition or kind of evidence that should cause the
agent to revise or abandon a previously learned rule.

## Failure Artifact

A failure artifact records a past mistake, failure mode, or misleading case
that helps refine future judgment.

## Domain Adapter

A domain adapter is the domain-specific layer that exposes tasks, renders domain
inputs, and provides validation or support utilities without hardcoding
domain-specific solution logic into the core runtime.

## Dialogic Learning

Dialogic learning is learning through structured back-and-forth exchange rather
than passive observation alone. In this repo, it refers to an agent learning
through purposeful dialogue with a user or expert.

In the image classification use cases, dialogic learning takes a concrete form:
a cheap "pupil" vision model makes a wrong prediction; an expert examines the
failure and explains — in plain language — what field mark or visual criterion
the pupil missed; the system converts that explanation into a testable rule;
the rule is validated against known images; and the pupil is re-run with the
rule active. Each cycle is one turn of the dialogue. The knowledge is
persistent: a rule registered in one session applies to future sessions without
repeating the exchange.

The current implementation supports **multi-round exchanges**: when the pupil
still fails after round 1, the system assembles a context block (which rules
were active, whether they fired, the pupil's stated reasoning, per-precondition
validator observations on the failure image) and re-engages the expert for
round 2. The expert's second response is shaped by the pupil's specific failure
mode rather than just the original image. This is closer to dialogue than a
single-pass injection.

**The long-term goal is true dialogic learning**, where the pupil actively
initiates questions, expresses degrees of uncertainty, and negotiates with the
tutor over which features were ambiguous or not visible. In that mode the
information flow is genuinely bidirectional: the pupil's expressed confusion
drives the tutor's next explanation, and the tutor's explanation is evaluated
against the pupil's actual reasoning gaps. Whether this is achievable depends
significantly on the capability of the pupil model. A pupil that produces only
a prediction and a brief justification gives the system little to work with.
A pupil that can articulate *what it was looking for*, *what it found*, and
*where its confidence broke down* — in enough detail for a tutor to respond
usefully — is a prerequisite for genuine back-and-forth.

This has been demonstrated to work with AI models as both pupil and expert —
see the [birds](../usecases/image-classification/birds/README.md) and
[dermatology](../usecases/image-classification/dermatology/README.md) use
cases.

## Expert-to-Agent Dialogic Learning

[Expert-to-agent dialogic learning](../specs/expert-to-agent-dialogic-learning.md)
is the specific pattern in which an agent learns deep, reusable knowledge from
a domain expert through carefully structured questioning, synthesis,
correction, and consolidation.

## Natural-Language DSL

A natural-language DSL is a constrained, well-defined pseudo-code style
expressed in ordinary language. It is designed to be readable by humans and
consistently interpretable by an LLM.

## Validation Record

A validation record captures the outcome of testing a candidate rule, solution,
or generalization against one or more cases.
