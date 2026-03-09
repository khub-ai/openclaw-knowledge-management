# Contributing

Thanks for your interest in contributing to **KHUB Knowledge Fabric**.

This repo implements a portable knowledge layer focused on **persistable interactive learning (PIL)**: capturing user knowledge (patterns, preferences, rubrics, constraints, value-judgments, etc.), validating/compacting it into artifacts, and safely recalling/applying it across sessions and agents.

## Scope and contribution types

We welcome contributions in these areas:

- **Knowledge artifact schemas**
  - JSON/YAML schemas, type definitions, validation rules
- **Acquisition & induction**
  - turning interaction traces into candidate artifacts
- **Validation & governance**
  - confirmation flows, confidence scoring, applicability scoping, revision/retirement
- **Retrieval & ranking**
  - context-aware recall beyond naive keyword search
- **Safety controls**
  - inspection, export, deletion, privacy controls
- **Integration**
  - plugin loading, compatibility with OpenClaw and other agent platforms
- **Docs & examples**
  - runnable demos, benchmark tasks, evaluation harnesses

If you’re unsure where to start, open an issue describing the idea and desired behavior.

## Development setup (pnpm workspace)

### Prerequisites

- Node.js LTS (recommend 20.x)
- pnpm (via Corepack or pnpm installer)

### Install

From repo root:

    pnpm install

### Run a package

Examples:

    pnpm --filter @khub-ai/knowledge-fabric dev
    pnpm --filter @you/skills-foo test

List all workspace projects:

    pnpm -r list --depth 0

## Branching and PRs

- Use a short-lived feature branch: `feat/<topic>` or `fix/<topic>`
- Keep PRs focused (one behavior change per PR when possible)
- Include rationale: what knowledge type is being handled, expected lifecycle, and safety considerations
- Add/adjust tests or at least a reproducible demo script

## Commit messages

Use clear, descriptive commits. Conventional commits are appreciated but not required.

Examples:

- `feat: add artifact schema for generalized patterns`
- `fix: prevent low-confidence artifacts from auto-apply`
- `docs: clarify retrieval ranking policy`

## Code style and quality

- Prefer small modules with explicit interfaces (schemas and policies should be inspectable)
- Avoid hard-coding environment-specific paths
- Keep “apply” behavior gated by explicit policy decisions (risk tier + confidence)
- Document new artifact types: intended use, examples, and revision rules

## Reporting security issues

If you believe you’ve found a security or privacy issue, please avoid filing a public issue. Contact the maintainers privately (or use your organization’s security process).

## License

By contributing, you agree that your contributions will be licensed under the repository license.

