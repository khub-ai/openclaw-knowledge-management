# Benchmarks and Walkthroughs

This directory contains instructional documents for every runnable test program and benchmark in the project. Each document explains what the program tests, how to run it, how to read its output, and how to use that output as a health signal for the pipeline.

## Purpose

The automated test suite (`pnpm test`) runs quickly, requires no API key, and is suitable for CI. It catches logic bugs in the pipeline. It does not catch:

- LLM quality regressions (extraction quality, tag normalization, consolidation output)
- Prompt structure problems that only manifest with a real model
- End-to-end integration failures (store → pipeline → retrieve → inject round-trip)
- Behavioral regressions in how the pipeline handles edge cases at the boundary of knowledge vs. noise

The programs documented here fill that gap. They are not part of CI — they require a live API key and take seconds to minutes — but they provide the most realistic signal about system health.

## Index

| # | Document | Program | Milestone | What it validates |
|---|---|---|---|---|
| 01 | [playground.md](./01-playground.md) | `apps/playground/index.ts` | 1b / 1c / 1d | Full PIL pipeline: extraction, accumulation, retrieval, apply, revise |

*Additional documents will be added as each milestone matures.*

## How these documents are structured

Each document follows a common structure:

1. **Purpose** — what the program tests and what milestone(s) it covers
2. **Coverage table** — which pipeline stages are exercised
3. **Prerequisites and how to run** — exact commands including environment setup
4. **Annotated output** — the expected output with explanations inline
5. **Healthy run checklist** — a concise table of pass/fail criteria
6. **Common variations** — what different store states produce
7. **Relationship to automated tests** — how this complements `pnpm test`
8. **Known quirks** — expected warnings or display issues that are not bugs

## Running programs against a clean store

Most programs read from and write to the shared JSONL store at `~/.openclaw/knowledge/artifacts.jsonl`. When store state from prior runs affects the output in ways that complicate interpretation, run with an isolated store:

```bash
KNOWLEDGE_STORE_PATH=/tmp/pil-clean.jsonl pnpm start
```

Or delete the shared store before a baseline run:

```bash
# macOS / Linux
rm ~/.openclaw/knowledge/artifacts.jsonl

# Windows PowerShell
Remove-Item $env:USERPROFILE\.openclaw\knowledge\artifacts.jsonl
```
