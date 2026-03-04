---
name: knowledge-foo
description: "Example skill demonstrating how to surface and apply knowledge artifacts from the PIL store"
metadata:
  openclaw:
    emoji: "🧠"
    requires:
      extensions: ["knowledge-management"]
---

# knowledge-foo

An example skill that shows how to query the PIL knowledge store before acting,
so the agent can reuse learned procedures, preferences, and conventions instead
of re-deriving them each session.

## Usage

```
@openclaw knowledge-foo <task description>
```

The agent will:
1. Call `knowledge_search` with the task description to recall relevant artifacts
2. Apply high-confidence artifacts automatically
3. Present low-confidence suggestions for user confirmation before proceeding

## Example

```
User: Summarise this document for me
Agent: [calls knowledge_search("summarisation preference")]
       → Found artifact: "User prefers bullet-point summaries, max 5 points, no filler phrases"
       → Confidence 0.92 → auto-applied
Agent: Here are the key points: …
```

## Adding your own skills

Copy this directory, rename it, and update the `name` and `description` fields
in the frontmatter. Use `knowledge_search` at the start of your skill's
execution to pull in any relevant stored knowledge.
