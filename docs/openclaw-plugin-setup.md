# Running PIL inside OpenClaw

This guide walks through installing `openclaw-plus` as a live plugin inside a running OpenClaw instance, so the agent has access to your persisted knowledge artifacts through the `knowledge_search` tool.

---

## What works right now

| Capability | Status |
|---|---|
| `knowledge_search` tool — agent queries the artifact store on demand | ✅ Wired |
| Pre-populating the store via the standalone playground | ✅ Works |
| Passive elicitation — agent learns automatically from every message | 🚧 Planned (Milestone 1c) |
| Automatic injection into every prompt | 🚧 Planned (Milestone 1d) |

In practical terms: the agent can **retrieve and apply** what it already knows. It will not yet **learn passively** from OpenClaw conversations — you teach it explicitly via the playground or computer-assistant demo, and the agent draws on that knowledge inside OpenClaw.

---

## Prerequisites

| Requirement | Check |
|---|---|
| OpenClaw ≥ 2026.1.26 | `openclaw --version` |
| Node.js ≥ 22.12.0 | `node --version` |
| OpenClaw gateway running | `openclaw status` |
| `openclaw-knowledge-management` repo cloned | `ls ~/src/openclaw-knowledge-management` |
| Repo dependencies installed | `pnpm install` from repo root |

> **WSL note:** All commands below should be run from WSL, where OpenClaw runs. The repo is at `~/src/openclaw-knowledge-management` (symlinked from `C:\_backup\openclaw\openclaw-knowledge-management`).

---

## Step 1 — Link the plugin

The `--link` flag tells OpenClaw to load the plugin directly from the source directory without copying it. Changes you make to the plugin source are reflected immediately after a gateway restart.

```bash
openclaw plugins install --link ~/src/openclaw-knowledge-management/packages/openclaw-plus
```

Expected output: something like `✓ Linked plugin knowledge-management from ~/src/openclaw-knowledge-management/packages/openclaw-plus`

To verify it was discovered:

```bash
openclaw plugins list
```

Look for `knowledge-management` in the output. If it appears but shows as disabled, continue to Step 2.

---

## Step 2 — Enable the plugin

```bash
openclaw plugins enable knowledge-management
```

Alternatively, add it to your config file manually (`~/.openclaw/openclaw.json`):

```json5
{
  plugins: {
    entries: {
      "knowledge-management": {
        enabled: true
      }
    }
  }
}
```

---

## Step 3 — Restart the gateway

Plugin changes take effect only after a full gateway restart:

```bash
openclaw gateway restart
```

Or stop and start:

```bash
openclaw stop
openclaw start
```

After restart, confirm the plugin loaded without errors:

```bash
openclaw plugins info knowledge-management
```

The output should show `status: loaded` and list the `knowledge_search` tool.

---

## Step 4 — Verify in a chat session

Start a chat session with your OpenClaw agent and ask:

> "What tools do you have available?"

The agent should mention `knowledge_search`. You can also invoke it directly:

> "Search your knowledge base for anything about file naming conventions."

If the store is empty, the tool returns an empty result — that is correct and expected before you populate it.

---

## Step 5 — Populate the knowledge store

The playground and computer-assistant demo write artifacts to the same JSONL file that OpenClaw reads:

```
~/.openclaw/knowledge/artifacts.jsonl
```

### Option A — Run the playground

The playground processes a hardcoded sample dialogue and stores whatever it extracts. Run it once to seed some test artifacts:

```bash
cd ~/src/openclaw-knowledge-management
export ANTHROPIC_API_KEY=sk-ant-...     # if not already set
pnpm start
```

After it completes, inspect the artifact file:

```bash
cat ~/.openclaw/knowledge/artifacts.jsonl
```

### Option B — Run the computer-assistant demo

The computer-assistant is a full interactive REPL that learns from your actual instructions across multiple sessions:

```bash
cd ~/src/openclaw-knowledge-management/apps/computer-assistant
node --loader ts-node/esm/transpile-only --no-warnings index.ts
```

Teach it something during the session (e.g. "always open PDFs in Evince") and the artifact will be available the next time you ask the OpenClaw agent to search its knowledge.

### Option C — Write artifacts directly

Artifacts are plain JSON lines. You can append one manually if you want to test quickly:

```bash
cat >> ~/.openclaw/knowledge/artifacts.jsonl << 'EOF'
{"id":"manual-001","kind":"preference","content":"User prefers responses in bullet-point format, not prose paragraphs.","certainty":"definitive","confidence":0.9,"stage":"consolidated","scope":"global","tags":["format","style","response"],"evidence":["manually added"],"evidenceCount":1,"salience":"high","provenance":{"createdAt":"2026-03-07T00:00:00Z","sourceConversation":"manual","createdBy":"user"},"lifecycle":{"status":"active"}}
EOF
```

---

## Step 6 — Use knowledge in a session

Once artifacts exist, the agent calls `knowledge_search` whenever it judges the query is relevant. You can also ask it explicitly:

> "Before we start — search your knowledge base for any preferences I've told you about."

The agent will retrieve relevant artifacts and apply them for the rest of the session. You can prompt it to do this at the start of every conversation until passive injection (Milestone 1d) is wired.

---

## Troubleshooting

### Plugin not appearing in `openclaw plugins list`

Check that `openclaw.plugin.json` is present in the linked directory:
```bash
cat ~/src/openclaw-knowledge-management/packages/openclaw-plus/openclaw.plugin.json
```
If missing, the plugin directory was not recognised as a valid plugin.

### `knowledge_search` not available to agent

Run `openclaw plugins info knowledge-management` — if status shows `error`, check OpenClaw logs:
```bash
openclaw logs --follow
```
Common cause: TypeScript transpilation failure. OpenClaw loads plugin TypeScript via jiti; if jiti cannot resolve a type import, the plugin will fail to load. File an issue with the log output.

### Store path mismatch

The plugin defaults to `~/.openclaw/knowledge/artifacts.jsonl`. If your OpenClaw runs under a different user or `OPENCLAW_STATE_DIR` is set, override the store path to match:

```bash
export KNOWLEDGE_STORE_PATH="$OPENCLAW_STATE_DIR/knowledge/artifacts.jsonl"
```

Add this to `~/.openclaw/.env` to make it permanent.

### Artifacts written by playground not appearing in OpenClaw

Both the playground and the plugin read from the same path by default (`~/.openclaw/knowledge/artifacts.jsonl`). If they differ, set `KNOWLEDGE_STORE_PATH` to the same value in both environments.

---

## Upgrading the plugin

Because the plugin is linked (not copied), pulling changes to the repo and restarting the gateway is all that is needed:

```bash
cd ~/src/openclaw-knowledge-management
git pull
pnpm install          # only needed if dependencies changed
openclaw gateway restart
```

---

## What comes next

When Milestones 1c and 1d are complete:

- **1c (Passive elicitation)**: The plugin will use OpenClaw's `message_received` hook to observe every message and extract knowledge automatically — no explicit teaching needed.
- **1d (Tier 1 triggering)**: Relevant artifacts will be injected into every prompt via the `before_prompt_build` hook, so the agent applies what it knows without being asked to search.

At that point, the agent will learn from your OpenClaw conversations directly and apply that knowledge transparently — the `knowledge_search` explicit call becomes a fallback rather than the primary path.
