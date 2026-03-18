# devchat-ui

A browser-based dashboard for monitoring and controlling multi-agent developer chatlogs in real time.

---

## What is devchat?

Modern AI-assisted development often involves more than one AI agent working alongside a human developer. Each agent may specialise in a different task — code review, spec writing, testing, documentation — and they need a way to communicate with each other and with the human lead without constant manual intervention.

**devchat** is a lightweight coordination protocol built on a single shared Markdown file (`chatlog.md`). Every participant — human or AI — appends timestamped entries to that file using a consistent format:

```
### YYYY-MM-DD HH:MM:SS - DEV#N

Message body in Markdown.
```

A companion watcher process (`watch.mjs`) monitors the file and automatically invokes the assigned AI agent whenever a new entry appears from an authorised source. The result is a self-sustaining conversation loop: the human writes, the agents read and reply, and the conversation accumulates as a permanent, auditable log.

**devchat-ui** adds a GUI layer on top of that protocol. Instead of editing the raw Markdown file in a text editor and watching terminal output to see agent replies, you get a live dashboard with collapsible entries, search, one-click agent control, and automation rules — all without leaving the browser.

---

## Why a browser-based GUI?

| Concern | Approach |
|---|---|
| Cross-platform | Node.js + any browser — no binary packaging |
| No build step | Vanilla JS + CSS; `marked` loaded from CDN |
| Lightweight | One external dependency (`ws` for WebSocket) |
| Upgradeable | Can be wrapped in Electron later with minimal changes |
| Multi-user | Any team member on the same machine can open the URL |

---

## Features

- **Real-time chatlog view** — new entries appear instantly via WebSocket; auto-scrolls unless you have scrolled up
- **Timeago timestamps** — shows "2 m ago", "just now", etc.; hover reveals the exact timestamp
- **Freshness emphasis** — entries under ~45 s glow with a pulsing "● Live" badge; entries under 5 min show a "New" badge with a coloured left border
- **Collapse / expand** — click any entry header to toggle; bulk controls for collapse-all, expand-all, collapse-before-date
- **Search** — full-text, client-side, with inline highlighting; `Escape` to clear
- **Filter sidebar** — by author, date range, "new only", "has code block"
- **Agent control panel** — Start / Stop / Trigger each agent; live stdout log drawer per agent
- **Compose & post** — Markdown editor with live preview; posts via append-only write (safe against concurrent agents); `Ctrl+Enter` shortcut
- **Automation rules** — trigger an agent automatically N seconds after a human (DEV#0) posts; rules stored in `ui-config.json`
- **Dark mode** — follows `prefers-color-scheme` automatically

---

## Requirements

- **Node.js 18+** on PATH
- **`ws` package** — install once (see below)
- **`claude` CLI** (`npm i -g @anthropic-ai/claude-code`) or **`codex`** — for the AI agents
- **`ANTHROPIC_API_KEY`** environment variable — required by `claude`

---

## Quick start

### 1. Install the one dependency

```bash
cd tools/devchat-ui
npm install
```

### 2. Start the server

**Windows:**
```cmd
tools\devchat-ui\ui.cmd --open
```

**Linux / macOS:**
```bash
bash tools/devchat-ui/ui.sh --open
```

`--open` launches the dashboard in your default browser automatically. Without it, open `http://localhost:3737` yourself.

### 3. Use it standalone (outside this repo)

Copy the `tools/devchat-ui/` directory anywhere, then:

```bash
npm install
node server.mjs --chatlog /path/to/your/chatlog.md --open
```

The UI does not depend on any other file in this repo.

---

## Server options

| Flag | Default | Description |
|---|---|---|
| `--port <n>` | `3737` | HTTP port |
| `--chatlog <path>` | from `ui-config.json` | Override chatlog path |
| `--config <path>` | `.private/devchats/ui-config.json` | Config file location |
| `--open` | off | Open browser on startup |

---

## Configuration — `ui-config.json`

Created automatically on first run. Edit via the ⚙ Settings panel in the UI or directly:

```json
{
  "chatlog": ".private/devchats/chatlog.md",
  "rules":   ".private/devchats/rules.txt",
  "port":    3737,
  "agents": [
    { "id": "DEV#1", "agent": "claudecode", "respondTo": ["DEV#0"] },
    { "id": "DEV#2", "agent": "claudecode", "respondTo": ["DEV#0"] }
  ],
  "automationRules": []
}
```

| Field | Description |
|---|---|
| `chatlog` | Path to the shared Markdown chatlog (relative to repo root) |
| `rules` | Path to `rules.txt` injected into each agent prompt |
| `port` | HTTP port for the UI server |
| `agents[].id` | Developer ID (e.g. `DEV#1`) |
| `agents[].agent` | `claudecode` or `codex` |
| `agents[].respondTo` | Authors whose entries trigger this agent (default `["DEV#0"]`) |
| `automationRules` | Trigger-action pairs; managed via the Automation panel |

---

## Automation rules

An automation rule fires a one-shot agent trigger automatically when a matching entry is written to the chatlog.

Default rules (added via the + button or manually):

```json
{
  "id": "...",
  "enabled": true,
  "label": "DEV#0 → DEV#1",
  "trigger": { "type": "entry_from", "author": "DEV#0" },
  "action":  { "type": "trigger_agent", "agentId": "DEV#1", "delayMs": 5000 }
}
```

The 5-second delay gives you time to post a follow-up before the agent fires.

---

## How it works — end to end

```
DEV#0 types in Compose box
  │
  ▼
POST /api/entries  →  fs.appendFile(chatlog.md)
  │
  ▼
ChatlogWatcher detects change  →  parses new entries  →  broadcasts via WebSocket
  │                                                            │
  ▼                                                            ▼
AutomationEngine evaluates rules                   Browser receives entries_added
  │                                                displays new card with "● Live" badge
  ▼
POST /api/agents/DEV#1/trigger  →  spawns claude --print  →  appends response
  │
  ▼
ChatlogWatcher detects agent reply  →  broadcasts  →  Browser adds DEV#1 card
```

If the continuous watcher (`watch.mjs`) is also running in its own terminal, it will detect the human entry independently and may reply concurrently. Both paths write with `fs.appendFile` so entries are never overwritten.

---

## Security

- The server binds to `127.0.0.1` only. It is **not** exposed on the network by default.
- No credentials are sent to the browser. `ANTHROPIC_API_KEY` stays server-side.
- `POST /api/entries` can only write to the configured chatlog; no arbitrary file access.

---

## Chatlog format

Every entry must follow this format (same as `watch.mjs` and `rules.txt`):

```
### YYYY-MM-DD HH:MM:SS - DEV#N

Body in Markdown.
```

- One blank line after the header.
- LF line endings.
- Entries are appended in chronological order; never deleted or reordered.

---

## Relationship to `devchat-watch`

`devchat-watch/watch.mjs` is a standalone continuous watcher that can run without the UI — useful on headless servers or in CI. `devchat-ui` can start and stop those same watch.mjs processes from the browser, or it can invoke agents directly via the Trigger button (a one-shot call that bypasses the file-watch cycle).

You can run both together: the UI for interactive sessions, `watch.mjs` in a background terminal for always-on monitoring.

---

## Troubleshooting

**`Cannot find package 'ws'`**
Run `npm install` inside `tools/devchat-ui/`.

**Port 3737 already in use**
Pass a different port: `node server.mjs --port 3800 --open`

**Chatlog not found**
Pass the path explicitly: `node server.mjs --chatlog /absolute/path/to/chatlog.md`

**Agent trigger fails with "claude not found"**
Install the Claude CLI: `npm i -g @anthropic-ai/claude-code`
Then set `ANTHROPIC_API_KEY` in your shell environment.

**Entries don't appear in real time**
The server uses `fs.watch`. On some network drives or virtual filesystems `fs.watch` does not fire reliably — use a local path.
