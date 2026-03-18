# Spec: devchat-ui — Cross-Platform GUI for Chatlog Monitoring and Agent Control

**Status:** Draft v0.1
**Author:** DEV#1
**Date:** 2026-03-18
**Related:** `tools/devchat-watch/`, `.private/devchats/chatlog.md`

---

## 1. Goal

`devchat-ui` is a lightweight browser-based GUI that replaces manual chatlog.md editing and terminal monitoring with a unified, real-time dashboard. It lets DEV#0 (and any observer) read the chatlog as it grows, post entries without touching the raw file, manually trigger or suppress agent responses, and configure automation rules — all from a single window.

---

## 2. Non-Goals (v1)

| Deferred | Reason |
|---|---|
| AI-generated section summaries | Requires LLM call per collapsed block; adds cost and latency |
| Multi-chatlog / multi-project support | Single-repo scope sufficient for v1 |
| Authentication / multi-user access control | Single-developer local server; network exposure is out of scope |
| Packaging as Electron binary | Browser-based local server is lighter and already cross-platform |
| Drag-and-drop entry reordering | Chatlog is append-only; reordering would break the audit trail |

---

## 3. Tech Stack

### Why a local Node.js web server (not Electron / Tauri)

| Criterion | Local server + browser | Electron | Tauri |
|---|---|---|---|
| Dependencies | Node.js only (already required) | Chromium binary (~150 MB) | Rust toolchain |
| Cross-platform | ✅ trivial | ✅ with packaging | ✅ with build setup |
| Distribution | `node server.mjs` or `.cmd` | Packaged installer | Packaged binary |
| Web dev skills | ✅ same as app dev | ✅ | ✅ |
| File system access | Server-side (full) | Main process (full) | Rust commands |
| Upgrade path | Wrap in Electron later | — | — |

The local server approach eliminates all binary packaging complexity while keeping the full Node.js ecosystem available server-side. If a standalone app experience becomes important later the server can be wrapped in Electron with minimal changes.

### Stack

| Layer | Choice | Notes |
|---|---|---|
| Server runtime | Node.js 18+ (`server.mjs`) | ESM, no transpiler |
| HTTP | `node:http` built-in | No Express needed for v1 |
| Real-time push | `ws` (WebSocket library) | Only external dependency |
| Frontend | Vanilla JS + CSS | Single `index.html`; no build step |
| Styling | CSS custom properties | Light/dark theme via `prefers-color-scheme` |
| Markdown rendering | `marked` (CDN) | Renders chatlog body content |

---

## 4. Architecture

```
┌───────────────────────────────────────────────────────┐
│  Browser  (index.html + app.js)                       │
│  ┌─────────────────────┐  ┌──────────────────────┐   │
│  │ ChatlogView          │  │ AgentPanel           │   │
│  │ EntryCard (collapse) │  │ Trigger / Start/Stop │   │
│  │ Compose box          │  │ Automation rules     │   │
│  │ Search / Filter bar  │  │ Status indicators    │   │
│  └──────┬──────────────┘  └─────────┬────────────┘   │
│         │   WebSocket + REST         │                 │
└─────────┼──────────────────────────┼─────────────────┘
          │ ws://localhost:PORT        │ GET/POST /api/*
┌─────────┴────────────────────────────────────────────┐
│  server.mjs  (Node.js)                               │
│  ┌────────────────────────────────────────────────┐  │
│  │ ChatlogWatcher  (fs.watch → parse → broadcast) │  │
│  │ AgentManager    (spawn / kill watch.mjs)        │  │
│  │ REST router     (/api/entries, /api/agents, …)  │  │
│  │ WebSocket hub   (broadcast to all clients)      │  │
│  │ AutomationEngine (rule eval on each new entry)  │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────┬───────────────────────────────┘
                       │ child_process.spawn
          ┌────────────┴────────────┐
          │  watch.mjs  instances   │
          │  DEV#1, DEV#2, …        │
          └─────────────────────────┘
                       │ fs.appendFile
          ┌────────────┴────────────┐
          │  chatlog.md             │
          └─────────────────────────┘
```

---

## 5. File Layout

```
tools/devchat-ui/
├── server.mjs            Node.js HTTP + WebSocket server
├── client/
│   ├── index.html        Single-page application shell
│   ├── app.js            UI logic (vanilla JS, ES modules via <script type=module>)
│   └── style.css         Layout, themes, entry cards
├── ui.cmd                Windows: node %~dp0server.mjs %*
├── ui.sh                 Unix:    node "$(dirname "$0")/server.mjs" "$@"
└── README.md
```

Configuration and state (gitignored via `.private/`):

```
.private/devchats/
├── ui-config.json        Agent definitions + automation rules (UI-editable)
└── ui-state.json         Collapse state, scroll position, sidebar width
```

---

## 6. Feature Specification

### 6.1 Real-Time Chatlog View

**Behaviour**

- On load, `GET /api/entries` fetches all parsed entries and renders them.
- `ChatlogWatcher` on the server uses `fs.watch` on `chatlog.md`. On change it re-parses
  the file, diffs against the last known state, and emits a `entries_added` WebSocket event
  with only the new entries.
- The client appends new entry cards without re-rendering the full list.
- If the user is scrolled near the bottom (within 200 px), the view auto-scrolls to the
  newest entry. If the user has scrolled up, a "↓ N new entries" badge appears at the
  bottom; clicking it scrolls and clears the badge.

**Entry parsing**

Each `### YYYY-MM-DD HH:MM:SS - DEV#N` block is parsed into:

```ts
interface Entry {
  id:        string;        // sha256(header line) — stable across re-reads
  timestamp: string;        // "YYYY-MM-DD HH:MM:SS"
  author:    string;        // "DEV#N"
  body:      string;        // raw Markdown body
  isNew:     boolean;       // true until dismissed
}
```

---

### 6.2 Entry Cards — Collapse / Expand

**Individual collapse**

- Each entry is a card with a sticky header (`timestamp · author`) and a collapsible body.
- Click anywhere on the header to toggle. State persists in `ui-state.json`.
- Collapsed card shows: first 100 characters of plain-text body (stripped Markdown) + "…".
- Keyboard: `Space` or `Enter` on a focused card header toggles it.

**Bulk collapse controls** (toolbar)

| Button | Action |
|---|---|
| Collapse all | Collapse every card |
| Expand all | Expand every card |
| Collapse before date | Collapse entries older than a selected date (date-picker) |
| Collapse by author | Collapse / expand all cards from a specific DEV#N |

**Thread groups** (v1 stretch goal)

Consecutive entries from the same author with no intervening DEV#0 entry are visually
grouped with a thin left-border accent. The group can be collapsed as a unit with a
single click on a group toggle. The collapsed group summary shows:
`DEV#N · N entries · first entry preview`.

---

### 6.3 Search and Filter

**Search bar** (top of chatlog panel)

- Full-text search across all entry bodies (client-side, case-insensitive).
- Matches are highlighted inline (CSS `mark`); non-matching cards are hidden.
- Pressing `Escape` clears the query.

**Filter sidebar** (left panel)

| Control | Behaviour |
|---|---|
| Author checkboxes | Show only entries from selected authors |
| Date range | Show only entries within start…end date |
| "Has code block" | Show only entries containing fenced code |
| "New only" | Show only entries received since the UI opened |

Filters are AND-combined. Active filters show a badge count. Filters reset on page reload
(not persisted — they are session-level).

---

### 6.4 Agent Control Panel

**Status display** (right sidebar or top bar for compact mode)

For each configured agent:

```
DEV#1  ● Running   last response: 08:20:23
       [Trigger ▶]  [Stop ■]  [View log]
```

```
DEV#2  ○ Stopped
       [Start ▶]  [Trigger ▶]  [View log]
```

Status is pushed via `agent_status` WebSocket events; no polling.

**Actions**

| Action | Implementation |
|---|---|
| Start | `POST /api/agents/DEV1/start` — server spawns `watch.mjs --dev-id DEV#1` |
| Stop | `POST /api/agents/DEV1/stop` — server sends SIGTERM to the child process |
| Trigger | `POST /api/agents/DEV1/trigger` — server runs a one-shot `processChange()` invocation (or sends a signal to the running process) outside the file-watch cycle, forcing an immediate response attempt |

**Live agent log**

Clicking "View log" opens a drawer showing the last 200 lines of stdout from that agent's
`watch.mjs` process, streamed via a dedicated `agent_log` WebSocket event per agent ID.

---

### 6.5 Compose and Post

A compose box at the bottom of the chatlog panel:

- Author selector: defaults to `DEV#0`; can select any configured agent (for testing).
- Multi-line textarea with Markdown preview toggle (split view).
- "Post" button calls `POST /api/entries` which appends a formatted entry to `chatlog.md`.
  The server uses `fs.appendFile` (same safe approach as `watch.mjs`).
- Keyboard shortcut: `Ctrl+Enter` / `Cmd+Enter` posts.
- After posting, the new entry appears immediately (optimistic add while waiting for
  the fs.watch echo).

---

### 6.6 Automation Rules

Rules are stored in `.private/devchats/ui-config.json` and editable in a "Automation" tab
in the right panel.

**Rule schema**

```ts
interface AutomationRule {
  id:       string;       // uuid
  enabled:  boolean;
  trigger: {
    type:   "entry_from";      // v1: only one trigger type
    author: string;            // e.g. "DEV#0"
  };
  action: {
    type:    "trigger_agent";
    agentId: string;           // e.g. "DEV#1"
    delayMs: number;           // grace period before invoking (default: 5000)
  };
  label:    string;            // human-readable description
}
```

**Built-in default rules** (pre-configured, can be disabled)

| Rule | Trigger | Action |
|---|---|---|
| Human → DEV#1 | entry_from: DEV#0 | trigger_agent: DEV#1, delay 5 s |
| Human → DEV#2 | entry_from: DEV#0 | trigger_agent: DEV#2, delay 8 s |

**v1 limitation:** Rules fire unconditionally when the trigger fires. The agent itself
decides via its prompt whether a response is warranted (same `NO_RESPONSE_NEEDED` path).

**Future triggers** (not v1): `entry_mentions_agent`, `keyword_match`,
`no_response_within_N_minutes`, `scheduled_cron`.

---

### 6.7 Configuration

`ui-config.json` schema:

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

All paths are relative to the repo root. The file is created on first run with defaults.

---

## 7. REST API

| Method | Path | Body | Response | Description |
|---|---|---|---|---|
| `GET` | `/api/entries` | — | `Entry[]` | All parsed entries |
| `POST` | `/api/entries` | `{ author, body }` | `Entry` | Append new entry |
| `GET` | `/api/agents` | — | `AgentStatus[]` | All agent states |
| `POST` | `/api/agents/:id/start` | — | `AgentStatus` | Start watcher process |
| `POST` | `/api/agents/:id/stop` | — | `AgentStatus` | Stop watcher process |
| `POST` | `/api/agents/:id/trigger` | — | `{ queued: true }` | Force one-shot response |
| `GET` | `/api/config` | — | `UIConfig` | Current config |
| `PUT` | `/api/config` | `Partial<UIConfig>` | `UIConfig` | Update config |
| `GET` | `/api/rules` | — | `AutomationRule[]` | All automation rules |
| `POST` | `/api/rules` | `AutomationRule` | `AutomationRule` | Add rule |
| `PUT` | `/api/rules/:id` | `Partial<AutomationRule>` | `AutomationRule` | Update rule |
| `DELETE` | `/api/rules/:id` | — | `{ ok: true }` | Delete rule |

Static files served at `/` → `client/index.html`.

---

## 8. WebSocket Events

All events are JSON. Server → client unless noted.

| Event | Payload | Description |
|---|---|---|
| `entries_added` | `{ entries: Entry[] }` | One or more new chatlog entries |
| `agent_status` | `{ id, status, lastResponseAt }` | Agent process state changed |
| `agent_log` | `{ id, line }` | One stdout line from a running watcher |
| `config_updated` | `{ config }` | Config was changed by another client |
| `ping` | — | Keepalive; client responds with `pong` (client → server) |

---

## 9. UI Layout (wireframe)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  devchat-ui                              [🔍 Search…]  [⚙ Config]  [?] │
├───────────────┬─────────────────────────────────────────┬───────────────┤
│  FILTER       │  CHATLOG                      [↕ All]  │  AGENTS       │
│               │                               [+ New]  │               │
│  Authors      │  ┌─ 2026-03-13 15:00 · DEV#0 ────────┐ │  DEV#1  ●    │
│  ■ DEV#0      │  │ asking DEV#2 to review the spec    │ │  08:20:23    │
│  ■ DEV#1      │  └───────────────────────────────────┘ │  [▶ Trigger] │
│  ■ DEV#2      │                                         │  [■ Stop]    │
│               │  ► 2026-03-13 15:19 · DEV#2  ▸ 8 lines │  [📄 Log]    │
│  Date range   │    Strong concept, but from an impl…    │               │
│  From: [    ] │                                         │  DEV#2  ○    │
│  To:   [    ] │  ► 2026-03-13 15:45 · DEV#1  ▸ 4 lines │  [▶ Start]   │
│               │    In response to DEV#2: Good catch…    │  [▶ Trigger] │
│  Filters      │                                         │               │
│  □ Code only  │  ┄────────── 3 more entries ──────────┄ │  AUTOMATION   │
│  □ New only   │                                         │               │
│               │  ┌─ 2026-03-18 08:20 · DEV#1 ── NEW ─┐ │  ■ DEV#0→DEV#1│
│               │  │ In response to DEV#2's two review… │ │  ■ DEV#0→DEV#2│
│               │  │ …                                  │ │  [+ Add rule] │
│               │  └───────────────────────────────────┘ │               │
│               │                                         │               │
│               ├─────────────────────────────────────────┤               │
│               │  As: [DEV#0 ▾]                          │               │
│               │  ┌─────────────────────────────────┐   │               │
│               │  │ Type a message…                  │   │               │
│               │  └─────────────────────────────────┘   │               │
│               │  [Preview]                    [Post ⏎]  │               │
└───────────────┴─────────────────────────────────────────┴───────────────┘
```

**Responsive behaviour:** below 900 px wide the sidebar panels collapse into a top tab bar
(`Chatlog | Agents | Automation | Filter`).

---

## 10. Server Startup

```
node tools/devchat-ui/server.mjs [options]

Options:
  --port <n>       HTTP port (default: 3737; overridden by ui-config.json)
  --chatlog <path> Override chatlog path
  --config <path>  Override ui-config.json path
  --open           Launch default browser on startup
```

On Windows:

```
tools\devchat-ui\ui.cmd --open
```

On Linux / macOS:

```
bash tools/devchat-ui/ui.sh --open
```

---

## 11. Security

- Server binds to `127.0.0.1` only. It is **not** exposed on the network by default.
  LAN access requires an explicit `--host 0.0.0.0` flag and is the user's responsibility.
- No authentication in v1 (local-only, same trust level as the filesystem).
- Agent invocations inherit the server process's environment (`ANTHROPIC_API_KEY`).
  No credentials are transmitted to the browser.
- `POST /api/entries` only appends to the configured chatlog. No arbitrary file write.

---

## 12. Dependencies

| Package | Purpose | Size |
|---|---|---|
| `ws` | WebSocket server | ~50 KB |

All other server-side code uses Node.js built-ins (`http`, `fs`, `path`, `crypto`, `child_process`).

Client-side (loaded from CDN, no build step):
- `marked` — Markdown → HTML rendering

---

## 13. Out of Scope for v1 / Future Extensions

| Feature | Notes |
|---|---|
| AI-generated collapsed summaries | LLM call per block; add after MVP validated |
| `@mention` routing (`@DEV#2 please review`) | Parse mentions in entry body to trigger specific agent |
| Keyword / regex triggers in automation rules | Extend `AutomationRule.trigger` schema |
| `no_response_within_N_minutes` trigger | Useful for nudging stalled threads |
| Activity heatmap / statistics panel | Entry count by author over time |
| Multi-chatlog tabs | Multiple projects open side by side |
| Export (PDF, HTML) | One-shot render of chatlog as a readable document |
| Electron packaging | Wrap server + client once the design is stable |

---

## 14. Open Questions

1. **`--open` browser launch**: `child_process.exec('open ...')` / `start` / `xdg-open` — should this be automatic on startup or opt-in via flag? (Recommend: opt-in flag to avoid surprising CI environments.)

2. **Collapse state persistence**: Should `ui-state.json` sync across browser tabs in the same session, or is per-tab state fine for v1?

3. **`Trigger` semantics when agent is not running**: Should triggering a stopped agent auto-start it first, or error? (Recommend: auto-start with a warning toast.)

4. **Automation rule delay**: The 5-second default gives the human time to post a follow-up before the agent fires. Should this be configurable per-rule or global? (Recommend: per-rule.)

5. **Port conflict handling**: If port 3737 is already in use, should the server find the next free port automatically or exit with a clear error? (Recommend: exit with error + message suggesting `--port`.)
