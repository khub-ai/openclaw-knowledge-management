# ARC-AGI-3 DD — design spec (strict mode)

Last updated: 2026-04-21 (commit range `629956e` → `c302003`)

This is the authoritative description of the strict-mode discovery
play loop.  Update this file whenever design assumptions change.


## 1. Purpose & scope

Build an agent that plays ARC-AGI-3 games under **competition-legal
conditions**:
- Input is only the public `obs` object returned by `env.step()`
  (frame pixels, state enum, `levels_completed`, `win_levels`,
  `available_actions`).
- Output is a sequence of integer actions per env tick.
- Every perception, every hypothesis, every plan is derived at
  runtime from that public input.  No source-reading, no private
  game state, no hand-authored game-specific beliefs.

The agent is hybrid: **TUTOR** (a Claude API call) provides perception
and hypothesis reasoning; **harness** (pure Python) provides
mechanical capabilities (BFS, wall learning, probing, persistence).
A distillation corpus is captured each session for later training of
a smaller model on the same reasoning pattern.


## 2. Prime directive

> Neither TUTOR nor the harness may read privileged game state.
> Only fields published on the public `obs` object are available at
> runtime.  No hardcoded mappings from obfuscated internal names to
> semantic labels.  No authored knowledge base entries that encode
> game mechanics read from game source.

**Three-tier rule** (the people, the code, the knowledge):

| Role | `obs.*` | `env._game.*` | `ls20.py` source | Inject into TUTOR/harness |
|---|---|---|---|---|
| TUTOR                 | yes | **no** | **no** | — (TUTOR is the thing we're building) |
| Harness code          | yes | **no** | **no** | — (harness is the thing we're building) |
| Claude Code (supervisor, offline) | yes | yes  | yes  | **no** — substrate improvements only |

**Litmus test for every harness change**: would this still work on a
fresh ARC-AGI-3 game whose source I have not read?  If no, it is
injection and belongs behind `--legacy`.

Legacy mode (`--legacy`) retains the pre-strict scaffolding solely
as a benchmark upper-bound so we can measure the gap between
"oracle with source" and "blind discovery."  Legacy is not a
competition configuration.


## 3. Division of labor

```
┌────────────────────────────────────────────────────────────────────┐
│ TUTOR (Claude Sonnet API call, ~$0.01–$0.02 per turn)              │
│   - Visual identification of components from the rendered frame     │
│   - Hypothesis about role (agent / wall / goal / trigger / pickup)  │
│   - Target selection: where should the agent go next                │
│   - Predict + revise (prediction error is a discovery signal)       │
│                                                                     │
│ Does NOT: navigate, count budget, learn walls, track cells.        │
│ TUTOR outputs a single MOVE_TO target_cell per turn.                │
├────────────────────────────────────────────────────────────────────┤
│ Harness (pure Python, deterministic)                                │
│   - Bootstrap: probe each available action; infer cell_size +       │
│                agent sprite fingerprint                             │
│   - Cell system: pixel <-> cell translation anchored at spawn       │
│   - Pixel element extraction: connected components, palette counts  │
│   - Auto-probe: touch top-N distinctive components; record effects  │
│   - BFS in cell space with empirically-learned walls                │
│   - Wall confidence: cap per session, 2-strike rule, invalidation   │
│                       on successful move                            │
│   - Level replay: if we've solved this level before, execute        │
│                   recorded action sequence silently                 │
│   - Multi-level chaining: a session runs until WIN or GAME_OVER     │
│                           or max_turns or cost_cap                  │
│   - Frame rendering: upscaled color PNG with cell grid + agent      │
│                      highlight + distinctive-component labels      │
│   - Distillation capture: every TUTOR call recorded for SFT         │
│   - HTML trace generation: readable per-session viewer              │
│                                                                     │
│ Does NOT: infer what a component means, pick targets, or judge      │
│ whether a move was "correct" beyond what obs says.                  │
├────────────────────────────────────────────────────────────────────┤
│ Claude Code (me, offline development supervisor)                    │
│   - Build the substrate and prompts                                 │
│   - May read source to understand mechanics for ARCHITECTURAL       │
│     design (e.g. "games have HUDs so I should expect HUD-like       │
│     components"), but MAY NOT encode specific game knowledge        │
│     into TUTOR or harness                                           │
│   - Review traces, update this spec                                 │
└────────────────────────────────────────────────────────────────────┘
```


## 4. Module map (in `usecases/arc-agi-3-dd/python/`)

| Module | Role |
|---|---|
| `discovery_play.py`        | Entry point for strict-mode sessions. Runs bootstrap → per-level probe → TUTOR turn loop → persist + render |
| `discovery_prompts.py`     | `SYSTEM_DISCOVERY` + `USER_DISCOVERY_TEMPLATE`. Reasons purely in cells; tells TUTOR about the annotated frame image |
| `discovery_bootstrap.py`   | Mechanical probe of ACTION1-N to learn (dr, dc) per action from frame diffs, zero LLM |
| `pixel_elements.py`        | Connected-components extraction over palette frame; frame diffing; delta narration |
| `cell_system.py`           | `CellSystem` class: pixel ↔ cell conversion. `cell_size` = min |action effect| (per-game); `origin` = agent spawn centroid (per-level) |
| `render_strict_trace.py`   | Renders a training-data dir into `trace.html` (auto-invoked at end of every session) |
| `backends.py`              | `call_anthropic(system, user, image_b64)` wrapper with retry + usage accounting |
| `navigator.py`             | Legacy BFS; strict mode has its own `_bfs_plan_cells()` in `discovery_play.py` |
| `run_play.py`              | Legacy (pre-strict) play loop. Preserved behind `--legacy` for benchmark comparison |
| `planner.py`, `solve_and_record.py` | Legacy BFS+ solver using privileged input. Preserved for benchmark; refuses to run under strict. |


## 5. Discovery loop (per strict-mode session)

### 5.1 Session startup

1. `env.reset()` → obs.
2. Load KB from `knowledge_base/<game>_runtime.json`:
   - `action_effects_learned` (may skip bootstrap)
   - `agent_fingerprint`
   - `walls_by_level_cells`
   - `hypotheses_by_level`
   - `blocked_targets_by_level_cells` is NOT persisted; always starts empty.
3. Load recorded solutions from `<game>_strict_solutions.json`.
4. If any actions are unknown: run bootstrap (4–8 env.steps, 0 LLM).
5. Reset env so turn 1 starts at spawn.
6. Infer cell system from action effects + agent centroid at spawn.
   Spawn = cell `(0, 0)` by construction.

### 5.2 Per-turn flow

```
loop turn 1..max_turns:
  if obs.state != NOT_FINISHED: break
  if cost_usd_total >= cost_cap_usd: break          # hard cost cap

  cur_lc = obs.levels_completed
  if cur_lc not in level_entry_hashes:              # first entry to level
      level_entry_hashes[cur_lc] = hash(frame)

  # REPLAY if solved
  if recorded_solution_for(cur_lc) and hash_matches:
      execute recorded steps via env.step (no TUTOR)
      continue

  # Per-level re-initialization on level change
  if cur_lc != last_seen_lc:
      cs = infer_cell_system(action_effects, agent_centroid)
      walls_cells = kb.walls_by_level_cells[cur_lc]
      last_seen_lc = cur_lc

  # AUTO-PROBE once per new level (zero LLM)
  if cur_lc not in probed_levels:
      probed_levels.add(cur_lc)
      for cand in top-3 distinctive components:
          navigate to cand.cell via BFS; record diff
      probe_results_by_level[cur_lc] = results

  # Build prompt
  extract components; annotate top-8 on image with #id labels;
  highlight agent cell with "YOU"
  build USER text: obs fields, cell system, action effects in cell
  deltas, components, probe results (first turn on level), prior
  hypotheses, tried_targets (session-local blocked + this-session
  reached-inert), last-3 history with CHANGES narration
  attach image as base64 PNG

  # TUTOR call
  rsp = call_anthropic(SYSTEM_DISCOVERY, user, image)
  parse {rationale, hypotheses, command, args{target_cell}, predict,
         revise}

  # Execute via BFS in cell space
  path = _bfs_plan_cells(agent_cell, target_cell, cell_actions,
                          walls_cells, cell_bounds, max_steps=60)
  execute up to _MAX_TOTAL_STEPS=15 env.steps with wall-hit rerouting
  on each env.step:
      if move failed (cell unchanged): walls_cells.add(cell, action)
      if move succeeded: walls_cells.discard(cell, action)
                         successful_moves.add(cell, action)

  # Record
  update history with {target, reached, cur_cell, lc_before, lc_after,
                        frame_diff_cells, delta, cost_usd}
  update distillation record with retrospective outcome

  # Record completed level
  if lc_after > lc_before:
      if shortest path so far: save to solutions.json with frame_hash
      cur_level_steps = []   # reset for next level
```

### 5.3 Session wrap-up

1. Compute session-local `session_blocked_cells` (targets never reached).
2. Merge walls into KB via the 2-strike rule:
   - Drop any wall `(c, a)` in `successful_moves` (invalidation).
   - Add walls hit ≥ 2 times this session (confirmation).
   - Preserve existing KB walls hit ≥ 1 time this session (re-confirmation).
3. Save KB, persist TUTOR's accumulated `hypotheses_by_level`.
4. Write session manifest + distillation turn records.
5. Mirror both dirs to `latest_strict/` for stable review paths.
6. Render `trace.html` inside `latest_strict/`.


## 6. Knowledge Base contract

File: `benchmarks/knowledge_base/<game_id>_runtime.json`

```json
{
  "game_id": "ls20-9607627b",
  "action_effects_learned": { "ACTION1": [-5, 0], ... },
  "agent_fingerprint": [palette, size, [h, w]],
  "cell_system": { "cell_size": 5, "origin": [r, c], "level_key": "0" },
  "walls_by_level_cells": { "0": [[cr, cc, "ACTIONn"], ...], ... },
  "blocked_targets_by_level_cells": {},      // always empty; session-local
  "hypotheses_by_level": { "0": ["(turn N, L0): ...", ...] },
  "last_updated": "ISO-8601"
}
```

File: `benchmarks/knowledge_base/<game_id>_strict_solutions.json`

```json
{
  "game_id": "...",
  "levels": {
    "0": {
      "game_steps": ["ACTION3", "ACTION3", ...],
      "step_count": 13,
      "frame_hash_on_entry": "cfe5196fb75182bb",
      "solver": "strict_mode_tutor",
      "solved_at": "ISO-8601",
      "session": "trial_..._strict"
    }
  }
}
```

**Persistence rules**:
- Walls persist when confirmed ≥ 2× in a single session, or already
  in KB and hit ≥ 1× this session (re-confirmation).
- Walls invalidate when a successful move is observed through them.
- Blocked-targets do NOT persist (they change as walls change).
- Hypotheses persist (append-only, capped at 40 latest).
- Solutions persist once, replaced only by shorter solutions.


## 7. Data outputs

### 7.1 Training corpus (distillation)

Path: `.tmp/training_data/<game_id>/trial_<ts>_strict/` and
mirrored at `.tmp/training_data/<game_id>/latest_strict/`.

- `metadata.json` — session summary (outcome, turns, cost, action effects).
- `turn_NNN.json` — one per TUTOR call with `{system, user,
  assistant, frame_b64, metadata}`.  `metadata` includes
  `advanced_level`, `target_reached`, `frame_diff_cells`,
  `delta_summary`, parsed fields, cost/tokens/latency.
- Both successful AND failed turns are preserved; distillation
  needs both.

### 7.2 Session logs

Path: `benchmarks/sessions/trial_<ts>_strict/` and mirrored at
`benchmarks/sessions/latest_strict/`.

- `play_log.jsonl` — per-turn log (target, rationale, path, reached, lc, diff)
- `manifest.json` — session summary
- `trace.html` — auto-rendered readable view (see 7.3)

### 7.3 HTML trace viewer (`render_strict_trace.py`)

Generates a single file `trace.html` showing per-turn cards:
- Annotated frame image (cell grid, agent "YOU" highlight, component labels #id)
- Color-coded strip: green (level advanced), yellow (reached target no advance), red (miss)
- TUTOR's rationale / hypotheses / predict / revise in code blocks
- Metadata bar: cost, tokens, latency, frame diff, delta summary
- Top summary: outcome, total cost, action effects, LC progression

Stable path the user can reload in any browser:

```
file:///C:/_backup/github/khub-knowledge-fabric/.tmp/training_data/ls20-9607627b/latest_strict/trace.html
```


## 8. Cost controls

| Control | Default |
|---|---|
| Hard session cost cap (stops before next TUTOR call) | $0.10 |
| Max TUTOR turns per session                           | 6 |
| Max tokens per TUTOR response                          | 1200 |
| Max env.steps per MOVE_TO command                      | 15 |
| Per-session new-wall recording cap                     | 18 |
| Running cost printed each turn                         | on |

Overrides via CLI flags: `--cost-cap 0.05`, `--max-turns 10`,
`--max-tokens 2000`.


## 9. Design rules

1. **Game-agnostic substrate.** Any harness mechanism must pass the
   litmus test in §2.  Cell size is inferred; wall palettes are
   discovered; agent sprite is identified by motion.

2. **TUTOR = perception + hypothesis; harness = mechanics.** Never
   ask TUTOR to do what a BFS can do.  Never ask the harness to
   guess what a component means.

3. **Discovery > assertion.** When something is ambiguous, the
   correct response is an EXPERIMENT that narrows the hypothesis
   space, not a commitment to a guess.  Auto-probe embodies this
   at the harness level.

4. **Observation before interpretation.** The CHANGES narration is
   raw (components disappeared / appeared / moved).  TUTOR assigns
   meaning, not the harness.

5. **Reversible wall learning.** A wall recorded once is tentative;
   a confirmed wall invalidates on observed success.  Single
   failures do not persist across sessions.

6. **Cross-session memory persists SELECTIVELY.** Walls persist
   only when confirmed.  Hypotheses persist as append-only log.
   Blocked-targets do NOT persist (they mean "unreachable with
   current walls" which is session-dependent).

7. **Solutions replay silently.** Once a level is solved, future
   sessions replay the recorded action sequence with no LLM cost.

8. **Data-driven review.** Every TUTOR call is captured for
   distillation; every session produces an HTML trace you can
   open without running code.

9. **Hard cost cap ALWAYS.** No session is allowed to spend more
   than the configured cap without explicit opt-in.


## 10. Current state & known gaps (as of 2026-04-21)

### What works under strict mode

- **1/7 solved**: 2 turns, $0.033 first time; free replay thereafter.
  L0 cross at cell `(-3, -3)` identified visually by TUTOR as
  "white cross"; level completes with the 13-step canonical solve.
- **Bootstrap**: action effects + agent fingerprint discovered
  deterministically from 4 env.steps.
- **Auto-probe**: L0 probe reaches the cross; L1 probes confirm
  the cross at `(1, 4)` is NOT directly reachable from spawn.
- **Cell system**: eliminates pixel-vs-sprite-bbox ambiguity that
  plagued earlier iterations.  TUTOR reasons in cells cleanly.
- **Wall confidence**: walls_learned per session dropped from 45+
  (pre-fix, poisoning BFS) to ~4 (post-fix, clean).
- **Hypothesis persistence**: TUTOR correctly identified
  "bottom yellow bar = move budget", "yellow squares = pickups",
  "black cross = goal" (though the cross is actually a trigger
  not the goal).
- **Trace viewer**: every session produces a readable `trace.html`
  with frame images + reasoning + outcomes.

### What doesn't work yet

- **2/7 not solved under strict mode.** Across ~10 sessions (~$2 total),
  no completion.  The blocker: the L1 cross at cell `(1, 4)` is
  not reachable from spawn with currently-confirmed walls; the
  legacy canonical path routes UP to row ~10 then RIGHT to col 51
  then DOWN (17+ cells).  BFS needs enough wall knowledge to
  discover this detour, and walls accumulate slowly (0–4 confirmed
  per session under the 2-strike rule).
- **TUTOR's cross hypothesis is wrong.** It consistently labels the
  palette-0 black cross as "goal" rather than "rotation trigger".
  It hasn't discovered the 3-visit pattern.  Once agent can reach
  the cross, a revision is likely; the gate is reachability.
- **Budget awareness is declarative, not actionable.** TUTOR has
  hypothesized that the yellow bar is the move budget, but there's
  no explicit per-turn readout of "current budget".  When budget
  depletes mid-turn and the level resets (diff_cells ~4000+),
  TUTOR only notices via the outcome.

### Known regression observed in the latest trace

The latest session's trace shows TUTOR bouncing between targets,
occasionally triggering a level reset (`lc` regression 1→0 with
`diff_cells=4093` on long paths).  Multiple causes compound:

1. `_MAX_TOTAL_STEPS = 15` per command is a hard cap; but BFS can
   still emit 15-step paths when a distant target is chosen, and
   executing all 15 at 2-budget-per-step consumes 30 of the 42
   budget in one turn.  One more long attempt drains to 0.
2. TUTOR commits to distant distinctive targets (cross at (1,4),
   pickup at (-5,-3)) before walls are known; BFS reroutes into
   dead ends.
3. Auto-probe consumes budget before TUTOR's first turn.  3 probes
   × up to 10 steps each = up to 30 budget used on probing.
   Combined with #1 this effectively puts L1 TUTOR on a tight
   budget leash from the start.


## 11. Open problems and candidate next levers

1. **Expose a budget proxy in the prompt.**  Track the size of the
   palette-11 horizontal HUD component each turn; include a
   "budget_bar_size" field in the prompt so TUTOR can reason about
   how many actions it has left.  Currently hypothesized but not
   actionable.

2. **Dynamic step cap tied to observed budget.**  If the budget
   proxy shrank by N per action in previous turns, cap the next
   command's env.steps to `(current_budget / N) - safety_margin`.

3. **Auto-probe should skip probes that would be too expensive.**
   A probe of a distant target (>5 cells) that will likely fail
   costs budget for near-zero information.  Prefer probes of
   components within 2 cells first.

4. **Trigger multi-visit discovery.**  After reaching a component,
   if CHANGES include a counter-like decrement (or a rotation
   indicator change), mark it as "trigger candidate" and instruct
   TUTOR to re-visit.

5. **Prompt correction on the "goal" misidentification.**  The
   black cross is a trigger, not a goal.  We can't tell TUTOR that
   (it'd be injection), but we can soften the prompt: distinguish
   "visually goal-shaped" from "where the level completes."

6. **Post-game KB consolidation via a cheap Claude call.**  After
   each session, a brief Haiku-grade call that takes the manifest
   + top hypotheses and writes a 1-paragraph "what we learned"
   note.  Loaded at next session's prompt startup.  Cost <$0.01
   per session.

7. **Action batching.**  TUTOR could output a sequence of 2–3
   targets per turn, executed in order.  Amortizes the LLM call
   cost.  Risks: staleness of intermediate-state decisions.


## 12. How to run / review

```bash
# Strict-mode play session (default cost cap $0.10, max_turns 6)
python discovery_play.py --game ls20-9607627b

# More turns / higher cap
python discovery_play.py --game ls20-9607627b --max-turns 10 --cost-cap 0.25

# Disable auto-replay (rerun from scratch each time)
python discovery_play.py --game ls20-9607627b --no-replay-solved

# Render a specific training-data dir as HTML (auto-invoked, but
# can be called manually on any dir of turn_NNN.json files)
python render_strict_trace.py /path/to/training_data_dir

# Legacy comparison (uses privileged env._game access -- not
# competition-legal, used only to benchmark)
python discovery_play.py --game ls20-9607627b --legacy
```

**Review paths (stable, no timestamp)**:

```
C:\_backup\github\khub-knowledge-fabric\.tmp\training_data\ls20-9607627b\latest_strict\trace.html
C:\_backup\github\khub-knowledge-fabric\.tmp\training_data\ls20-9607627b\latest_strict\metadata.json
C:\_backup\github\khub-knowledge-fabric\usecases\arc-agi-3-dd\benchmarks\sessions\latest_strict\play_log.jsonl
C:\_backup\github\khub-knowledge-fabric\usecases\arc-agi-3-dd\benchmarks\knowledge_base\ls20-9607627b_runtime.json
C:\_backup\github\khub-knowledge-fabric\usecases\arc-agi-3-dd\benchmarks\knowledge_base\ls20-9607627b_strict_solutions.json
```
