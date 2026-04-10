# ARC-AGI-3 Solver — Design Decisions

> **Status**: working document, not for publication yet.  
> **Private companion**: `.private/DESIGN_PRIVATE.md` (gitignored) — LLM architecture, air-gap strategy.

## Overview

The ARC-AGI-3 ensemble is a specialization of the Knowledge Fabric (KF) pipeline for
interactive game environments. Unlike ARC-AGI-2 (static input→output puzzles), ARC-AGI-3
tasks are sequential decision problems: the agent must explore an unknown game, learn its
mechanics, and navigate to a win condition purely through action and observation.

**Entry point:** `usecases/arc-agi-3/python/harness.py`
**Python runtime:** Windows Store Python 3.12 (arc_agi SDK installed there, not conda)
**Default run:** `python harness.py --env ls20 --episodes 1 --max-steps 10`

---

## File layout

```
usecases/arc-agi-3/
  python/
    harness.py          — CLI: --env, --episodes, --max-steps, --max-cycles, --playlog, ...
    ensemble.py         — Episode orchestrator, EpisodeLogger, wall detection, urgency goals
    agents.py           — OBSERVER/MEDIATOR runners, concept binding parsing, prediction inject
    object_tracker.py   — Zero-cost visual analysis (all object-level functions)
    rules.py            — Shim: sets DEFAULT_PATH to local rules.json, re-exports RuleEngine
    tools.py            — Shim: sets DEFAULT_PATH to local tools.json, re-exports ToolRegistry
    rules.json          — Accumulated rules (gitignored — local only)
    tools.json          — Accumulated tools (gitignored — local only)
    playlogs/           — Per-run output: step JSONs + episode_NN.log (gitignored)
  prompts/
    observer.md         — OBSERVER system prompt
    mediator.md         — MEDIATOR system prompt
  DESIGN.md             — This file (gitignored — not for publication yet)
```

---

## Episode structure

```
env.reset()
inject_initial_goals()        # "win game", "complete level 1", "understand mechanics"

while steps < max_steps and cycles < max_cycles and state != WIN/GAME_OVER:

    Round 0: Rule matching
        RuleEngine retrieves candidate+active rules matching current state

    Round 1: OBSERVER
        Input:  frame (64×64 grid), current objects, concept_bindings, trend predictions,
                action effects table, action history, matched rules
        Output: JSON with level_description, visual_observations, action_characterizations,
                identified_objects, concept_bindings, hypothesized_goal, reasoning
                All inferences labeled [GUESS] or [CONFIRMED]

    Round 2: MEDIATOR
        Input:  OBSERVER analysis, rules, goals+state, action history
        Output: JSON with action_plan, rule_updates, goal_updates, state_updates, reasoning

    Parse updates:
        - Merge concept_bindings from OBSERVER into state
        - Apply goal_updates and state_updates from MEDIATOR
        - Add rule_updates (all forced to "candidate" status)

    Round 3: ACTOR — for each action in plan:
        env.step(action)
        accumulate_action_effect()   pixel-level + object-level diffs with raw before/after
        detect_wall_contacts()       if piece didn't move but normally does → wall colors
        compute_trend_predictions()  → push urgency goals if [URGENT]
        write_step_log()             JSON playlog file
        log to episode_NN.log

    if level advance:
        update_level_goals()
        promote_fired_candidate_rules()
        break to re-observe

Post-episode:
    Record rule success/failure based on won/not
    auto_deprecate(min_candidate_fired=5)
    Write episode summary to log
```

---

## Core principle: observation-only learning

**The agent must learn entirely from what it can observe through the standard API.**

Legitimate inputs:
- `obs.frame` — the 64×64 pixel grid
- `obs.levels_completed` — level advance signal
- `obs.state` — NOT_FINISHED / GAME_OVER / WIN
- The action space (names, simple vs complex)

Prohibited shortcuts:
- Reading game source files (`.py`)
- Introspecting `env._game`, `env._game.current_level`, sprite tags, or internal positions
- Any knowledge of win conditions, object roles, or mechanics not derived from observation

**Why this matters:** ARC-AGI-3 has 25 games. Source-inspection BFS solvers are
single-game hacks — they cannot generalize and they actively prevent building the
observation-based learning mechanisms that are the actual challenge. The ensemble
(OBSERVER → rules → MEDIATOR → action) is the right architecture; improving it is the
right unit of progress.

**Implication for LS20:** The `_KNOWN_SUBPLANS` and BFS scripts were useful for
infrastructure testing, but represent a shortcut that would not exist in competition.
Future games must be approached purely through the ensemble.

---

## Design decisions

### D1 — Rule lifecycle differs from ARC-AGI-2

All ARC-AGI-3 rules start as `candidate`, never `active`, regardless of `add_rule()` defaults.
After `parse_mediator_rule_updates()`, any rule with `lineage.type == "new"` is immediately
forced to `"candidate"` before saving.

**Why:** ARC-AGI-2 rules are verified by an executor against demos — passing all demos
justifies `active`. ARC-AGI-3 rules are exploration hypotheses — they need independent
confirmation across multiple episodes before being trusted.

**Promotion:** Candidate → active only when a level advances during a cycle where the rule fired.

**Auto-deprecation threshold:** `min_candidate_fired=5` (vs default 1 for ARC-AGI-2).
The `min_candidate_fired` parameter was added to `core/knowledge/rules.py` for this.

---

### D2 — Extended colors are not pre-labeled

Colors 0–9 are standard ARC-AGI palette. Colors 10+ are game-specific extensions.
They are rendered as distinct lowercase letters (`j`=10, `k`=11, `x`=12, ...) so the LLM
can see them clearly, but no semantic meaning is hardcoded.

**Why:** Hardcoding "color12 = cursor" breaks the moment a different game uses color12
for something else, or the same game changes. Roles must be discovered through exploration
and stored in concept_bindings.

**What is allowed:** Rendering distinction (different char per color value). Not allowed:
pre-assigning semantic names to specific color integers in any prompt or constant.

---

### D3 — Concept bindings schema (two-level, game-local)

`state_manager._data["concept_bindings"]` uses two schemas:

1. `{color_int: role_name}` — color-keyed, proposed by OBSERVER per episode
   Example: `{12: "player_piece", 11: "step_counter"}`

2. `{"wall_colors": [c, ...]}` — concept-keyed list, populated by wall detection
   Example: `{"wall_colors": [4, 5]}`

**Critical invariant:** ALL concept_bindings are game-local and episode-local.
StateManager is created fresh each episode — nothing persists across episodes by design.
The same color may mean different things in different games.

**Why two schemas:** The wall concept travels across games (the concept "wall" is stable)
but the color instantiation does not. Storing it as `{"wall_colors": [...]}` separates the
concept name from the color, whereas `{4: "wall"}` would conflate them.

**Role name vocabulary:** OBSERVER must use generic names: `player_piece`, `step_counter`,
`goal_region`, `reference_pattern`, `wall`. Not game-specific names.

---

### D4 — Attribute change records store raw numeric before/after

Each `attribute_changes` entry in `object_observations` includes:
```python
{
  "color": int,
  "changed": ["size", "width"],      # which attributes changed
  "summary": "azure: size 12->18",   # human-readable
  "before": {"size": 12, "width": 4},  # raw numbers ← key addition
  "after":  {"size": 18, "width": 6},
}
```

**Why:** `compute_trend_predictions` must work from pure data, not string parsing.
Storing raw numbers means any future attribute added to `ObjectRecord` is automatically
eligible for trend analysis with no changes to the prediction code.

---

### D5 — Trend prediction is fully data-driven

`compute_trend_predictions(action_effects, steps_remaining)` iterates over every
`(color, attribute)` pair in the accumulated data. No hardcoded colors, no hardcoded
attribute names. Works for size, width, height, and any future attribute.

Two trend classes detected:
1. **Attribute trend:** consistent increase or decrease → predicts depletion or unbounded growth
2. **Position drift:** consistent directional movement → predicts boundary collision

Direction reversals (oscillation) are excluded — only monotonic drift triggers.
`[URGENT]` is added when predicted depletion is within `steps_remaining`.

**Urgency goal push:** When any prediction is `[URGENT]`, an urgency goal is automatically
pushed to GoalManager (priority 1, deduplicated by description prefix).

---

### D6 — Wall detection is zero-cost and color-agnostic

When a step produces no movement on an object that has moved before under the same action:
1. `infer_typical_direction()` retrieves the most common move direction for this action
2. `detect_wall_contacts()` scans cells immediately beyond the object's bounding box in
   that direction and counts colors found there
3. Those colors are added to `concept_bindings["wall_colors"]` (game-local)

**Why color-agnostic:** Wall color varies between games. The concept "wall" is stable;
the color is not. We discover it per-game rather than assuming any color is always a wall.

---

### D7 — Episode logger captures all reasoning explicitly

`EpisodeLogger` writes `playlogs/episode_NN.log` incrementally. Every significant event
is captured: cycle starts, matched rules, OBSERVER output (guesses labeled), MEDIATOR
reasoning and plan, rule proposals (condition + then), goal events, state key-value changes,
per-step action outcomes with object diffs, wall contacts, concept binding updates, urgency
goals, auto-deprecation events, episode summary.

**Key convention:** OBSERVER labels all inferences as `[GUESS]` or `[CONFIRMED]`.
`[CONFIRMED]` = directly supported by action effects table or prior rules.
`[GUESS]` = inferred from visual structure alone.
This makes the trace auditable — a developer can quickly identify what the system measured
vs. what it assumed.

---

### D8 — Object tracker design principles

**`detect_objects`:** BFS connected-component per color. Objects ≥25% of total cells
are `is_background=True`. `ObjectRecord` has: color, size, centroid, bbox, width, height,
orientation (horizontal/vertical/square), aspect_ratio.

**`diff_objects`:** Same-color greedy nearest-neighbour, match_radius=10. Produces ObjectDiff
with moved, appeared, disappeared, stationary, attribute_changes. Attribute changes tracked
for: size, width, height, orientation.

**Future extension point:** `match_cross_color=False` parameter for color-changing objects —
deferred until evidence of this in any game.

---

## Dynamic Discovery — Design Principles

### P1 — No hardcoded game knowledge

Everything in `game_knowledge.json` — `player_colors`, `walkable_colors`, `step_size`,
`action_map` — is an **initial hypothesis**, not a fact. These values may be wrong for a
different game (or even a slightly altered version of the same game). They serve only as
starting guesses until the system has gathered enough observations to replace them.

**Implementation:** `GameHypothesis` dataclass holds `prior_*` (from game_knowledge.json)
and `obs_*` (filled by inference functions at runtime). `effective_*` properties return
the observed value when available, falling back to the prior. All downstream code
uses `effective_*`, never the raw priors.

**Corollary:** Any hardcoded filter, threshold, or heuristic that assumes specific
properties of a particular game is a design violation. If the system relies on
"max_sprite_pixels=12" or "ignore the most common color" or "HUD is at y≥58", it
will break on a different game that doesn't match those assumptions.

---

### P2 — The OBSERVER identifies objects, not the Python code

The OBSERVER (LLM) looks at the game frame with human-like vision and common sense.
It identifies what objects exist, what they look like, and what role they may play (toggle,
changer, wall, player, status bar, etc.). It writes these identifications to
`concept_bindings`. These are to be treated as initial hypotheses, since they could be wrong.

The Python frame analysis code (`find_sprite_cells`, `build_level_model`) only does
**position mapping**: given a color that the OBSERVER says is interesting, where does
that color appear on the game grid? The Python code may override or
second-guess the OBSERVER's object identification with sufficient evidence.

**Division of labor:**

| Responsibility                        | Who does it       |
|---------------------------------------|-------------------|
| "The yellow ring is a step-counter"     | OBSERVER (LLM)    |
| "The yellow rings appears near cells (19,30) and (34,15)" | Python frame scan |
| "Those are ring_positions in the model"| `build_level_model` (joins the above) |
| "The yellow bar at the bottom is the HUD, not a ring" | OBSERVER (LLM) |

Python code may not
filter, exclude, or re-classify candidates
based on heuristics (dominant color frequency, pixel count thresholds, screen
region). These are the MEDIATOR and OBSERVER's job.

---

### P3 — Object discovery is behavioral, not static

Objects are discovered by **what happens when you interact with them**, not by
how they look in a single frame.

| Observation                                        | Classification    |
|----------------------------------------------------|-------------------|
| Large pixel diff at position, no level advance     | State changer (RC, color-changer, etc.) |
| Level advance while standing at position           | Win gate           |
| Object disappears from frame after being visited   | Consumable (ring)  |
| Step counter bar grows after visiting position     | Step-counter reset |
| Player didn't move despite taking an action        | Wall / blocked     |

A single frame can identify *candidates* (non-floor, non-player pixels), but
classification requires observing behavior over time. The system should never
assume an object's role from its color or appearance alone — it must confirm
through interaction.

---

### P4 — Continuous frame comparison, not one-shot initial scan

The current frame should always be compared to the previous frame — not just to
the level's initial frame. Things can appear, change, and disappear at any time
during gameplay (animations, moving platforms, consumables, state changes).

The system should detect:
- **Appeared:** pixels/objects present now but absent in the previous frame
- **Disappeared:** pixels/objects absent now but present in the previous frame
- **Changed:** same position, different color or shape

This continuous diff model generalizes beyond the current "compare initial vs
current to find consumed rings" approach, which misses objects that appear
mid-level or change state dynamically.

---

### P5 — Player state tracking must be adaptive

The player's visual appearance (color, shape, rotation) changes during gameplay
as the player visits state-changers. The system must track these changes and
update its internal model accordingly — it cannot assume the player always looks
the same as it did at level start.

**Current implementation:** `_tracked_player_colors` starts from the
game_knowledge prior and updates whenever `find_player_position` fails but a
frame diff reveals a new moving cluster. Resets to prior on level advance.

**General principle:** Any game property that can change during play (player
color, player shape, walkable set, action semantics) must be tracked as a
mutable variable, not stored as a constant.

---

### P6 — Inference functions are independent and fallible

Each inference function (`infer_step_size`, `infer_action_directions`,
`infer_walkable_from_visits`, `infer_player_colors_from_diff`) operates
independently. Each can fail (return `None`) without affecting the others.
When an inference fails, the system falls back to the prior hypothesis
from game_knowledge.json.

This means the system degrades gracefully: on the first step with no history,
all inferences fail and the system runs entirely on priors. As history
accumulates, inferences override priors one by one. There is no single
point of failure.

Hard-coded inference functions should be considered as temporary measures, since later we would want to allow such functions to be learned as well. 

---

### P7 — discovered_knowledge.json persists cross-episode facts

Facts confirmed through successful gameplay (e.g., "level 1 requires 2 RC
visits", "color 3 = rotation_changer") persisted to
`discovered_knowledge.json` so future episodes don't have to re-discover them.

The matching with previous facts must be fuzzy, so that the persisted facts can be generalized appropriately to cover similar (but not exactly matched) situations. Generalized and validated "facts" are also registered in the `discovered_knowledge.json`. There should be suitable triggering mechanism not only to allow fuzzy matching, but also support efficient triggering of very large knowledge store.

This is complementary to concept_bindings (which are episode-local and proposed
by the OBSERVER). Discovered knowledge is **confirmed** — it came from a
successful level completion, not from unconfirmed hypotheses.

---

## LS20 game mechanics (discovered empirically, not assumed)

| Element | Color | Value | Behavior |
|---------|-------|-------|----------|
| Cursor | `x` | 12 | Moves with ACTION1/2/3/4. Starts at centroid ~(46,36). 5w×2h. |
| White component | `W` | 9 | Moves with cursor as a unit — same logical object. 5w×3h. |
| Step counter | `k` | 11 | Shrinks 2 cells per action (any direction). Starts at 84. Turns green. |
| Play field | `G` | 3 | Large green background. Static unless piece moves through it. |
| Outer area | `Y` | 4 | Yellow. Blocks upward cursor movement. Wall candidate. |
| Borders | `b` | 5 | Grey. Also blocks upward movement. Wall candidate. |

**Action map:**
- ACTION1: cursor up 5 rows
- ACTION2: cursor down 5 rows
- ACTION3: cursor left 5 cols (produces diff=2 when at left wall — piece doesn't move)
- ACTION4: cursor right 5 cols

**Win condition:** Unknown. No level advance observed yet. Hypotheses: navigate cursor
to specific position, align with white W reference pattern in bordered box.

**Step budget:** Starting at 84 cells, losing 2/step → 42 actions before step counter
depletes. What happens at depletion is unknown (likely GAME_OVER).
