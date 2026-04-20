"""PLAY prompts — drive TUTOR through sequential ls20 L1 actions.

Given the model's working understanding (from Round-2 revised assessment),
a frame, and the CHANGE_REPORT from the last action, ask for ONE next
action plus a short rationale.  Model must reply with strict JSON so the
harness can act on it.
"""
from __future__ import annotations

import json


SYSTEM_PLAY = """You are a visual reasoning subsystem for a symbolic game-playing engine,
now in PLAY mode.  You have already analysed the initial frame (see
WORKING_KNOWLEDGE).  Each turn you receive the current frame, the
CHANGE_REPORT describing what changed since your last action, and a
short recent-history summary.  Your job is to pick ONE next action.

EVIDENCE PRECEDENCE: CHANGE_REPORT from the live environment is ground
truth.  WORKING_KNOWLEDGE is your prior theory; any prior_knowledge inside
it comes from other games and can be wrong for this specific level.
When CHANGE_REPORT contradicts WORKING_KNOWLEDGE or prior_knowledge,
TRUST CHANGE_REPORT and update your theory via revise_knowledge.  Do
NOT second-guess evidence you have just seen.

CHANGE_REPORT FIELDS:
  primary_motion -- the smallest reliably-tracked element that actually
       moved this turn.  In most games this IS the agent/cursor.  Use
       its pre_bbox -> post_bbox as your primary ground truth for what
       the last action did.
  element_motions -- ALL tracked elements.  Each carries a flag
       "tracker_unreliable": true when the element's declared bbox is
       huge (tracker can latch onto a smaller moving thing inside) or
       pre/post areas differ by >10x.  IGNORE unreliable motions; they
       are tracker artefacts, not the game.
  counter_changes -- per-counter/readout fill before/after + direction.
  appearances / disappearances -- novel-colour components or vanished
       tracked elements.
  unexplained_regions -- clustered residual changed cells the tracker
       could NOT attribute.  before_patch / after_patch let you see the
       raw cells.

Decision priority:
  1. If primary_motion is set and moved as you predicted, proceed.
  2. If primary_motion contradicts your prediction, update theory.
  3. If primary_motion is null but diff_cells > 0, inspect appearances
     and unexplained_regions.

Return a single JSON object with keys:
  "action":    one of AVAILABLE_ACTIONS (e.g. "ACTION1"), or "RESET"
  "action_sequence": optional list of 1..5 actions (including the one in
               "action", which should match the first entry).  Use this
               ONLY when you are confident about the next several steps
               (e.g., "I need to press ACTION1 three more times to reach
               the target row").  The harness will execute them back-to-
               back without calling you, then show you the resulting
               CHANGE_REPORT in one shot.  If any step has a meaningful
               branch / uncertainty, keep the sequence short.
  "rationale": one or two sentences on why this action (or sequence),
               what you expect CHANGE_REPORT to show
  "predict":   short object with your expected observation, e.g.
               {{ "agent_dr": -5, "agent_dc": 0 }} or
               {{ "counter_direction": "-" }} -- omit keys you don't
               predict.  If an action's prediction does not match, flag
               it next turn in "revise_knowledge".
  "revise_knowledge": optional string -- if the last CHANGE_REPORT
               contradicted your model, tersely say what you now think
               is different.  Empty string if no revision.
  "done":     true | false -- set true ONLY when STATE is WIN or
               GAME_OVER and no further action makes sense.

Reply with strict JSON only.  No prose, no code fences."""


USER_TEMPLATE = """PLAY TURN {turn}

GAME: {game_id}  STATE: {state}  LEVELS: {levels_completed}/{win_levels}
AVAILABLE_ACTIONS: {action_labels}

WORKING_KNOWLEDGE (your current theory of the game):
{working_knowledge}

RECENT_HISTORY (last {history_n} actions):
{recent_history}

LAST_CHANGE_REPORT (harness semantic summary of what happened since your
previous action; empty on turn 1):
{change_report_json}

CURRENT_FRAME (64x64 grid, palette 0-15):
{frame_text}

Pick ONE next action.  Reply with the JSON schema in the system message."""


def build_play_user_message(
    *,
    turn:              int,
    game_id:           str,
    state:             str,
    levels_completed:  int,
    win_levels:        int,
    action_labels:     list[str],
    working_knowledge: str,
    recent_history:    list[dict],
    change_report:     dict | None,
    frame_text:        str,
) -> str:
    hist_lines = []
    for h in recent_history[-5:]:
        hist_lines.append(
            f"  turn {h.get('turn')}: {h.get('action')} state={h.get('state')} "
            f"primary_motion={h.get('primary_motion','?')} "
            f"counter={h.get('counter_summary','?')}"
        )
    if not hist_lines:
        hist_lines = ["  (none)"]
    return USER_TEMPLATE.format(
        turn              = turn,
        game_id           = game_id,
        state             = state,
        levels_completed  = levels_completed,
        win_levels        = win_levels,
        action_labels     = action_labels,
        working_knowledge = working_knowledge,
        history_n         = len(recent_history[-5:]),
        recent_history    = "\n".join(hist_lines),
        change_report_json = json.dumps(change_report or {}, indent=2),
        frame_text        = frame_text,
    )


SYSTEM_POSTGAME = """You just finished a session of an ARC-AGI-3 game.  The outcome is in
OUTCOME.  Write a short, dense knowledge note for FUTURE plays of this
game (or its siblings in the ls-series).  Be concrete: name elements,
coordinates, action-effects, win-condition details, mistakes to avoid.

Prose is fine but keep it tight -- less than 300 words.  No fences."""


POSTGAME_TEMPLATE = """POST-GAME KNOWLEDGE CAPTURE

GAME: {game_id}
OUTCOME: {outcome}   ({turns} turns, final_state={final_state}, levels={levels_completed}/{win_levels})

WORKING_KNOWLEDGE at end of play:
{working_knowledge}

ACTION_TRACE (turn -> action -> brief):
{action_trace}

Write a single knowledge note (< 300 words) that the next play session
(or a sibling ls-game) should receive as PRIOR_KNOWLEDGE.  Cover:
  - what each action does (if known)
  - what each named element is / does
  - the win condition and any scored counter semantics
  - traps / anti-patterns you hit this run
  - 1-2 speculative hypotheses still worth testing

No fences, no headers, just the note."""


def build_postgame_user_message(
    *,
    game_id:           str,
    outcome:           str,
    turns:             int,
    final_state:       str,
    levels_completed:  int,
    win_levels:        int,
    working_knowledge: str,
    action_trace:      list[dict],
) -> str:
    trace_lines = []
    for h in action_trace:
        brief = h.get("rationale") or ""
        brief = brief[:80] + ("…" if len(brief) > 80 else "")
        trace_lines.append(
            f"  {h.get('turn'):>3}: {h.get('action'):<10} — {brief}"
        )
    return POSTGAME_TEMPLATE.format(
        game_id           = game_id,
        outcome           = outcome,
        turns             = turns,
        final_state       = final_state,
        levels_completed  = levels_completed,
        win_levels        = win_levels,
        working_knowledge = working_knowledge,
        action_trace      = "\n".join(trace_lines) if trace_lines else "  (empty)",
    )
