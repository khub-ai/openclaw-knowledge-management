# Trial 20260420T042204Z — Round-2B diagnostic

- TUTOR: `claude-sonnet-4-6`
- PUPIL: `google/gemma-4-26b-a4b-it`

**Prior knowledge injected:**

```
Prior knowledge from previous plays of the ls* game family (ls20, ls21, ...).

Treat these as strong priors — learned from other levels and other ls-series
games — but still subject to per-level evidence from the grounding pack.

1. The white cross (color 0 pixel cluster in the main floor area) is NOT
   the agent.  In ls-family games it is a "change indicator": it marks the
   cell where the next state transition will happen when an action fires.

2. ACTION1..ACTION4 are NOT directional movement keys.  They are
   "state-mutation" actions: each one attempts to change a property of the
   cell under the cross (or a neighbouring cell) — for example, toggling a
   floor tile, activating a switch, or imprinting a glyph.

3. The yellow horizontal bar at the bottom of the frame is a PROGRESS
   counter.  The level win condition is to fill it completely, usually by
   making the playing-field state match a reference pattern encoded
   somewhere on the frame.

4. The three red squares at the bottom-right are REMAINING ATTEMPTS (lives).
   Each failed episode reduces them by one.

5. Small bordered boxes with an internal glyph (here: the top-centre
   bordered room and the bottom-left corner icon) are USUALLY
   "reference glyphs" — they encode the target state the player must
   recreate on the playing field.  They often appear in matched PAIRS:
   one serves as the reference, the other as a write-target.

6. A win in ls-family games typically requires recreating the reference
   glyph (item 5) on the playing field using the state-mutation actions
   (item 2), tracked by the progress bar (item 3).

These priors come from other levels — they may still be wrong for this
specific level.  If the grounding pack directly contradicts any of them,
trust the grounding pack.
```

## 1. Tier-A findings (coordinates & visual grounding)

### TUTOR
- _none flagged_

### PUPIL
- element 5 `gap_left` bbox=[0, 0, 63, 63]: bbox covers 4096/4096 cells — implausibly large for a discrete element

## 2. Evidence contradictions (Round 1 → probe facts)

### TUTOR
- _none detected by current heuristics_

### PUPIL
- _none detected by current heuristics_

## 3. Tier-B guess changes (Round 1 → Round 2)

### TUTOR
- element 1: agent → target
- element 2: agent → switch
- element 6: collectible → agent
- element 7: unknown → target
- element 9: readout → counter
- element 3: retracted (was wall)
- element 5: retracted (was unknown)
- element 12: retracted (was portal)
- element 13: retracted (was wall)
- element 14: retracted (was unknown)
- element 15: added (now hazard)
- element 16: added (now unknown)

### PUPIL
- _no function revisions_

## 4. Disagreements after evidence (arbitration candidates)

- **element function** at IoU=0.44: TUTOR(2 `change_indicator`)=switch vs PUPIL(2 `player_cursor`)=unknown
- **element function** at IoU=0.6: TUTOR(4 `main_floor_area`)=wall vs PUPIL(14 `left_sub_room`)=unknown
- **element function** at IoU=0.67: TUTOR(6 `agent_cursor`)=agent vs PUPIL(4 `main_floor_area`)=unknown
- **element function** at IoU=0.67: TUTOR(6 `agent_cursor`)=agent vs PUPIL(6 `blue_orange_block`)=collectible
- **element function** at IoU=0.67: TUTOR(6 `agent_cursor`)=agent vs PUPIL(13 `right_corridor_section`)=wall
- **element function** at IoU=1.0: TUTOR(8 `progress_bar`)=readout vs PUPIL(11 `progress_counter`)=counter
- **element function** at IoU=0.4: TUTOR(11 `bottom_ui_strip`)=readout vs PUPIL(11 `progress_counter`)=counter

## 5. Round-1 open_questions not touched by probes

### TUTOR
- _all covered (or no open questions)_

### PUPIL
- _all covered (or no open questions)_

## 6. Harness reliability notes

- ELEMENT_MOVED tracks a 'signature colour' (colour most over-represented in the element's pre-bbox vs the rest of the grid), then returns the bbox of the nearest connected component of that colour after the instructions run.
- If the element's distinctive colour also appears elsewhere, the nearest-component heuristic may still produce a misleading bbox. Treat a post_bbox whose area is >>10x the element's pre-bbox as a tracking failure, not a real movement.
- CHANGE_REPORT aggregates the above per-element tracker into a single structured summary: element_motions (dr/dc vs pre_bbox), disappearances, appearances (novel-colour components), counter_changes (for counter/readout elements: fill count before/after), unexplained_regions (clustered residual diff cells with before/after patches), and a full_frame_fallback when diff > 30% of frame.

## 7. Revision notes (what each model said changed)

### TUTOR
- **element 2**: Retracted 'player_cursor' (element 2) as agent. The white cross (color 0/1 pattern at rows 30-32, cols 20-22) did NOT move in any probe. Per prior knowledge, it is a 'change indicator' marking where the next state transition will occur. — _P1, P2, P3, P4, P5 all show element 2 (player_cursor) with dr=0, dc=0, moved=false across all actions._
- **element 1**: Retracted 'agent_icon_top' (element 1) as agent. It never moved in any probe. — _P1-P5 all show element 1 with dr=0, dc=0, moved=false._
- **element 6**: Identified the ACTUAL moving element: the colored block (originally called 'blue_orange_block', element 6) moves in response to actions. ACTION1 moved it dr=-5 (upward), ACTION3 moved it dc=-5 (leftward). This is the agent/cursor. — _P1: element 6 moved dr=-5, dc=0. P5: element 6 moved dr=0, dc=-5. This is the only element that consistently moves in response to directional actions._
- **initial_strategy**: Revised action mapping: ACTION1=move up (dr=-5), ACTION3=move left (dc=-5). ACTION2 produced no movement of element 6 (only 2 diff cells, likely the counter bar). ACTION4 unknown. — _P1 shows element 6 moving dr=-5 with ACTION1. P5 shows element 6 moving dc=-5 with ACTION3. P2 shows element 6 not moving with ACTION2._
- **element 11**: Revised 'bottom_ui_strip' (element 11) function: it is a COUNTER/progress bar that decrements by 1 per action taken. Each action reduces fill from 42 toward 0 (or toward a target). Per prior knowledge this is the progress bar. — _P1: fill 42->41. P2: fill 42->41. P4 (5x ACTION1): fill 42->37 (5 decrements). P5: fill 42->41. Each action costs 1 unit._
- **element 1, element 12**: Revised 'agent_icon_top' (element 1) and 'inner_room_top' (element 12) to 'reference_glyph' function. Per prior knowledge, small bordered boxes with internal glyphs are reference patterns the player must recreate. — _Prior knowledge item 5: bordered boxes with internal glyphs are reference glyphs. Element 1/12 never moved and has a distinctive internal pattern (color 9 cells)._
- **element 7**: Revised 'left_panel_icon' (element 7) to 'reference_glyph' function - it is the second reference glyph in the matched pair. — _Prior knowledge item 5: reference glyphs appear in matched pairs. Element 7 is a small bordered box with internal L-shaped glyph, never moved._
- **element 6**: Revised element 6 (blue_orange_block) function from 'collectible' to 'agent'. It is the movable cursor/agent that the player controls. — _It is the only element that moves in response to ACTION1 and ACTION3, moving by exactly 5 cells per action in the respective direction._
- **element 4, element 13**: Revised 'main_floor_area' (element 4) - the tracker was confused. The element 4 tracker was tracking the same color-3 region as element 6's movement destination. The actual main floor is the large color-3 region. The agent (element 6) moves within it. — _Elements 4, 6, 13 all showed identical motion in P1 (dr=-5) and P5 (dc=-5), suggesting the tracker was finding the same connected component. The agent is element 6 (the 12-cell colored block with color 9 and 12)._
- **element 2**: Revised 'white cross' (element 2, player_cursor) function to 'target' or 'switch'. Per prior knowledge it is a change indicator marking where state transitions happen. — _Prior knowledge item 1: white cross is a change indicator, not the agent. It never moved in any probe._
- **element 8**: Revised 'progress_bar_yellow' (element 8) - the actual yellow bar is in rows 62-63, cols 13-54 (the 11 element). Element 8 bbox was wrong. The counter decrements with each action. — _Counter changes show fill going from 42 to 37 over 5 actions. The yellow bar (color 11) in the bottom strip is the progress counter._
- **elements 3, 5, 13, 14**: Dropped elements 3, 5, 13, 14 as separate elements - they were tracking artifacts or sub-regions of the main floor. The main floor (color-3 region) is one large navigable area. — _Tracker confusion: elements 3, 4, 13 all showed identical motion in probes, indicating they were tracking the same or overlapping color-3 regions. Element 14 showed nonsensical motion (dr=33, dc=-20) indicating tracking failure._

### PUPIL
- **1, 12**: Redefined element 1 (agent_icon_top) and 12 (inner_room_top) as static reference/portal elements rather than agent/portal. — _P1 and P2 showed element 1 did not move when ACTION1 or ACTION2 were called, and it is located in a distinct bordered room._
- **2**: Redefined element 2 (player_cursor) as a static UI element or indicator. — _P1 and P2 showed element 2 did not move._
- **4, 6**: Corrected element 4 (main_floor_area) and 6 (blue_orange_block) to be part of a moving 'canvas' or 'viewport'. — _P1 showed that ACTION1 moves the main floor area and the block upwards (dr=-5), suggesting the actions manipulate the viewport or the grid itself._
- **11**: Identified element 11 (bottom_ui_strip) as the primary progress counter. — _P1 and P2 showed that ACTION1 and ACTION2 decrement the 'fill' of element 11, and P4 showed repeated ACTION1 decrements it further._
- **14**: Re-evaluated element 14 (left_sub_room) as a moving component of the grid/viewport. — _P1 showed element 14 moves significantly (dr=33, dc=-20) when ACTION1 is called, indicating it is part of the shifting game field._
