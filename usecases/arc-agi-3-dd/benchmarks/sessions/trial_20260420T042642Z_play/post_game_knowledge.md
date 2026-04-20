ls20-9607627b knowledge note for future play:

ACTIONS: Prior knowledge says ACTION1-4 are state-mutation actions, NOT movement. They modify the cell under the white cross (change indicator). Only ACTION1 was tested this run (1 action taken, session ended). ACTION2 appeared to produce no visible change in earlier probing — possibly blocked, wrong cell state, or requires precondition. ACTION3 and ACTION4 directions/effects unknown. Do not assume any action moves an agent cursor.

ELEMENTS:
- White cross (color 0/1, ~rows 30-32, cols 20-22): Change indicator. Marks the cell that will be mutated when an action fires. It may itself move after each action to indicate the next target cell — watch for this.
- Small bordered boxes (top-center ~rows 8-15, cols 32-40; bottom-left ~rows 53-62, cols 1-10): Reference glyphs encoding the target pattern. One is likely the reference, one the write-target on the playing field. The internal color-9 arrangement is what must be recreated.
- Yellow bar (bottom strip, row 61-62): Progress counter. Decrements with each action (42→41 per action observed). Win = fill it completely, meaning you need to INCREASE it, not deplete it. Decrementing suggests wrong actions or wrong sequence — this is a trap. Possibly the bar fills when correct matches are made and depletes on wrong actions.
- Red dots (bottom-right): Remaining lives/attempts. Preserve them.
- Color-12/9 block (~rows 44-46, cols 34-38): May be agent cursor OR a colored tile that is part of the pattern to match. Unclear if player-controlled.
- Gap/pit (~rows 29-37, cols 29-33): Color-4 region within floor — possible hazard or passage.

WIN CONDITION: Recreate the reference glyph pattern on the playing field using state-mutation actions under the change indicator, tracked by the progress bar filling upward.

HYPOTHESES TO TEST: (1) Does the change indicator move after each action, cycling through target cells? (2) Does ACTION2 toggle a different property than ACTION1 (e.g., color vs. presence)?