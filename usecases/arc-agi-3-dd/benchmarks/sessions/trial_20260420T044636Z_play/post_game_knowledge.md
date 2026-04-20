Level ls20-9607627b (and likely ls-family siblings): 45 turns spent, 0/7 levels completed. Key findings and corrections to prior knowledge:

The white cross (change_indicator, ~rows 30-32, cols 20-22) IS movable. ACTION1 moves it upward by ~5 rows. ACTION3 moves it left by ~5 cols. ACTION2 and ACTION4 likely move it down and right respectively (not confirmed but strongly suspected by symmetry). The cross is NOT a fixed marker — it is a cursor that the player repositions over target cells.

The four actions are directional movement keys for the change_indicator cursor, not state-mutation actions per the prior knowledge. The cursor must be navigated to specific cells, then presumably an action fires a mutation at that location. However, it remains unclear which action (if any) actually triggers the mutation vs. just moves the cursor — this was never isolated.

The yellow progress bar DECREMENTS with each action taken (starts ~42, loses 1-2 per action). It does NOT fill up toward a win — it is a move budget/timer. Running it to zero causes level reset and costs one red attempt. Do NOT spend turns on blind probing; the budget is tight (~20-25 effective moves).

The reference glyphs (top-center bordered box, bottom-left bordered box) encode the target pattern. The internal color-9 arrangement in the top glyph is what must be recreated on the playing field.

Critical trap this run: Spent all 45 turns probing action effects instead of acting. The move budget is ~42 actions total — probing more than 5-6 actions is fatal.

Recommended strategy: Use ACTION1/ACTION3 to move the change_indicator cursor to cells matching the reference glyph pattern, then use ACTION2 or ACTION4 to stamp/imprint the target color. Test ACTION2 and ACTION4 immediately (1-2 presses each) to confirm they move the cursor down/right, then navigate directly to target cells.