"""Harness-side BFS navigator for ARC-AGI-3 play loop.

Given a set of known action effects (action_name -> (dr, dc)) and a
start position, plans a path to a target position using BFS over the
reachable discrete grid.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

GRID_H = 64
GRID_W = 64

# The navigator intentionally DOES NOT hardcode which palette values are
# walls.  Under the prime directive, which palette is impassable must be
# discovered at runtime (by attempting moves and observing failures), not
# read from source.  Callers of build_passable_grid must supply
# wall_palettes explicitly.  The legacy default of {4} (derived by
# Claude Code from reading ls20.py) is preserved as a _LEGACY_WALL_PALETTES
# constant strictly for legacy-mode benchmarking; it MUST NOT be used as
# a default in strict code paths.
_LEGACY_WALL_PALETTES: frozenset[int] = frozenset({4})   # legacy only


def _passable(nr: int, nc: int, passable_grid) -> bool:
    """Return True if cell (nr,nc) is navigable according to passable_grid."""
    if passable_grid is None:
        return True
    return bool(passable_grid[nr, nc])


def build_passable_grid(
    frame_grid,
    wall_palettes: frozenset[int] | None = None,
) -> np.ndarray:
    """Return a 64x64 bool array: True = agent can enter this cell.

    wall_palettes must be supplied by the caller and reflects palettes
    EMPIRICALLY discovered to be impassable (by trying to move onto them
    and observing that the agent did not move).  No default is provided
    because hardcoding which palette is a wall would constitute source-
    derived injection (the prime directive forbids this).

    For legacy-mode runs, callers may explicitly pass
    navigator._LEGACY_WALL_PALETTES.  New/strict code paths must accumulate
    wall_palettes from runtime observation.
    """
    if wall_palettes is None:
        # No palettes known yet -- every cell is tentatively passable.
        # The agent will discover wall palettes by attempting moves and
        # observing blockage.
        arr = np.asarray(frame_grid, dtype=np.int32)
        return np.ones(arr.shape[:2], dtype=bool)
    arr = np.asarray(frame_grid, dtype=np.int32)
    passable = np.ones(arr.shape[:2], dtype=bool)
    for p in wall_palettes:
        passable[arr == p] = False
    return passable


def bfs_navigate(
    start:          tuple[int, int],
    target:         tuple[int, int],
    action_effects: dict[str, tuple[int, int]],
    max_steps:      int = 60,
    walls:          set | None = None,
    passable_grid           = None,
) -> Optional[list[str]]:
    """Return the shortest action sequence to move from `start` to `target`.

    action_effects maps action name -> (dr, dc).  Only actions with non-zero
    displacement are used for navigation; zero-displacement actions are ignored.

    Returns a list of action names (possibly empty if already at target), or
    None if the target is unreachable within max_steps.
    """
    if start == target:
        return []

    move_actions = {a: (dr, dc) for a, (dr, dc) in action_effects.items()
                    if dr != 0 or dc != 0}
    if not move_actions:
        return None

    sr, sc = start
    tr, tc = target

    # BFS: state = (row, col), track best path
    queue: deque[tuple[tuple[int, int], list[str]]] = deque([((sr, sc), [])])
    visited: dict[tuple[int, int], int] = {(sr, sc): 0}

    while queue:
        (r, c), path = queue.popleft()
        if len(path) >= max_steps:
            continue
        for action, (dr, dc) in move_actions.items():
            nr = max(0, min(GRID_H - 1, r + dr))
            nc = max(0, min(GRID_W - 1, c + dc))
            if walls and (r, c, action) in walls:
                # Trust runtime-discovered walls unconditionally.
                # These come from game-authoritative cursor positions and
                # reflect walls the palette-4 analysis may have missed
                # (e.g. L1 walls that use a different palette colour).
                continue
            if not _passable(nr, nc, passable_grid):
                continue  # palette-level wall — skip
            steps = len(path) + 1
            if (nr, nc) in visited and visited[(nr, nc)] <= steps:
                continue
            new_path = path + [action]
            if (nr, nc) == (tr, tc):
                return new_path
            visited[(nr, nc)] = steps
            queue.append(((nr, nc), new_path))

    return None  # unreachable within max_steps


def nearest_reachable(
    start:          tuple[int, int],
    target:         tuple[int, int],
    action_effects: dict[str, tuple[int, int]],
    max_steps:      int = 60,
    walls:          set | None = None,
    passable_grid           = None,
) -> Optional[tuple[tuple[int, int], list[str]]]:
    """Return (closest_reachable_pos, path) to get as close as possible to target.

    Used when the exact target cell is not on the reachable grid.
    """
    move_actions = {a: (dr, dc) for a, (dr, dc) in action_effects.items()
                    if dr != 0 or dc != 0}
    if not move_actions:
        return None

    sr, sc = start
    tr, tc = target
    best_pos = (sr, sc)
    best_path: list[str] = []
    best_dist = (sr - tr) ** 2 + (sc - tc) ** 2

    queue: deque[tuple[tuple[int, int], list[str]]] = deque([((sr, sc), [])])
    visited: dict[tuple[int, int], int] = {(sr, sc): 0}

    while queue:
        (r, c), path = queue.popleft()
        dist = (r - tr) ** 2 + (c - tc) ** 2
        if dist < best_dist:
            best_dist = dist
            best_pos = (r, c)
            best_path = path
        if len(path) >= max_steps:
            continue
        for action, (dr, dc) in move_actions.items():
            nr = max(0, min(GRID_H - 1, r + dr))
            nc = max(0, min(GRID_W - 1, c + dc))
            if walls and (r, c, action) in walls:
                continue  # trust runtime-discovered walls unconditionally
            if not _passable(nr, nc, passable_grid):
                continue
            steps = len(path) + 1
            if (nr, nc) in visited and visited[(nr, nc)] <= steps:
                continue
            visited[(nr, nc)] = steps
            queue.append(((nr, nc), path + [action]))

    if best_pos == (sr, sc):
        return None
    return best_pos, best_path
