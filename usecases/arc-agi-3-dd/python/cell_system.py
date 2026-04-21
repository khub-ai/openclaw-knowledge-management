"""Cell-coordinate system for strict-mode play.

Rationale
---------
The agent moves in fixed pixel increments (the magnitudes of the entries
in action_effects).  Positions differing by a non-multiple of that
magnitude are unreachable.  But the game's sprites render at arbitrary
pixel positions -- a rotation trigger sprite might be at pixel row 31
even though the agent only visits rows that are multiples of 5 from
spawn.

Reasoning in raw pixels forces TUTOR to constantly distinguish
"distinct reachable positions" (every 5 pixels) from "the sub-pixel
offset at which a sprite was rendered."  This is a source of bugs:
TUTOR tries to MOVE_TO the sprite's reported centroid, the BFS snaps
to the nearest reachable pixel, TUTOR interprets `reached=False` as
failure, never realises the agent's sprite OVERLAPPED the target.

This module defines a CELL coordinate system where:
  - cell_size = min |action_effect| magnitude across all learned actions
    (it's the stride of the agent).
  - origin  = agent's centroid at level start (so spawn is always cell (0,0))
  - cell (cr, cc) represents a square region of `cell_size x cell_size`
    pixels, centered at pixel (origin_r + cr*cell_size, origin_c + cc*cell_size)

Any pixel (pr, pc) belongs to cell (cr, cc) iff
  cr = round((pr - origin_r) / cell_size)
  cc = round((pc - origin_c) / cell_size)

Under this system:
  - The agent always sits exactly at a cell center (its centroid is on
    the integer-cell grid by construction).
  - A sprite rendered at pixel (31, 21) with spawn-origin (45, 36) and
    cell_size=5 maps to cell (-3, -3), which is ALSO where the agent
    ends up if it issues ACTION1+ACTION1+ACTION1+ACTION3+ACTION3+ACTION3
    from spawn.  Agent-sprite overlap ambiguity disappears.
  - BFS plans in cell coordinates; each action moves by a unit vector
    in cell space; walls are (cell_r, cell_c, action) triples.

Game-agnostic: cell_size is not hardcoded.  It's derived from the
action effects learned in the bootstrap phase.  For a game where moves
are 3 pixels apart instead of 5, cell_size becomes 3 and the whole
system adapts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CellSystem:
    """A one-level coordinate system derived from the agent's stride."""

    cell_size: int
    origin_r:  int
    origin_c:  int

    # ---------- pixel <-> cell conversions ----------

    def pix_to_cell(self, pr: int, pc: int) -> tuple[int, int]:
        """Map a pixel position to the cell that contains it (nearest center)."""
        cr = round((pr - self.origin_r) / self.cell_size)
        cc = round((pc - self.origin_c) / self.cell_size)
        return (int(cr), int(cc))

    def cell_to_pix(self, cr: int, cc: int) -> tuple[int, int]:
        """Map a cell to its center pixel."""
        pr = self.origin_r + cr * self.cell_size
        pc = self.origin_c + cc * self.cell_size
        return (int(pr), int(pc))

    def cell_bbox_pix(self, cr: int, cc: int) -> list[int]:
        """Return pixel bbox [r_min, c_min, r_max, c_max] for a cell (inclusive)."""
        cr_r, cr_c = self.cell_to_pix(cr, cc)
        half = self.cell_size // 2
        # For odd cell sizes, bbox is symmetric around center.
        # For even cell sizes, top/left get one fewer pixel than bottom/right.
        return [cr_r - half, cr_c - half,
                cr_r + (self.cell_size - 1 - half),
                cr_c + (self.cell_size - 1 - half)]

    # ---------- bulk transforms ----------

    def component_to_cell(self, comp: dict) -> tuple[int, int]:
        """Given a component dict from pixel_elements, return its cell address.

        Uses the centroid for the primary mapping.  A sprite whose bbox
        spans multiple cells will be reported as belonging to the cell
        containing its centroid -- if you need multi-cell awareness,
        call `component_cells_covered` instead.
        """
        return self.pix_to_cell(comp["centroid"][0], comp["centroid"][1])

    def component_cells_covered(self, comp: dict) -> set[tuple[int, int]]:
        """Return the set of cells any pixel of this component falls into.

        For a sprite straddling a cell boundary this can be more than one
        cell.  Used when checking whether the agent's position overlaps
        a sprite's occupancy.
        """
        r_min, c_min, r_max, c_max = comp["bbox"]
        out: set[tuple[int, int]] = set()
        # Sample a sparse grid; the cell mapping is piecewise-constant,
        # so corners + center are sufficient to find all containing cells.
        for r in (r_min, (r_min + r_max) // 2, r_max):
            for c in (c_min, (c_min + c_max) // 2, c_max):
                out.add(self.pix_to_cell(r, c))
        return out

    def action_to_cell_delta(self, dr_pix: int, dc_pix: int) -> tuple[int, int]:
        """Convert a pixel-space action effect to a cell-space delta."""
        # dr_pix and dc_pix are multiples of cell_size by construction.
        return (dr_pix // self.cell_size if dr_pix else 0,
                dc_pix // self.cell_size if dc_pix else 0)


# ---------------------------------------------------------------------------
# Inference -- derive a CellSystem from bootstrap observations
# ---------------------------------------------------------------------------

def infer_cell_system(
    action_effects:  dict[str, tuple[int, int]],
    agent_centroid:  tuple[int, int],
) -> CellSystem:
    """Infer a CellSystem from learned action effects + current agent position.

    cell_size is the min nonzero |dr| or |dc| across all actions -- the
    fundamental step unit.  For ls20-style games all four actions have
    magnitude 5, so cell_size = 5.  For games with heterogeneous move
    magnitudes, the GCD would be more correct; the caller can override.

    Origin is set to the agent_centroid so that spawn is exactly cell (0,0).
    """
    magnitudes: list[int] = []
    for dr, dc in action_effects.values():
        if dr != 0:
            magnitudes.append(abs(dr))
        if dc != 0:
            magnitudes.append(abs(dc))
    if not magnitudes:
        raise ValueError("Cannot infer cell_size: no nonzero action effects")
    # GCD in case actions differ; for uniform-magnitude games this == min.
    import math
    cell_size = magnitudes[0]
    for m in magnitudes[1:]:
        cell_size = math.gcd(cell_size, m)
    return CellSystem(
        cell_size = int(cell_size),
        origin_r  = int(agent_centroid[0]),
        origin_c  = int(agent_centroid[1]),
    )


# ---------------------------------------------------------------------------
# Convenience: project a list of components into cell space for prompting
# ---------------------------------------------------------------------------

def components_in_cells(
    comps:  Iterable[dict],
    cs:     CellSystem,
) -> list[dict]:
    """Return a new list of component dicts with cell coordinates added.

    Each output component includes all the original pixel-level fields,
    plus:
      cell        : (cr, cc) of the centroid
      cells_covered : list of all cells any part of this component falls in
    Useful for showing TUTOR cell-native coordinates while preserving
    pixel-level geometry for debugging.
    """
    out: list[dict] = []
    for c in comps:
        cell = cs.component_to_cell(c)
        covered = sorted(cs.component_cells_covered(c))
        new = dict(c)
        new["cell"]          = list(cell)
        new["cells_covered"] = [list(x) for x in covered]
        out.append(new)
    return out
