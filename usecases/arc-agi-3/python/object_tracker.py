"""
object_tracker.py — Object-level frame analysis for ARC-AGI-3.

Provides zero-LLM-cost object detection and tracking across frames so that
the ensemble can reason about "the azure rectangle moved right 1 step" rather
than "52 cells changed color."

Public API
----------
detect_objects(frame)
    -> list[ObjectRecord]
    Find all connected components (4-connectivity) in a 2-D color grid.

diff_objects(frame_before, frame_after)
    -> ObjectDiff
    Match objects across two frames and classify each change as moved,
    appeared, or disappeared.

format_object_diff(diff, max_entries=8)
    -> str
    Produce a compact human-readable summary for prompt injection.

summarize_action_effects(action_effects)
    -> str
    Render the full accumulated action-effect table (object-level) for
    injection into OBSERVER / MEDIATOR prompts.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Color names (same as ARC-AGI palette)
# ---------------------------------------------------------------------------

_COLOR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "azure",
    9: "white",
    # Colors beyond 9 are game-specific extensions — named generically so the
    # system discovers their roles through exploration rather than hardcoding.
}


def color_name(c: int) -> str:
    return _COLOR_NAMES.get(int(c), f"color{c}")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ObjectRecord:
    """One connected component of the same color."""
    color: int
    size: int                        # number of cells
    centroid: tuple[float, float]    # (row, col)
    bbox: dict                       # r_min/r_max/c_min/c_max
    is_background: bool = False      # True if very large relative to frame

    # Derived geometry (populated by detect_objects)
    width: int = 0                   # cols spanned (c_max - c_min + 1)
    height: int = 0                  # rows spanned (r_max - r_min + 1)

    @property
    def aspect_ratio(self) -> float:
        """width / height; >1 = horizontal, <1 = vertical, 1 = square."""
        return self.width / self.height if self.height else 1.0

    @property
    def orientation(self) -> str:
        """'horizontal', 'vertical', or 'square' based on bounding-box shape."""
        r = self.aspect_ratio
        if r > 1.3:
            return "horizontal"
        if r < 0.77:
            return "vertical"
        return "square"

    @property
    def label(self) -> str:
        return (
            f"{color_name(self.color)}"
            f"(size={self.size}, {self.width}w x {self.height}h, {self.orientation})"
        )

    def distance_to(self, other: "ObjectRecord") -> float:
        return math.hypot(
            self.centroid[0] - other.centroid[0],
            self.centroid[1] - other.centroid[1],
        )


@dataclass
class MovedObject:
    obj: ObjectRecord
    delta_r: float
    delta_c: float
    from_centroid: tuple[float, float]
    to_centroid: tuple[float, float]

    @property
    def direction(self) -> str:
        dr, dc = self.delta_r, self.delta_c
        parts = []
        if abs(dr) > 0.5:
            parts.append("down" if dr > 0 else "up")
        if abs(dc) > 0.5:
            parts.append("right" if dc > 0 else "left")
        return "+".join(parts) if parts else "stationary"

    @property
    def magnitude(self) -> float:
        return math.hypot(self.delta_r, self.delta_c)


@dataclass
class AttributeChange:
    """Records which attributes changed between the before and after versions of a matched object."""
    color: int                              # color of the object (key for lookup)
    before: ObjectRecord
    after: ObjectRecord
    changed: list[str] = field(default_factory=list)  # e.g. ["size", "orientation", "width"]

    @property
    def summary(self) -> str:
        parts = []
        for attr in self.changed:
            bv = getattr(self.before, attr, None)
            av = getattr(self.after,  attr, None)
            if attr == "orientation":
                parts.append(f"orientation {bv}->{av}")
            elif attr == "size":
                parts.append(f"size {bv}->{av}")
            elif attr == "width":
                parts.append(f"width {bv}->{av}")
            elif attr == "height":
                parts.append(f"height {bv}->{av}")
            else:
                parts.append(f"{attr} {bv}->{av}")
        label = color_name(self.color)
        return f"{label}: " + ", ".join(parts) if parts else f"{label}: no attribute change"


@dataclass
class ObjectDiff:
    """Result of comparing two frames at the object level."""
    moved: list[MovedObject] = field(default_factory=list)
    appeared: list[ObjectRecord] = field(default_factory=list)
    disappeared: list[ObjectRecord] = field(default_factory=list)
    stationary: list[ObjectRecord] = field(default_factory=list)
    attribute_changes: list[AttributeChange] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.moved or self.appeared or self.disappeared or self.attribute_changes)

    @property
    def total_moved_cells(self) -> int:
        return sum(m.obj.size for m in self.moved)


# ---------------------------------------------------------------------------
# Connected-component detection
# ---------------------------------------------------------------------------

def detect_objects(
    frame: list,
    background_threshold: float = 0.40,
) -> list[ObjectRecord]:
    """
    Find all connected components (4-connectivity) in a 2-D color grid.

    Parameters
    ----------
    frame : list[list[int]]
        2-D grid of integer color values.
    background_threshold : float
        Objects whose size exceeds this fraction of total cells are marked
        as background (large uniform regions like the game floor).
        Default 0.40 = 40%.  Raised from 0.25 so that mid-size structural
        regions (play arenas, platforms) are not silently excluded.

    Returns
    -------
    List of ObjectRecord, sorted by size descending (largest first).
    """
    if not frame or not frame[0]:
        return []

    rows = len(frame)
    cols = len(frame[0])
    total_cells = rows * cols
    visited = [[False] * cols for _ in range(rows)]
    objects: list[ObjectRecord] = []

    for r0 in range(rows):
        for c0 in range(cols):
            if visited[r0][c0]:
                continue
            color = int(frame[r0][c0])
            # BFS
            cells: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque()
            q.append((r0, c0))
            visited[r0][c0] = True
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and not visited[nr][nc]
                            and int(frame[nr][nc]) == color):
                        visited[nr][nc] = True
                        q.append((nr, nc))

            size = len(cells)
            if size == 0:
                continue

            rs = [r for r, c in cells]
            cs = [c for r, c in cells]
            centroid = (sum(rs) / size, sum(cs) / size)
            r_min, r_max = min(rs), max(rs)
            c_min, c_max = min(cs), max(cs)
            bbox = {"r_min": r_min, "r_max": r_max, "c_min": c_min, "c_max": c_max}
            is_bg = (size / total_cells) >= background_threshold

            objects.append(ObjectRecord(
                color=color,
                size=size,
                centroid=centroid,
                bbox=bbox,
                is_background=is_bg,
                width=c_max - c_min + 1,
                height=r_max - r_min + 1,
            ))

    objects.sort(key=lambda o: o.size, reverse=True)
    return objects


# ---------------------------------------------------------------------------
# Object tracking (frame diff)
# ---------------------------------------------------------------------------

def diff_objects(
    frame_before: list,
    frame_after: list,
    match_radius: float = 10.0,
) -> ObjectDiff:
    """
    Compare two frames at the object level.

    For each color present in either frame, this function matches objects
    by minimum centroid distance (greedy nearest-neighbour, same color only).

    Parameters
    ----------
    match_radius : float
        Maximum centroid distance to consider two objects the same entity.
        Objects further apart are treated as disappeared + appeared.
    """
    if not frame_before or not frame_after:
        return ObjectDiff()

    before_objs = detect_objects(frame_before)
    after_objs  = detect_objects(frame_after)

    # Group by color
    def by_color(objs: list[ObjectRecord]) -> dict[int, list[ObjectRecord]]:
        d: dict[int, list[ObjectRecord]] = {}
        for o in objs:
            d.setdefault(o.color, []).append(o)
        return d

    before_by_color = by_color(before_objs)
    after_by_color  = by_color(after_objs)

    result = ObjectDiff()
    all_colors = set(before_by_color) | set(after_by_color)

    for color in all_colors:
        b_list = before_by_color.get(color, [])
        a_list = list(after_by_color.get(color, []))

        matched_a: set[int] = set()

        for b_obj in b_list:
            # Find closest unmatched after-object of same color
            best_idx: Optional[int] = None
            best_dist = float("inf")
            for i, a_obj in enumerate(a_list):
                if i in matched_a:
                    continue
                d = b_obj.distance_to(a_obj)
                if d < best_dist:
                    best_dist = d
                    best_idx = i

            if best_idx is not None and best_dist <= match_radius:
                matched_a.add(best_idx)
                a_obj = a_list[best_idx]
                delta_r = a_obj.centroid[0] - b_obj.centroid[0]
                delta_c = a_obj.centroid[1] - b_obj.centroid[1]

                if abs(delta_r) < 0.5 and abs(delta_c) < 0.5:
                    result.stationary.append(b_obj)
                else:
                    result.moved.append(MovedObject(
                        obj=a_obj,
                        delta_r=round(delta_r, 1),
                        delta_c=round(delta_c, 1),
                        from_centroid=b_obj.centroid,
                        to_centroid=a_obj.centroid,
                    ))

                # Check for attribute changes between matched before/after objects
                changed_attrs: list[str] = []
                if b_obj.size != a_obj.size:
                    changed_attrs.append("size")
                if b_obj.width != a_obj.width:
                    changed_attrs.append("width")
                if b_obj.height != a_obj.height:
                    changed_attrs.append("height")
                if b_obj.orientation != a_obj.orientation:
                    changed_attrs.append("orientation")
                if changed_attrs:
                    result.attribute_changes.append(AttributeChange(
                        color=color,
                        before=b_obj,
                        after=a_obj,
                        changed=changed_attrs,
                    ))
            else:
                # No match found — object disappeared
                result.disappeared.append(b_obj)

        # Unmatched after-objects appeared
        for i, a_obj in enumerate(a_list):
            if i not in matched_a:
                result.appeared.append(a_obj)

    return result


# ---------------------------------------------------------------------------
# Formatting for prompts
# ---------------------------------------------------------------------------

def format_object_diff(diff: ObjectDiff, max_entries: int = 8) -> str:
    """Compact human-readable summary of an ObjectDiff for prompt injection."""
    if not diff.has_changes:
        return "no object-level changes detected"

    lines: list[str] = []
    count = 0

    for m in diff.moved:
        if count >= max_entries:
            break
        if m.obj.is_background:
            continue
        dr = f"{m.delta_r:+.0f}r" if abs(m.delta_r) >= 0.5 else ""
        dc = f"{m.delta_c:+.0f}c" if abs(m.delta_c) >= 0.5 else ""
        delta_str = " ".join(filter(None, [dr, dc])) or "~0"
        lines.append(
            f"  MOVED   {m.obj.label} "
            f"{m.direction} by ({delta_str})  "
            f"centroid {_fmt_pt(m.from_centroid)} -> {_fmt_pt(m.to_centroid)}"
        )
        count += 1

    for o in diff.appeared:
        if count >= max_entries:
            break
        if o.is_background:
            continue
        lines.append(f"  APPEARED  {o.label} at centroid {_fmt_pt(o.centroid)}")
        count += 1

    for o in diff.disappeared:
        if count >= max_entries:
            break
        if o.is_background:
            continue
        lines.append(f"  GONE      {o.label} from centroid {_fmt_pt(o.centroid)}")
        count += 1

    for ac in diff.attribute_changes:
        if count >= max_entries:
            break
        if ac.before.is_background:
            continue
        lines.append(f"  CHANGED   {ac.summary}")
        count += 1

    if not lines:
        # Only background changes — report briefly
        bg_moved = [m for m in diff.moved if m.obj.is_background]
        if bg_moved:
            lines.append(
                f"  only background region(s) shifted "
                f"(e.g. {bg_moved[0].obj.label} {bg_moved[0].direction})"
            )
        else:
            lines.append("  only background-level changes")

    return "\n".join(lines)


def summarize_action_effects(action_effects: dict) -> str:
    """
    Render the full accumulated action-effect table for prompt injection.

    Uses the `object_observations` field if present (richer), otherwise
    falls back to pixel-level diff stats.
    """
    if not action_effects:
        return "  (no action effects recorded yet)"

    lines: list[str] = []
    for action, e in sorted(action_effects.items()):
        total    = e.get("total_calls", 0)
        nonzero  = e.get("nonzero_calls", 0)
        advances = e.get("level_advances", 0)

        obj_obs: list[dict] = e.get("object_observations", [])
        if obj_obs:
            # Summarize the most consistent object-level effect
            summary = _consensus_object_summary(obj_obs)
        else:
            # Fallback: pixel-level
            obs_list = e.get("observations", [{}])
            last = obs_list[-1] if obs_list else {}
            summary = f"pixel diff={last.get('diff', 0)}"

        lines.append(
            f"  {action}: called={total} changed={nonzero} "
            f"level_advances={advances}\n"
            f"    {summary}"
        )

    return "\n".join(lines)


def _consensus_object_summary(obj_obs: list[dict]) -> str:
    """
    Given a list of per-step object observations, produce a consensus summary.

    Looks for the most frequently-moved non-background object and describes
    the typical direction and magnitude.
    """
    if not obj_obs:
        return "(no object observations)"

    # Count moves per (color, direction)
    move_counts: dict[tuple[int, str], list[float]] = {}
    no_change_count = 0

    for obs in obj_obs:
        moved = obs.get("moved", [])
        if not moved and not obs.get("appeared") and not obs.get("disappeared"):
            no_change_count += 1
            continue
        for m in moved:
            if m.get("is_background"):
                continue
            key = (m["color"], m["direction"])
            move_counts.setdefault(key, []).append(m.get("magnitude", 0))

    if not move_counts:
        if no_change_count == len(obj_obs):
            return "no object movement observed (all calls had no effect)"
        return "changes detected but no clear object movement pattern"

    # Find the most frequent move pattern
    best_key = max(move_counts, key=lambda k: len(move_counts[k]))
    color, direction = best_key
    magnitudes = move_counts[best_key]
    avg_mag = sum(magnitudes) / len(magnitudes)
    freq = len(magnitudes)
    total = len(obj_obs)

    return (
        f"{color_name(color)} object moves {direction} "
        f"~{avg_mag:.1f} cells/call  "
        f"(observed {freq}/{total} calls)"
    )


def summarize_current_objects(frame: list, concept_bindings: dict | None = None) -> str:
    """
    One-line summary of all non-background objects in the current frame.

    concept_bindings supports two schemas:
      {color_int: "role_name"}    -- color-keyed roles (player_piece, step_counter)
      {"wall_colors": [c, ...]}   -- concept-keyed list (wall is game-local)
    Colors in wall_colors are labeled "wall?" to signal game-local uncertainty.
    """
    objs = detect_objects(frame)
    bindings = concept_bindings or {}
    wall_colors: set = set(bindings.get("wall_colors", []))
    lines = []
    for o in objs:
        if o.is_background:
            continue
        if o.color in wall_colors:
            concept = "wall?"
        else:
            raw = bindings.get(o.color)
            if isinstance(raw, dict):
                role = raw.get("role", "")
                conf = raw.get("confidence", 0.0)
                concept = f"{role}({conf:.0%})" if role else None
            else:
                concept = raw  # plain string or None
        name = f"{concept}:{color_name(o.color)}" if concept else color_name(o.color)
        lines.append(
            f"  {name}: size={o.size} centroid={_fmt_pt(o.centroid)} "
            f"{o.width}w x {o.height}h {o.orientation}"
        )
    return "\n".join(lines) if lines else "  (no foreground objects detected)"


def compute_trend_predictions(action_effects: dict, steps_remaining: int) -> list[str]:
    """
    Scan accumulated object_observations for monotonic trends across ANY
    tracked attribute and project future states.

    Works entirely from raw numeric before/after values stored in
    attribute_changes records — no string parsing, no hardcoded attribute names.
    Any attribute stored by the object tracker (size, width, height, and any
    future additions) is automatically eligible for trend analysis.

    Detects:
    - Attribute trend: any numeric attribute consistently increasing or
      decreasing each action → predicts depletion (→0) or unbounded growth.
    - Position drift: object consistently moving in one direction across
      all observed moves → predicts it will hit a boundary.

    Requires ≥ 2 observations with consistent sign to fire (avoids noise).
    """
    if not action_effects:
        return []

    predictions: list[str] = []

    # -----------------------------------------------------------------------
    # Collect per-(color, attribute) time series from raw numeric records.
    # We accumulate deltas (after - before) across all stored observations.
    # The last seen "after" value for each attribute gives the current state.
    # -----------------------------------------------------------------------
    attr_deltas:   dict[tuple, list[float]] = {}  # (color, attr) -> [Δ per obs]
    attr_last_val: dict[tuple, float]       = {}  # (color, attr) -> last after value

    move_deltas_r: dict[int, list[float]] = {}  # color -> [Δrow per move obs]
    move_deltas_c: dict[int, list[float]] = {}  # color -> [Δcol per move obs]

    for _action, e in sorted(action_effects.items()):
        for obs in e.get("object_observations", []):
            for ac in obs.get("attribute_changes", []):
                color  = ac.get("color")
                before = ac.get("before", {})
                after  = ac.get("after",  {})
                if color is None or not before or not after:
                    continue
                for attr in ac.get("changed", []):
                    bv = before.get(attr)
                    av = after.get(attr)
                    if bv is None or av is None:
                        continue
                    try:
                        delta = float(av) - float(bv)
                    except (TypeError, ValueError):
                        continue
                    key = (color, attr)
                    attr_deltas.setdefault(key, []).append(delta)
                    attr_last_val[key] = float(av)

            for mv in obs.get("moved", []):
                color = mv.get("color")
                if color is None or mv.get("is_background"):
                    continue
                move_deltas_r.setdefault(color, []).append(float(mv.get("delta_r", 0)))
                move_deltas_c.setdefault(color, []).append(float(mv.get("delta_c", 0)))

    # -----------------------------------------------------------------------
    # Attribute trend analysis (generic — works for size, width, height, etc.)
    # -----------------------------------------------------------------------
    for (color, attr), deltas in attr_deltas.items():
        if len(deltas) < 2:
            continue
        nonzero = [d for d in deltas if abs(d) > 0.1]
        if len(nonzero) < 2:
            continue
        signs = set(1 if d > 0 else -1 for d in nonzero)
        if len(signs) != 1:
            continue  # mixed — no reliable trend
        avg_delta = sum(nonzero) / len(nonzero)
        if abs(avg_delta) < 0.5:
            continue

        cname = color_name(color)
        last_val = attr_last_val.get((color, attr))
        if avg_delta < 0:
            if last_val is not None and last_val > 0:
                steps_to_zero = int(last_val / abs(avg_delta))
                urgency = " [URGENT]" if steps_to_zero <= steps_remaining else ""
                predictions.append(
                    f"PREDICTION: {cname}(color{color}) {attr} shrinking "
                    f"~{abs(avg_delta):.1f}/action, currently ~{last_val:.0f}. "
                    f"Reaches 0 in ~{steps_to_zero} more actions.{urgency}"
                )
        else:
            if last_val is not None:
                predictions.append(
                    f"PREDICTION: {cname}(color{color}) {attr} growing "
                    f"~{avg_delta:.1f}/action, currently ~{last_val:.0f} — "
                    f"unbounded growth."
                )

    # -----------------------------------------------------------------------
    # Position drift trend analysis
    # -----------------------------------------------------------------------
    for color in set(move_deltas_r) | set(move_deltas_c):
        dr_list = move_deltas_r.get(color, [])
        dc_list = move_deltas_c.get(color, [])
        cname   = color_name(color)
        drift_parts: list[str] = []

        for axis, deltas, pos_label, neg_label in (
            ("row", dr_list, "down", "up"),
            ("col", dc_list, "right", "left"),
        ):
            nonzero = [d for d in deltas if abs(d) > 0.1]
            if len(nonzero) < 2:
                continue
            signs = set(1 if d > 0 else -1 for d in nonzero)
            if len(signs) != 1:
                continue  # reverses — bounded oscillation, not a drift
            avg = sum(nonzero) / len(nonzero)
            direction = pos_label if avg > 0 else neg_label
            drift_parts.append(f"{direction} ~{abs(avg):.1f} cells/{axis}/action")

        if drift_parts:
            predictions.append(
                f"PREDICTION: {cname}(color{color}) consistently drifting "
                + ", ".join(drift_parts)
                + " — will reach a boundary if uncorrected."
            )

    return predictions


def detect_wall_contacts(
    frame: list,
    obj: ObjectRecord,
    direction: str,
) -> dict:
    """
    When an object fails to move in a direction, examine what is immediately
    adjacent to it in that direction to identify what stopped it.

    Parameters
    ----------
    frame : current game frame (list of lists)
    obj   : the object that did not move
    direction : "up", "down", "left", "right" (the direction that was blocked)

    Returns
    -------
    dict with:
      "adjacent_colors": Counter of colors found in the cells just beyond the
                         object's bounding box in the blocked direction.
      "sample_cells":    A few (row, col) positions that were inspected.
      "direction":       The blocked direction.
    """
    from collections import Counter

    if not frame:
        return {}

    rows = len(frame)
    cols = len(frame[0]) if rows else 0
    bbox = obj.bbox

    # Determine which edge to scan and in which direction to step
    if direction == "up":
        edge_r = bbox["r_min"] - 1
        scan   = [(edge_r, c) for c in range(bbox["c_min"], bbox["c_max"] + 1)
                  if 0 <= edge_r < rows]
    elif direction == "down":
        edge_r = bbox["r_max"] + 1
        scan   = [(edge_r, c) for c in range(bbox["c_min"], bbox["c_max"] + 1)
                  if 0 <= edge_r < rows]
    elif direction == "left":
        edge_c = bbox["c_min"] - 1
        scan   = [(r, edge_c) for r in range(bbox["r_min"], bbox["r_max"] + 1)
                  if 0 <= edge_c < cols]
    elif direction == "right":
        edge_c = bbox["c_max"] + 1
        scan   = [(r, edge_c) for r in range(bbox["r_min"], bbox["r_max"] + 1)
                  if 0 <= edge_c < cols]
    else:
        return {}

    color_counts: Counter = Counter()
    for r, c in scan:
        color_counts[int(frame[r][c])] += 1

    return {
        "direction":       direction,
        "adjacent_colors": dict(color_counts),
        "sample_cells":    scan[:5],
    }


def infer_typical_direction(action_effects: dict, action_name: str) -> Optional[str]:
    """
    From accumulated move observations for a given action, return the most
    common non-stationary direction the player piece moved, or None if unknown.
    """
    obs_list = action_effects.get(action_name, {}).get("object_observations", [])
    dir_counts: dict[str, int] = {}
    for obs in obs_list:
        for mv in obs.get("moved", []):
            if mv.get("is_background"):
                continue
            d = mv.get("direction", "stationary")
            if d != "stationary":
                dir_counts[d] = dir_counts.get(d, 0) + 1
    return max(dir_counts, key=lambda k: dir_counts[k]) if dir_counts else None


def _fmt_pt(pt: tuple) -> str:
    return f"({pt[0]:.0f},{pt[1]:.0f})"


# ---------------------------------------------------------------------------
# Structural analysis — containment, spatial relations, landmark context
# ---------------------------------------------------------------------------

@dataclass
class ContainmentRelation:
    """One object whose bounding box is fully enclosed within another's."""
    container: ObjectRecord   # outer object (e.g. a bordered box)
    content:   ObjectRecord   # inner object (e.g. a pattern inside the box)


@dataclass
class SpatialRelation:
    """A directional or alignment relationship between two objects."""
    obj_a:    ObjectRecord
    obj_b:    ObjectRecord
    relation: str    # "above", "below", "left", "right"
    col_aligned: bool   # centroids within COL_ALIGN_TOL columns
    row_aligned: bool   # centroids within ROW_ALIGN_TOL rows
    distance: float


_COL_ALIGN_TOL = 4   # columns — centroids this close share a column
_ROW_ALIGN_TOL = 4   # rows


def detect_containment(objects: list[ObjectRecord]) -> list[ContainmentRelation]:
    """Return all (container, content) pairs where content.bbox ⊆ container.bbox.

    Only considers foreground objects (is_background=False).
    Different colors only — an object cannot contain a same-color piece.
    """
    fg = [o for o in objects if not o.is_background]
    relations: list[ContainmentRelation] = []
    for i, inner in enumerate(fg):
        for j, outer in enumerate(fg):
            if i == j or inner.color == outer.color:
                continue
            b_in  = inner.bbox
            b_out = outer.bbox
            if (b_out["r_min"] <= b_in["r_min"]
                    and b_out["r_max"] >= b_in["r_max"]
                    and b_out["c_min"] <= b_in["c_min"]
                    and b_out["c_max"] >= b_in["c_max"]
                    and outer.size > inner.size):
                relations.append(ContainmentRelation(container=outer, content=inner))
    return relations


def detect_spatial_relations(
    objects: list[ObjectRecord],
    max_distance: float = 40.0,
) -> list[SpatialRelation]:
    """Return pairwise directional relationships for all foreground object pairs
    within max_distance of each other (centroid-to-centroid).

    Each pair (A, B) appears once with A above/left of B.
    """
    fg = [o for o in objects if not o.is_background]
    relations: list[SpatialRelation] = []
    for i in range(len(fg)):
        for j in range(i + 1, len(fg)):
            a, b = fg[i], fg[j]
            dr = b.centroid[0] - a.centroid[0]   # positive = b is below a
            dc = b.centroid[1] - a.centroid[1]   # positive = b is right of a
            dist = math.hypot(dr, dc)
            if dist > max_distance:
                continue
            # Primary direction: describes where A is relative to B.
            # dr > 0 means b is lower (higher row) → a is above b.
            # dc > 0 means b is to the right → a is to the left of b.
            rel = ("above" if dr > 0 else "below") if abs(dr) >= abs(dc) \
                else ("left" if dc > 0 else "right")
            col_aligned = abs(dc) <= _COL_ALIGN_TOL
            row_aligned = abs(dr) <= _ROW_ALIGN_TOL
            relations.append(SpatialRelation(
                obj_a=a, obj_b=b,
                relation=rel,
                col_aligned=col_aligned,
                row_aligned=row_aligned,
                distance=round(dist, 1),
            ))
    return relations


def auto_detect_concepts(
    action_effects: dict,
    objects: list[ObjectRecord],
    concept_bindings: dict,
) -> dict[int, dict]:
    """Zero-cost heuristic concept detection that runs after every step.

    Scans action_effects for behavioral signatures and returns suggested
    concept_bindings updates — structured identically to OBSERVER output so
    they flow through the same merge path.

    Rules (all produce [GUESS] unless otherwise noted):
    - step_counter  : bar-shaped object (aspect ≥ 3 or ≤ 0.33) whose size
                      decreases by a consistent non-zero delta on every
                      observed action.  Confidence:
                        1 observation  → 0.35  (one shrink seen)
                        2 observations, consistent delta → 0.60
                        3+ observations, consistent delta → 0.85

    New signatures can be added here as new games are explored.
    Already-bound roles with confidence ≥ 0.7 are not downgraded.
    """
    obj_by_color: dict[int, ObjectRecord] = {o.color: o for o in objects}
    suggestions: dict[int, dict] = {}

    # --- Collect per-color size-change deltas across all actions/observations ---
    deltas_by_color: dict[int, list[int]] = {}
    for effects in action_effects.values():
        for obs in effects.get("object_observations", []):
            for ac in obs.get("attribute_changes", []):
                if "size" not in ac.get("changed", []):
                    continue
                color = ac.get("color")
                before = ac.get("before", {}).get("size")
                after  = ac.get("after",  {}).get("size")
                if color is None or before is None or after is None:
                    continue
                deltas_by_color.setdefault(color, []).append(after - before)

    # --- Step-counter signature ---
    for color, deltas in deltas_by_color.items():
        obj = obj_by_color.get(color)
        if obj is None:
            continue

        # Don't downgrade a role already bound with high confidence.
        existing = concept_bindings.get(color)
        if isinstance(existing, dict):
            if existing.get("role") not in (None, "step_counter"):
                continue          # already bound to something else
            if existing.get("confidence", 0) >= 0.7 and existing.get("role") == "step_counter":
                continue          # already confirmed, nothing to add

        # All observed deltas must be negative (strictly shrinking every action).
        if not deltas or not all(d < 0 for d in deltas):
            continue

        # Consistency: all deltas equal (constant depletion rate).
        consistent = len(set(deltas)) == 1
        n = len(deltas)

        if n == 1:
            conf = 0.35
        elif consistent:
            conf = 0.85 if n >= 3 else 0.60
        else:
            conf = 0.40   # shrinking but variable rate — weaker guess

        # Bar-shaped objects are the canonical step-counter form; penalise others.
        ar = obj.aspect_ratio
        if not (ar >= 3.0 or ar <= 0.33):
            conf *= 0.5   # not bar-shaped — still possible but less likely

        if conf < 0.25:
            continue

        suggestions[color] = {
            "role":          "step_counter",
            "confidence":    round(conf, 2),
            "observations":  n,
            "auto_detected": True,
        }

    return suggestions


def detect_contacts(
    player_colors: set[int],
    objects: list[ObjectRecord],
    margin: int = 5,
) -> dict[int, int]:
    """Return {color: min_gap} for non-player foreground objects within *margin* cells.

    min_gap is the minimum axis-aligned bounding-box distance in cells:
      0  — bboxes share at least one cell (player is physically on top of the object)
      1  — bboxes are adjacent (touching edges, no shared cells — side-by-side)
      N  — bboxes are N cells apart (near-contact, proximity trigger)

    Callers choose their own threshold:
      on_top_of = {c for c, g in contacts.items() if g == 0}
      touching  = {c for c, g in contacts.items() if g <= 1}
      nearby    = {c for c, g in contacts.items() if g <= 5}
      all_seen  = set(contacts)          # everything within margin

    margin=5 keeps one action-step worth of proximity in view.
    """
    player_objs = [o for o in objects if o.color in player_colors]
    target_objs = [o for o in objects
                   if o.color not in player_colors and not o.is_background]
    result: dict[int, int] = {}
    for po in player_objs:
        pb = po.bbox
        for to in target_objs:
            tb = to.bbox
            # Gap on each axis: 0 means bboxes share cells (actual overlap),
            # 1 means adjacent (touching edges but no shared cells).
            row_gap = max(0, max(pb["r_min"], tb["r_min"]) - min(pb["r_max"], tb["r_max"]))
            col_gap = max(0, max(pb["c_min"], tb["c_min"]) - min(pb["c_max"], tb["c_max"]))
            gap = max(row_gap, col_gap)   # Chebyshev-style: worst-axis gap
            if gap <= margin:
                prev = result.get(to.color)
                if prev is None or gap < prev:
                    result[to.color] = gap
    return result


def infer_action_directions(
    action_effects: dict,
) -> dict[str, tuple[int, int]]:
    """Derive (dr, dc) per action from accumulated action_effects observations.

    Scans all object_observations for non-background moved objects and returns
    the most commonly observed non-zero (dr, dc) per action name.

    Returns {} for actions with no non-stationary observations yet.
    """
    result: dict[str, tuple[int, int]] = {}
    for action_name, effects in action_effects.items():
        dr_votes: dict[int, int] = {}
        dc_votes: dict[int, int] = {}
        for obs in effects.get("object_observations", []):
            for mv in obs.get("moved", []):
                if mv.get("is_background"):
                    continue
                dr = mv.get("delta_r", 0)
                dc = mv.get("delta_c", 0)
                if dr != 0 or dc != 0:
                    dr_votes[dr] = dr_votes.get(dr, 0) + 1
                    dc_votes[dc] = dc_votes.get(dc, 0) + 1
        if dr_votes or dc_votes:
            best_dr = max(dr_votes, key=lambda k: dr_votes[k]) if dr_votes else 0
            best_dc = max(dc_votes, key=lambda k: dc_votes[k]) if dc_votes else 0
            best_dr = int(round(best_dr))
            best_dc = int(round(best_dc))
            if best_dr != 0 or best_dc != 0:
                result[action_name] = (best_dr, best_dc)
    return result


def detect_arena_delta(
    frame_before: list,
    frame_after: list,
    player_colors: set[int],
    min_size_change: int = 3,
) -> dict:
    """Compare two frames — ignoring player-colored pixels — to identify world-state changes.

    Filters out the player object so that only changes in the environment
    (not the player's own movement) are reported.  Useful for inferring causal
    effects of contact events: "what changed in the world when the player
    touched this object?"

    Parameters
    ----------
    frame_before, frame_after : list[list[int]]
        Frames immediately before and after the action that caused contact.
    player_colors : set[int]
        Colors belonging to the player (excluded from comparison).
    min_size_change : int
        Minimum absolute pixel-count change to report a size change (noise filter).

    Returns
    -------
    dict with keys:
        "appeared"     : list of {color, size, centroid, bbox} — new non-player objects
        "disappeared"  : list of {color, size, centroid, bbox} — objects that vanished
        "changed_size" : list of {color, before_size, after_size, delta} — resized objects
        "any_change"   : bool — True if any of the above lists is non-empty
    """
    def _non_player_objs(frame: list) -> dict[int, "ObjectRecord"]:
        return {
            o.color: o
            for o in detect_objects(frame)
            if o.color not in player_colors and not o.is_background
        }

    before_objs = _non_player_objs(frame_before)
    after_objs  = _non_player_objs(frame_after)
    all_colors  = set(before_objs) | set(after_objs)

    appeared:     list[dict] = []
    disappeared:  list[dict] = []
    changed_size: list[dict] = []

    for color in all_colors:
        bef = before_objs.get(color)
        aft = after_objs.get(color)
        if bef is None and aft is not None:
            appeared.append({
                "color": color,
                "size": aft.size,
                "centroid": aft.centroid,
                "bbox": {"r_min": aft.r_min, "r_max": aft.r_max,
                         "c_min": aft.c_min, "c_max": aft.c_max},
            })
        elif bef is not None and aft is None:
            disappeared.append({
                "color": color,
                "size": bef.size,
                "centroid": bef.centroid,
                "bbox": {"r_min": bef.r_min, "r_max": bef.r_max,
                         "c_min": bef.c_min, "c_max": bef.c_max},
            })
        elif bef is not None and aft is not None:
            delta = aft.size - bef.size
            if abs(delta) >= min_size_change:
                changed_size.append({
                    "color": color,
                    "before_size": bef.size,
                    "after_size":  aft.size,
                    "delta":       delta,
                })

    any_change = bool(appeared or disappeared or changed_size)
    return {
        "appeared":     appeared,
        "disappeared":  disappeared,
        "changed_size": changed_size,
        "any_change":   any_change,
    }


def _compress_route(route: list[str]) -> str:
    """Compress ['ACTION1','ACTION1','ACTION3'] -> '2xACTION1 + 1xACTION3'."""
    if not route:
        return "(no moves)"
    parts: list[str] = []
    cur, count = route[0], 1
    for act in route[1:]:
        if act == cur:
            count += 1
        else:
            parts.append(f"{count}x{cur}")
            cur, count = act, 1
    parts.append(f"{count}x{cur}")
    return " + ".join(parts)


def plan_route(
    frame: list,
    player_objects: list[ObjectRecord],
    target_centroid: tuple[float, float],
    action_directions: dict[str, tuple[int, int]],
    max_steps: int = 20,
    target_colors: set[int] | None = None,
) -> list[str] | None:
    """BFS path planner in action-space.

    Finds the shortest sequence of actions that moves the player so its
    bounding box overlaps or is adjacent to the target centroid.  Returns
    None if the target is unreachable within max_steps, or if inputs are
    insufficient (no player, no action directions).

    Parameters
    ----------
    frame : list[list[int]]
        Current game frame — used to build the passability map.
    player_objects : list[ObjectRecord]
        All objects identified as the player (dynamic colors).
    target_centroid : (row, col)
        Goal centroid to reach.
    action_directions : {action_name: (dr, dc)}
        Confirmed movement deltas per action.  Must have at least one entry.
    max_steps : int
        BFS depth limit (default 20).
    target_colors : set[int], optional
        Colors of all exploration target objects.  Their cells are removed
        from the blocked set so the player can navigate into them.  Pass all
        uncontacted object colors together so co-located objects of different
        colors (e.g., a cross made of two colors) don't block each other.
    """
    from collections import deque

    if not player_objects or not action_directions:
        return None

    frame_h = len(frame)
    frame_w = len(frame[0]) if frame else 0
    if frame_h == 0 or frame_w == 0:
        return None

    player_colors = {o.color for o in player_objects}

    # Passability map.  Only SMALL-TO-MEDIUM objects are treated as walls.
    # Large non-background objects (> 10% of frame area) are arena surfaces
    # that the player navigates ON — they must remain passable.  This avoids
    # erroneously blocking the arena interior when it uses the same detection
    # threshold as foreground walls but is structurally different.
    #
    # Rule: block a cell only if it belongs to an object that is:
    #   - not background (is_background=False)
    #   - not the player
    #   - not an exploration target
    #   - smaller than 10% of the total frame area
    #
    # This correctly passes arena surfaces (typically 15-30% of frame) while
    # blocking walls and UI elements (typically 0.1-6% of frame).
    _passable_targets = target_colors or set()
    objects = detect_objects(frame)
    total_cells = frame_h * frame_w
    wall_size_limit = total_cells * 0.10   # anything larger is arena, not wall
    blocked: set[tuple[int, int]] = set()
    for o in objects:
        if o.is_background or o.color in player_colors or o.color in _passable_targets:
            continue
        if o.size > wall_size_limit:
            continue  # large object — treat as arena surface (passable)
        for r in range(o.bbox["r_min"], o.bbox["r_max"] + 1):
            for c in range(o.bbox["c_min"], o.bbox["c_max"] + 1):
                if r < frame_h and c < frame_w and frame[r][c] == o.color:
                    blocked.add((r, c))

    tr, tc = target_centroid

    def _bfs_from(candidate: "ObjectRecord") -> list[str] | None:
        """Run BFS starting from a specific candidate player object."""
        p_h = int(candidate.bbox["r_max"]) - int(candidate.bbox["r_min"]) + 1
        p_w = int(candidate.bbox["c_max"]) - int(candidate.bbox["c_min"]) + 1
        start_r = int(candidate.bbox["r_min"])
        start_c = int(candidate.bbox["c_min"])

        def centroid_of(r_min: int, c_min: int) -> tuple[float, float]:
            return (r_min + p_h / 2, c_min + p_w / 2)

        def is_valid(r_min: int, c_min: int) -> bool:
            if r_min < 0 or c_min < 0:
                return False
            if r_min + p_h > frame_h or c_min + p_w > frame_w:
                return False
            for r in range(r_min, r_min + p_h):
                for c in range(c_min, c_min + p_w):
                    if (r, c) in blocked:
                        return False
            return True

        def is_goal(r_min: int, c_min: int) -> bool:
            # Require the player bbox to actually contain the target centroid
            # (same semantics as detect_contacts gap=0: actual cell overlap).
            tr_int = int(round(tr))
            tc_int = int(round(tc))
            return (r_min <= tr_int < r_min + p_h and
                    c_min <= tc_int < c_min + p_w)

        start = (start_r, start_c)
        if not is_valid(start_r, start_c):
            return None

        visited: set[tuple[int, int]] = {start}
        queue: deque = deque([(start, [])])

        while queue:
            (r, c), path = queue.popleft()
            if len(path) >= max_steps:
                continue
            for action_name, (dr, dc) in action_directions.items():
                nr, nc = r + int(dr), c + int(dc)
                if (nr, nc) in visited:
                    continue
                if not is_valid(nr, nc):
                    continue
                new_path = path + [action_name]
                if is_goal(nr, nc):
                    return new_path
                visited.add((nr, nc))
                queue.append(((nr, nc), new_path))
        return None

    # Try each candidate player object as the starting position; return the
    # shortest successful route.  This handles cases where the same color
    # appears in both the UI (static) and the playing area (dynamic).
    best_route: list[str] | None = None
    for candidate in player_objects:
        route = _bfs_from(candidate)
        if route is not None:
            if best_route is None or len(route) < len(best_route):
                best_route = route

    return best_route


def _nav_hint(player_centroid: tuple, target_centroid: tuple) -> str:
    """Straight-line nav hint ignoring walls (fallback when BFS unavailable)."""
    dr = target_centroid[0] - player_centroid[0]
    dc = target_centroid[1] - player_centroid[1]
    parts = []
    if abs(dr) >= 2.5:
        n = round(abs(dr) / 5)
        parts.append(f"{n}x{'ACTION2' if dr > 0 else 'ACTION1'}")
    if abs(dc) >= 2.5:
        n = round(abs(dc) / 5)
        parts.append(f"{n}x{'ACTION4' if dc > 0 else 'ACTION3'}")
    if not parts:
        return "already adjacent"
    return " + ".join(parts)


def format_structural_context(
    frame: list,
    concept_bindings: dict | None = None,
    known_dynamic_colors: set[int] | None = None,
    explored_colors: set[int] | None = None,
    action_directions: dict[str, tuple[int, int]] | None = None,
    contact_events: list[dict] | None = None,
) -> str:
    """Produce a human-readable structural context string for the OBSERVER prompt.

    Highlights:
    - Confirmed action directions (skip re-characterization)
    - Container/content pairs (bordered boxes with patterns inside)
    - Column and row alignments between objects (especially player ↔ goal)
    - Objects whose color has been seen moving (dynamic) vs never moved (static)
    - Contact events: causal world-state changes observed when player touched objects
    - Exploration manifest with BFS-computed routes (or straight-line fallback)

    Parameters
    ----------
    frame : list[list[int]]
        Current frame.
    concept_bindings : dict, optional
        {color_int: role | {"role":..., ...}} for labeling objects by role.
    known_dynamic_colors : set[int], optional
        Colors observed moving in previous steps.
    explored_colors : set[int], optional
        Colors the player has already made full contact with.
    action_directions : dict, optional
        {action_name: (dr, dc)} confirmed from observations.  When provided,
        the exploration manifest shows BFS-computed exact routes instead of
        straight-line approximations, and a directions summary is prepended
        so the MEDIATOR does not waste steps re-characterizing actions.
    contact_events : list[dict], optional
        Records of world-state changes observed when the player touched specific
        objects.  Each entry: {"touched_color": int, "step": int, "delta": dict}
        where delta has "appeared", "disappeared", "changed_size" sub-lists.
        Rendered as a "Contact history" section so agents can reason causally
        about which objects unlock other objects or trigger state transitions.
    """
    objects = detect_objects(frame)
    containment = detect_containment(objects)
    spatial = detect_spatial_relations(objects)
    bindings = concept_bindings or {}
    dynamic = known_dynamic_colors or set()
    explored = explored_colors or set()
    directions = action_directions or {}
    events = contact_events or []

    def _role(color: int) -> str:
        raw = bindings.get(color)
        if isinstance(raw, dict):
            return raw.get("role", "")
        return raw or ""

    def _label(o: ObjectRecord) -> str:
        role = _role(o.color)
        name = f"{color_name(o.color)}"
        if role:
            name = f"{role}({name})"
        nature = "dynamic" if o.color in dynamic else "static"
        return f"{name} size={o.size} {o.width}w×{o.height}h @{_fmt_pt(o.centroid)} [{nature}]"

    lines: list[str] = []

    # --- Confirmed action directions (suppresses re-characterization) ---
    if directions:
        dir_labels = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
        dir_parts = []
        for act in sorted(directions):
            dr, dc = directions[act]
            # Normalise to unit direction for label
            label = dir_labels.get((0 if dr == 0 else (-1 if dr < 0 else 1),
                                    0 if dc == 0 else (-1 if dc < 0 else 1)), "?")
            dist = abs(dr) if dr != 0 else abs(dc)
            dir_parts.append(f"{act}={label}{dist}")
        lines.append(
            "Action directions (confirmed — do NOT waste steps re-characterizing): "
            + "  ".join(dir_parts)
        )

    # --- Containers ---
    if containment:
        lines.append("Containers (bordered regions enclosing other objects):")
        # Group by container
        by_container: dict[int, list[ContainmentRelation]] = {}
        for rel in containment:
            key = id(rel.container)
            by_container.setdefault(key, []).append(rel)
        for rels in by_container.values():
            c = rels[0].container
            lines.append(f"  BOX  {_label(c)}")
            for rel in rels:
                lines.append(f"    └─ {_label(rel.content)}")

    # --- Column/row alignments (most useful for navigation) ---
    # Skip container↔content pairs — already shown in Containers section.
    container_content_pairs: set[frozenset] = {
        frozenset({id(r.container), id(r.content)}) for r in containment
    }
    align_lines: list[str] = []
    for rel in spatial:
        if not (rel.col_aligned or rel.row_aligned):
            continue
        if frozenset({id(rel.obj_a), id(rel.obj_b)}) in container_content_pairs:
            continue
        kind = []
        if rel.col_aligned:
            kind.append("same-col")
        if rel.row_aligned:
            kind.append("same-row")
        align_lines.append(
            f"  {_label(rel.obj_a)}  {rel.relation}  {_label(rel.obj_b)}"
            f"  [{', '.join(kind)}, dist={rel.distance}]"
        )
    if align_lines:
        lines.append("Spatial alignments:")
        lines.extend(align_lines)

    # --- Uncontained static foreground objects not yet role-bound ---
    contained_ids = {id(r.content) for r in containment}
    contained_ids |= {id(r.container) for r in containment}
    orphan_static = [
        o for o in objects
        if not o.is_background
        and o.color not in dynamic
        and id(o) not in contained_ids
    ]
    if orphan_static:
        lines.append("Other static foreground objects (not yet role-bound):")
        for o in orphan_static:
            lines.append(f"  {_label(o)}")

    # --- Exploration manifest -------------------------------------------
    # All non-background, non-player foreground objects, grouped by whether
    # the player has already made contact with them.  Gives MEDIATOR a clear
    # checklist: what is still unknown and how to reach it.
    fg_non_player = [o for o in objects
                     if not o.is_background and o.color not in dynamic]

    # Determine player centroid for navigation hints
    player_objs = [o for o in objects if o.color in dynamic]
    if player_objs and fg_non_player:
        # Use centroid of the largest dynamic object as the player reference
        player_ref = max(player_objs, key=lambda o: o.size)
        p_centroid = player_ref.centroid

        untouched = [o for o in fg_non_player if o.color not in explored]
        touched   = [o for o in fg_non_player if o.color in explored]

        lines.append("")
        lines.append(
            f"Exploration manifest  "
            f"({len(touched)} contacted, {len(untouched)} not yet contacted):"
        )
        if touched:
            lines.append("  Contacted:")
            for o in touched:
                lines.append(f"    [DONE] {_label(o)}")
        if untouched:
            lines.append("  NOT YET CONTACTED — behavior unknown:")
            player_objs_list = [o for o in objects if o.color in dynamic]
            # All untouched colors are made passable together so co-located
            # objects of different colors (e.g., a cross of two colors) don't
            # block each other during BFS routing.
            untouched_colors = {o.color for o in untouched}
            for o in untouched:
                if directions:
                    route = plan_route(
                        frame, player_objs_list, o.centroid, directions,
                        max_steps=30,
                        target_colors=untouched_colors,
                    )
                    if route:
                        nav_str = f"route: {_compress_route(route)}"
                    else:
                        # BFS couldn't find a path — walls or arena shape block the
                        # straight approach.  Flag for LLM-assisted route planning.
                        nav_str = (
                            f"nav≈{_nav_hint(p_centroid, o.centroid)} "
                            f"[NEEDS-PLANNING: BFS found no direct route — "
                            f"reason likely a wall or arena gap; plan a detour]"
                        )
                else:
                    nav_str = f"nav≈{_nav_hint(p_centroid, o.centroid)}"
                lines.append(f"    [TODO] {_label(o)}  {nav_str}")

    # --- Contact history (causal world-state changes from touch events) ---
    # Each entry records what changed in the environment when the player
    # made full contact with a specific object color.  This lets agents
    # reason: "touching colorX caused colorY to appear — I should go touch Y."
    if events:
        lines.append("")
        lines.append("Contact history (world changes observed when player touched an object):")
        for ev in events:
            color = ev.get("touched_color")
            step  = ev.get("step", "?")
            delta = ev.get("delta", {})
            effects: list[str] = []
            for a in delta.get("appeared", []):
                cr, cc = a["centroid"]
                effects.append(
                    f"color{a['color']}({color_name(a['color'])}) APPEARED "
                    f"size={a['size']} @({cr:.0f},{cc:.0f})"
                )
            for d in delta.get("disappeared", []):
                cr, cc = d["centroid"]
                effects.append(
                    f"color{d['color']}({color_name(d['color'])}) DISAPPEARED "
                    f"(was size={d['size']} @({cr:.0f},{cc:.0f}))"
                )
            for ch in delta.get("changed_size", []):
                sign = "+" if ch["delta"] > 0 else ""
                effects.append(
                    f"color{ch['color']}({color_name(ch['color'])}) size changed "
                    f"{ch['before_size']}->{ch['after_size']} ({sign}{ch['delta']})"
                )
            if effects:
                lines.append(
                    f"  [step {step}] touching color{color}({color_name(color)}) caused:"
                )
                for eff in effects:
                    lines.append(f"    → {eff}")
            else:
                lines.append(
                    f"  [step {step}] touching color{color}({color_name(color)}): "
                    f"no detected world change (may be a passive/inert object)"
                )

    return "\n".join(lines) if lines else "  (no structural context detected)"
