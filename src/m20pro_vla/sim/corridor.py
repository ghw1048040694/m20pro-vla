"""S3 corridor scene family: one corridor serving several rooms.

Why this exists
---------------
S2 (``sim/rooms.py``) puts every task object in a single room behind one
doorway.  The robot's decision is still shallow: walk to the door, look in,
pick a side.  A VLA trained only on S2 would learn "drive forward, then look",
which is exactly the behaviour that fails to transfer to a building.

S3 adds the structure that makes the task a *building* rather than a room:

* one straight corridor with an entrance at one end;
* several rooms hung off that corridor (a side wing on each long wall, plus a
  room at the far end);
* **each task object in a different room**, so "find the red cube" requires
  locating the right doorway, not just any doorway.

The robot starts outside the entrance.  Every object is occluded from that
pose by construction, and every room is sealed: the grid BFS in
:func:`reachability_report` proves both halves separately (rooms reachable with
the doorways open, no room reachable with the doorways plugged).

Everything lowers to plain :class:`~m20pro_vla.sim.mujoco.ObstacleSpec` boxes,
so the scene is materialised by the existing
:func:`~m20pro_vla.sim.mujoco.build_scene` pipeline: no new asset, no new
dynamics, and no change to ``mujoco.py`` or to the S2 module.

Ownership rule (this is what keeps the walls from doubling up)
--------------------------------------------------------------
A wall shared between the corridor and a wing is owned **by the wing**: the
wing draws its own four sides (door on the shared side) and the corridor omits
that stretch of wall entirely.  Wing interiors therefore sit exactly one wall
thickness away from the corridor interior:

    north wing:  wing.y0 == corridor.y1 + t
    south wing:  wing.y1 == corridor.y0 - t
    east  wing:  wing.x0 == corridor.x1 + t
    west  wing:  wing.x1 == corridor.x0 - t

:func:`touching_side` recovers the attachment from the geometry, and
:func:`validate_multiroom` rejects a wing that does not satisfy it.  Nothing is
inferred from a hand-written "door_wall" field, so the layout cannot silently
disagree with itself.

Scene tiers
-----------
S1  open plane, 3 objects, 1 screen wall        (regression baseline)
S2  single room, four walls, one doorway        (sim/rooms.py; training)
S3  corridor + three rooms, object per room     (this module; training/OOD)
"""

from __future__ import annotations

import dataclasses
from collections import deque
from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

from .mujoco import OBJECTS, ObjectSpec, ObstacleSpec
from .rooms import (
    DOOR_CLEARANCE_MARGIN_M,
    DOOR_CLEAR_WIDTH_M,
    DOOR_EDGE_MARGIN_M,
    MIN_OBJECT_SEPARATION_M,
    OBJECT_HALF_SIZES,
    ROBOT_FOOTPRINT_DIAMETER_M,
    ROBOT_FOOTPRINT_RADIUS_M,
    SENSOR_HIGH_HEIGHT_M,
    SENSOR_LOW_HEIGHT_M,
    WALL_HEIGHT_M,
    WALL_RGBA,
    WALL_SIDES,
    WALL_THICKNESS_M,
    RoomSpec,
    line_of_sight_blocked,
    room_obstacles,
)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

# The corridor has to be wide enough that a centred doorway still leaves a wall
# stub at each corner (DOOR_EDGE_MARGIN_M) instead of eating the whole wall.
CORRIDOR_WIDTH_M = 2.20
MIN_CORRIDOR_WIDTH_M = DOOR_CLEAR_WIDTH_M + 2.0 * DOOR_EDGE_MARGIN_M
MIN_CORRIDOR_LENGTH_M = 3.00

MIN_ROOM_SIDE_M = 2.20
# Objects sit this far from every wall of their own room.  Combined with the
# footprint radius this keeps the object's own cell reachable, so a room that
# validates is a room the robot can actually stand next to.
ROOM_OBJECT_MARGIN_M = 0.60

OPPOSITE_SIDE = {"west": "east", "east": "west", "south": "north", "north": "south"}

S3_CORRIDOR_X = (1.00, 6.60)
S3_CORRIDOR_Y = (-1.10, 1.10)
S3_START_XY = (-1.00, 0.00)
S3_START_YAW = 0.0

# Canonical placement: one object per room, each deep enough inside its room
# that the start-to-object ray is stopped by a long stretch of wall.
S3_OBJECT_POSITIONS: dict[str, tuple[float, float]] = {
    "green_cylinder": (3.40, 2.90),  # north wing
    "yellow_box": (5.40, -2.60),     # south wing
    "red_cube": (8.20, 0.95),        # east end room
}


@dataclass(frozen=True)
class WingSpec:
    """A rectangular room hung off the corridor, with one doorway.

    ``door_center`` is expressed in the wing's own frame along the shared wall:
    an x coordinate for a wing attached to the corridor's north or south wall,
    a y coordinate for one attached to the east or west wall.
    """

    name: str
    interior_x: tuple[float, float]
    interior_y: tuple[float, float]
    door_center: float
    door_width: float = DOOR_CLEAR_WIDTH_M
    object_name: str = ""

    @property
    def interior_center(self) -> tuple[float, float]:
        return (
            0.5 * (self.interior_x[0] + self.interior_x[1]),
            0.5 * (self.interior_y[0] + self.interior_y[1]),
        )

    @property
    def width(self) -> float:
        return float(self.interior_x[1] - self.interior_x[0])

    @property
    def height(self) -> float:
        return float(self.interior_y[1] - self.interior_y[0])

    def door_gap(self) -> tuple[float, float]:
        half = 0.5 * float(self.door_width)
        return (float(self.door_center) - half, float(self.door_center) + half)


def default_wings() -> tuple[WingSpec, ...]:
    """The canonical three-room layout: north wing, south wing, east end room."""
    return (
        WingSpec(
            name="north_room",
            interior_x=(1.80, 4.20),
            interior_y=(1.22, 3.72),
            door_center=3.00,
            object_name="green_cylinder",
        ),
        WingSpec(
            name="south_room",
            interior_x=(4.20, 6.60),
            interior_y=(-3.72, -1.22),
            door_center=5.40,
            object_name="yellow_box",
        ),
        WingSpec(
            name="end_room",
            interior_x=(6.72, 9.22),
            interior_y=(-1.60, 1.60),
            door_center=0.00,
            object_name="red_cube",
        ),
    )


@dataclass(frozen=True)
class CorridorSpec:
    """One corridor with an entrance and a set of attached rooms."""

    name: str = "s3_corridor_rooms"
    corridor_x: tuple[float, float] = S3_CORRIDOR_X
    corridor_y: tuple[float, float] = S3_CORRIDOR_Y
    entrance_side: str = "west"
    entrance_center: float = 0.00
    entrance_width: float = DOOR_CLEAR_WIDTH_M
    rooms: tuple[WingSpec, ...] = field(default_factory=default_wings)
    wall_thickness: float = WALL_THICKNESS_M
    wall_height: float = WALL_HEIGHT_M
    rgba: tuple[float, float, float, float] = WALL_RGBA

    @property
    def corridor_length(self) -> float:
        return float(self.corridor_x[1] - self.corridor_x[0])

    @property
    def corridor_width(self) -> float:
        return float(self.corridor_y[1] - self.corridor_y[0])

    @property
    def corridor_center(self) -> tuple[float, float]:
        return (
            0.5 * (self.corridor_x[0] + self.corridor_x[1]),
            0.5 * (self.corridor_y[0] + self.corridor_y[1]),
        )

    def entrance_gap(self) -> tuple[float, float]:
        half = 0.5 * float(self.entrance_width)
        return (float(self.entrance_center) - half, float(self.entrance_center) + half)

    def wing_named(self, name: str) -> WingSpec | None:
        for wing in self.rooms:
            if wing.name == name:
                return wing
        return None


@dataclass(frozen=True)
class CorridorEpisode:
    """One S3 scene instance: building geometry + task objects + start pose."""

    spec: CorridorSpec = field(default_factory=CorridorSpec)
    objects: tuple[ObjectSpec, ...] = ()
    start_xy: tuple[float, float] = S3_START_XY
    start_yaw: float = S3_START_YAW


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def touching_side(wing: WingSpec, spec: CorridorSpec) -> str | None:
    """Return the corridor side this wing hangs off, or ``None`` if detached.

    The comparison is against the exact one-wall-thickness offset, so a wing
    that merely overlaps the corridor is *not* accepted as attached.
    """
    t = float(spec.wall_thickness)
    x0, x1 = spec.corridor_x
    y0, y1 = spec.corridor_y
    tolerance = 1.0e-6
    if abs(wing.interior_y[0] - (y1 + t)) <= tolerance:
        return "north"
    if abs(wing.interior_y[1] - (y0 - t)) <= tolerance:
        return "south"
    if abs(wing.interior_x[0] - (x1 + t)) <= tolerance:
        return "east"
    if abs(wing.interior_x[1] - (x0 - t)) <= tolerance:
        return "west"
    return None


def _corridor_centrelines(
    spec: CorridorSpec,
) -> dict[str, tuple[str, float, float, float]]:
    """Map each corridor side to (axis, centreline, span_lo, span_hi).

    The span runs one wall thickness past both interior corners so the four
    corridor walls close the corners instead of leaving a diagonal leak.
    """
    x0, x1 = spec.corridor_x
    y0, y1 = spec.corridor_y
    t = float(spec.wall_thickness)
    half = 0.5 * t
    return {
        "west": ("y", x0 - half, y0 - t, y1 + t),
        "east": ("y", x1 + half, y0 - t, y1 + t),
        "south": ("x", y0 - half, x0 - t, x1 + t),
        "north": ("x", y1 + half, x0 - t, x1 + t),
    }


def _wall_box(
    name: str,
    axis: str,
    line: float,
    lo: float,
    hi: float,
    thickness: float,
    height: float,
    rgba: tuple[float, float, float, float],
) -> ObstacleSpec:
    """Build one axis-aligned wall box spanning ``[lo, hi]`` on ``line``."""
    half_t = 0.5 * float(thickness)
    half_h = 0.5 * float(height)
    half_len = 0.5 * float(hi - lo)
    centre = 0.5 * float(lo + hi)
    if axis == "y":
        position = (float(line), centre, half_h)
        size = (half_t, half_len, half_h)
    else:
        position = (centre, float(line), half_h)
        size = (half_len, half_t, half_h)
    return ObstacleSpec(name=name, kind="box", position=position, size=size, rgba=rgba)


def _segments_around_gaps(
    lo: float, hi: float, gaps: list[tuple[float, float]], minimum: float = 1.0e-9
) -> list[tuple[float, float]]:
    """Complement of ``gaps`` inside ``[lo, hi]``, clamped and sorted."""
    clipped = sorted(
        (max(float(lo), g_lo), min(float(hi), g_hi))
        for g_lo, g_hi in gaps
        if g_hi > lo and g_lo < hi
    )
    segments: list[tuple[float, float]] = []
    cursor = float(lo)
    for gap_lo, gap_hi in clipped:
        if gap_lo > cursor:
            segments.append((cursor, gap_lo))
        cursor = max(cursor, gap_hi)
    if cursor < float(hi):
        segments.append((cursor, float(hi)))
    return [(a, b) for a, b in segments if b - a > minimum]


def _corridor_attachment_spans(
    spec: CorridorSpec,
) -> dict[str, list[tuple[float, float]]]:
    """Per corridor side, the stretches removed because a wing owns that wall."""
    t = float(spec.wall_thickness)
    spans: dict[str, list[tuple[float, float]]] = {}
    for wing in spec.rooms:
        side = touching_side(wing, spec)
        if side is None:
            continue
        if side in ("north", "south"):
            span = (wing.interior_x[0] - t, wing.interior_x[1] + t)
        else:
            span = (wing.interior_y[0] - t, wing.interior_y[1] + t)
        spans.setdefault(side, []).append(span)
    return spans


def corridor_walls(spec: CorridorSpec) -> list[ObstacleSpec]:
    """Corridor walls, with the entrance and every wing attachment left open."""
    attachments = _corridor_attachment_spans(spec)
    entrance = spec.entrance_gap()
    walls: list[ObstacleSpec] = []
    for side, (axis, line, lo, hi) in _corridor_centrelines(spec).items():
        gaps = list(attachments.get(side, ()))
        if side == spec.entrance_side:
            gaps.append(entrance)
        for index, (segment_lo, segment_hi) in enumerate(_segments_around_gaps(lo, hi, gaps)):
            walls.append(
                _wall_box(
                    f"corridor_wall_{side}_{index}",
                    axis,
                    line,
                    segment_lo,
                    segment_hi,
                    spec.wall_thickness,
                    spec.wall_height,
                    spec.rgba,
                )
            )
    return walls


def wing_walls(wing: WingSpec, spec: CorridorSpec, *, strict: bool = True) -> list[ObstacleSpec]:
    """The wing's three non-shared walls; the shared wall is its door wall.

    With ``strict=False`` a detached wing yields no walls instead of raising:
    validation has to be able to *report* a detached wing rather than crash on
    the very input it exists to reject.
    """
    side = touching_side(wing, spec)
    if side is None:
        if strict:
            raise ValueError(f"wing {wing.name!r} is not attached to the corridor")
        return []
    shared = OPPOSITE_SIDE[side]
    room = RoomSpec(
        name=wing.name,
        interior_x=wing.interior_x,
        interior_y=wing.interior_y,
        door_wall=shared,
        door_center=wing.door_center,
        door_width=wing.door_width,
        wall_thickness=spec.wall_thickness,
        wall_height=spec.wall_height,
        rgba=spec.rgba,
    )
    walls: list[ObstacleSpec] = []
    for wall in room_obstacles(room):
        _, _, wall_side, index = wall.name.split("_")
        if wall_side == shared:
            continue
        walls.append(dataclasses.replace(wall, name=f"wing_{wing.name}_{wall_side}_{index}"))
    return walls


def shared_door_wall(
    wing: WingSpec, spec: CorridorSpec, *, strict: bool = True
) -> list[ObstacleSpec]:
    """The wing's own wall on the shared side, carrying the doorway."""
    side = touching_side(wing, spec)
    if side is None:
        if strict:
            raise ValueError(f"wing {wing.name!r} is not attached to the corridor")
        return []
    shared = OPPOSITE_SIDE[side]
    room = RoomSpec(
        name=wing.name,
        interior_x=wing.interior_x,
        interior_y=wing.interior_y,
        door_wall=shared,
        door_center=wing.door_center,
        door_width=wing.door_width,
        wall_thickness=spec.wall_thickness,
        wall_height=spec.wall_height,
        rgba=spec.rgba,
    )
    walls = []
    for wall in room_obstacles(room):
        _, _, wall_side, index = wall.name.split("_")
        if wall_side != shared:
            continue
        walls.append(dataclasses.replace(wall, name=f"wing_{wing.name}_door_{index}"))
    return walls


def all_walls(spec: CorridorSpec, *, strict: bool = True) -> list[ObstacleSpec]:
    """Every wall box in the building, in a deterministic order.

    ``strict=False`` skips wings that are not attached (used by validation, so
    an invalid spec produces a report rather than an exception).
    """
    walls = corridor_walls(spec)
    for wing in spec.rooms:
        walls.extend(wing_walls(wing, spec, strict=strict))
        walls.extend(shared_door_wall(wing, spec, strict=strict))
    return walls


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1] + 1.0e-9:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(a, b) for a, b in merged]


def boundary_intervals(spec: CorridorSpec) -> dict[str, dict]:
    """Covered stretches and remaining gaps along each corridor wall line.

    Every wall box whose footprint touches the wall's thickness band
    contributes its projected interval, whether that box belongs to the
    corridor or to an attached wing.  The complement inside the corridor span
    is therefore the exact set of openings on that side.

    This is the structural invariant that keeps the shared walls honest: the
    corridor omits the stretch a wing owns, so if the wing's own wall were
    missing (or misaligned) the coverage would show a hole that is not a
    doorway.  ``validate_multiroom`` checks the gaps against the doorways.
    """
    t = float(spec.wall_thickness)
    half = 0.5 * t
    walls = all_walls(spec, strict=False)
    report: dict[str, dict] = {}
    for side, (axis, line, lo, hi) in _corridor_centrelines(spec).items():
        band_lo, band_hi = line - half, line + half
        intervals: list[tuple[float, float]] = []
        for wall in walls:
            wx, wy = float(wall.position[0]), float(wall.position[1])
            hx, hy = float(wall.size[0]), float(wall.size[1])
            if axis == "y":
                if wx - hx > band_hi + 1.0e-9 or wx + hx < band_lo - 1.0e-9:
                    continue
                wall_lo, wall_hi = wy - hy, wy + hy
            else:
                if wy - hy > band_hi + 1.0e-9 or wy + hy < band_lo - 1.0e-9:
                    continue
                wall_lo, wall_hi = wx - hx, wx + hx
            if wall_hi <= lo or wall_lo >= hi:
                continue
            intervals.append((max(float(lo), wall_lo), min(float(hi), wall_hi)))
        merged = _merge_intervals(intervals)
        report[side] = {
            "span": (float(lo), float(hi)),
            "covered": merged,
            "gaps": _segments_around_gaps(lo, hi, intervals),
        }
    return report


def door_plug_walls(spec: CorridorSpec) -> list[ObstacleSpec]:
    """Walls that would close every opening: used by the seal test."""
    walls: list[ObstacleSpec] = []
    for wing in spec.rooms:
        side = touching_side(wing, spec)
        if side is None:
            continue
        shared = OPPOSITE_SIDE[side]
        axis, line, _, _ = _corridor_centrelines(spec)[side]
        gap_lo, gap_hi = wing.door_gap()
        if shared in ("north", "south"):
            plug_line = wing.interior_y[0] - 0.5 * spec.wall_thickness if shared == "south" else wing.interior_y[1] + 0.5 * spec.wall_thickness
        else:
            plug_line = wing.interior_x[0] - 0.5 * spec.wall_thickness if shared == "west" else wing.interior_x[1] + 0.5 * spec.wall_thickness
        walls.append(
            _wall_box(
                f"seal_{wing.name}_door",
                axis,
                plug_line,
                gap_lo,
                gap_hi,
                spec.wall_thickness,
                spec.wall_height,
                spec.rgba,
            )
        )
    entrance = spec.entrance_gap()
    axis, line, _, _ = _corridor_centrelines(spec)[spec.entrance_side]
    walls.append(
        _wall_box(
            "seal_entrance",
            axis,
            line,
            entrance[0],
            entrance[1],
            spec.wall_thickness,
            spec.wall_height,
            spec.rgba,
        )
    )
    return walls


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_multiroom(spec: CorridorSpec) -> dict:
    """Check the building is physically solvable before anything is collected.

    Every rule is a hard constraint: a building that fails one of them cannot
    yield a single valid demonstration episode.
    """
    issues: list[str] = []
    t = float(spec.wall_thickness)
    x0, x1 = spec.corridor_x
    y0, y1 = spec.corridor_y
    required_door = ROBOT_FOOTPRINT_DIAMETER_M + DOOR_CLEARANCE_MARGIN_M
    door_widths = [float(wing.door_width) for wing in spec.rooms] or [float(spec.entrance_width)]

    metrics: dict = {
        "building": spec.name,
        "corridor_length_m": spec.corridor_length,
        "corridor_width_m": spec.corridor_width,
        "room_count": len(spec.rooms),
        "min_door_width_m": float(min(door_widths)),
        "robot_footprint_diameter_m": ROBOT_FOOTPRINT_DIAMETER_M,
        "door_required_width_m": required_door,
        "wall_height_m": float(spec.wall_height),
        "wall_thickness_m": t,
        "entrance_side": spec.entrance_side,
        "entrance_clear_width_m": float(spec.entrance_width),
    }

    if spec.corridor_length <= 0.0 or spec.corridor_width <= 0.0:
        issues.append("corridor extent must be positive")
    if spec.corridor_width < MIN_CORRIDOR_WIDTH_M - 1.0e-9:
        issues.append(
            f"corridor width {spec.corridor_width:.2f} m < {MIN_CORRIDOR_WIDTH_M:.2f} m "
            f"(doorway {DOOR_CLEAR_WIDTH_M:.2f} m + two {DOOR_EDGE_MARGIN_M:.2f} m corner stubs)"
        )
    if spec.corridor_length < MIN_CORRIDOR_LENGTH_M:
        issues.append(
            f"corridor length {spec.corridor_length:.2f} m < {MIN_CORRIDOR_LENGTH_M:.2f} m"
        )
    if t <= 0.0:
        issues.append("wall thickness must be positive")
    if float(spec.wall_height) < SENSOR_HIGH_HEIGHT_M:
        issues.append(
            f"wall height {float(spec.wall_height):.2f} m does not clear the camera "
            f"height {SENSOR_HIGH_HEIGHT_M:.2f} m, so rooms would not occlude"
        )
    if float(spec.wall_height) < SENSOR_LOW_HEIGHT_M:
        issues.append(
            f"wall height {float(spec.wall_height):.2f} m does not clear the LiDAR "
            f"height {SENSOR_LOW_HEIGHT_M:.2f} m"
        )

    if spec.entrance_side not in WALL_SIDES:
        issues.append(f"unknown entrance_side {spec.entrance_side!r}; expected one of {WALL_SIDES}")
    if float(spec.entrance_width) < required_door:
        issues.append(
            f"entrance {float(spec.entrance_width):.2f} m is narrower than the required "
            f"{required_door:.2f} m (robot {ROBOT_FOOTPRINT_DIAMETER_M:.2f} m + "
            f"{DOOR_CLEARANCE_MARGIN_M:.2f} m margin)"
        )

    attachments = _corridor_attachment_spans(spec)
    if spec.entrance_side in attachments:
        issues.append(
            f"entrance side {spec.entrance_side!r} also carries a wing; the corridor wall "
            "there is removed, so the entrance would not exist"
        )

    # Entrance gap must sit inside the corridor span with a corner stub.
    if spec.entrance_side in WALL_SIDES:
        _, _, span_lo, span_hi = _corridor_centrelines(spec)[spec.entrance_side]
        gap_lo, gap_hi = spec.entrance_gap()
        if gap_lo < span_lo + DOOR_EDGE_MARGIN_M or gap_hi > span_hi - DOOR_EDGE_MARGIN_M:
            issues.append(
                f"entrance span [{gap_lo:.2f}, {gap_hi:.2f}] must stay "
                f"{DOOR_EDGE_MARGIN_M:.2f} m inside [{span_lo:.2f}, {span_hi:.2f}]"
            )

    if not spec.rooms:
        issues.append("a corridor scene needs at least one wing")

    seen_names: set[str] = set()
    for wing in spec.rooms:
        prefix = f"wing {wing.name!r}"
        if wing.name in seen_names:
            issues.append(f"duplicate wing name {wing.name!r}")
        seen_names.add(wing.name)

        if wing.width <= 0.0 or wing.height <= 0.0:
            issues.append(f"{prefix}: interior extent must be positive")
            continue
        if min(wing.width, wing.height) < MIN_ROOM_SIDE_M - 1.0e-9:
            issues.append(
                f"{prefix}: interior {wing.width:.2f} x {wing.height:.2f} m is smaller than "
                f"{MIN_ROOM_SIDE_M:.2f} m on its short side"
            )

        side = touching_side(wing, spec)
        if side is None:
            issues.append(
                f"{prefix}: not attached to the corridor; its shared wall must sit exactly one "
                f"wall thickness ({t:.2f} m) off the corridor interior"
            )
            continue
        shared = OPPOSITE_SIDE[side]
        if float(wing.door_width) < required_door:
            issues.append(
                f"{prefix}: doorway {float(wing.door_width):.2f} m is narrower than the required "
                f"{required_door:.2f} m"
            )

        # The doorway must sit inside both the wing's own wall and the corridor.
        if shared in ("north", "south"):
            wing_lo, wing_hi = wing.interior_x
            corridor_lo, corridor_hi = spec.corridor_x
        else:
            wing_lo, wing_hi = wing.interior_y
            corridor_lo, corridor_hi = spec.corridor_y
        w_gap_lo, w_gap_hi = wing.door_gap()
        if w_gap_lo < wing_lo + DOOR_EDGE_MARGIN_M or w_gap_hi > wing_hi - DOOR_EDGE_MARGIN_M:
            issues.append(
                f"{prefix}: doorway span [{w_gap_lo:.2f}, {w_gap_hi:.2f}] must stay "
                f"{DOOR_EDGE_MARGIN_M:.2f} m inside its own wall [{wing_lo:.2f}, {wing_hi:.2f}]"
            )
        if w_gap_lo < corridor_lo + 1.0e-9 or w_gap_hi > corridor_hi - 1.0e-9:
            issues.append(
                f"{prefix}: doorway span [{w_gap_lo:.2f}, {w_gap_hi:.2f}] is not inside the "
                f"corridor span [{corridor_lo:.2f}, {corridor_hi:.2f}]; the door would open "
                "past the end of the corridor"
            )

        if not wing.object_name:
            issues.append(f"{prefix}: no task object assigned")

    # Wings must not overlap each other or the corridor interior.  Touching
    # along a shared wall line is legitimate: the wing flush with the corridor's
    # far end shares that wall face, and its own wall box covers it.
    corridor_box = (x0, x1, y0, y1)
    boxes = [(w.name, w.interior_x[0], w.interior_x[1], w.interior_y[0], w.interior_y[1]) for w in spec.rooms]
    for name, bx0, bx1, by0, by1 in boxes:
        if bx0 < corridor_box[1] and bx1 > corridor_box[0] and by0 < corridor_box[3] and by1 > corridor_box[2]:
            issues.append(f"wing {name!r} overlaps the corridor interior")
    gaps: list[float] = []
    for index, (name_a, ax0, ax1, ay0, ay1) in enumerate(boxes):
        for name_b, bx0, bx1, by0, by1 in boxes[index + 1:]:
            overlap_x = min(ax1, bx1) - max(ax0, bx0)
            overlap_y = min(ay1, by1) - max(ay0, by0)
            if overlap_x > 1.0e-9 and overlap_y > 1.0e-9:
                issues.append(f"wings {name_a!r} and {name_b!r} overlap")
                continue
            gap_x = max(ax0 - bx1, bx0 - ax1)
            gap_y = max(ay0 - by1, by0 - ay1)
            if gap_x > 0.0 and gap_y > 0.0:
                gaps.append(float(np.hypot(gap_x, gap_y)))
            elif gap_x > 0.0:
                gaps.append(float(gap_x))
            else:
                gaps.append(float(gap_y))
    metrics["min_room_gap_m"] = float(min(gaps)) if gaps else float("nan")

    objects = [wing.object_name for wing in spec.rooms if wing.object_name]
    if len(set(objects)) != len(objects):
        issues.append(f"two rooms claim the same task object: {objects}")

    walls = all_walls(spec, strict=False)
    metrics["wall_count"] = len(walls)
    if len(walls) < 8:
        issues.append(f"only {len(walls)} wall boxes built; the building is not closed")

    # Doorways must be real gaps: each flanked by exactly two stub walls that
    # do not overlap the opening.
    for wing in spec.rooms:
        side = touching_side(wing, spec)
        if side is None:
            continue
        gap_lo, gap_hi = wing.door_gap()
        flanks = [
            w for w in all_walls(spec, strict=False)
            if w.name.startswith(f"wing_{wing.name}_door_")
        ]
        metrics[f"{wing.name}_door_flanks"] = len(flanks)
        if len(flanks) != 2:
            issues.append(
                f"wing {wing.name!r}: expected 2 stub walls flanking the doorway, built {len(flanks)}"
            )
        for wall in flanks:
            if side in ("north", "south"):
                wall_lo = float(wall.position[0]) - float(wall.size[0])
                wall_hi = float(wall.position[0]) + float(wall.size[0])
            else:
                wall_lo = float(wall.position[1]) - float(wall.size[1])
                wall_hi = float(wall.position[1]) + float(wall.size[1])
            if wall_hi > gap_lo + 1.0e-9 and wall_lo < gap_hi - 1.0e-9:
                issues.append(f"{wall.name} overlaps the doorway span [{gap_lo:.2f}, {gap_hi:.2f}]")

    entrance_flanks = [
        w for w in corridor_walls(spec) if w.name.startswith(f"corridor_wall_{spec.entrance_side}_")
    ]
    metrics["entrance_flanks"] = len(entrance_flanks)
    if len(entrance_flanks) != 2:
        issues.append(
            f"expected 2 stub walls flanking the entrance, built {len(entrance_flanks)}"
        )

    # Structural invariant: each corridor wall line is covered end to end except
    # at the openings we intend -- a wing's doorway where it attaches, plus the
    # entrance.  Any other hole means a wall is missing or misaligned.
    for side, entry in boundary_intervals(spec).items():
        expected: list[tuple[float, float]] = []
        if side == spec.entrance_side:
            expected.append(spec.entrance_gap())
        for wing in spec.rooms:
            if touching_side(wing, spec) == side:
                expected.append(wing.door_gap())
        expected = _merge_intervals(expected)
        actual = _merge_intervals(entry["gaps"])
        metrics[f"{side}_openings"] = [[round(a, 4), round(b, 4)] for a, b in actual]
        mismatched = len(actual) != len(expected) or any(
            abs(a[0] - b[0]) > 1.0e-6 or abs(a[1] - b[1]) > 1.0e-6
            for a, b in zip(actual, expected)
        )
        if mismatched:
            issues.append(
                f"corridor {side} wall is not sealed: openings {actual} but the intended "
                f"doorways/entrance on that side are {expected}"
            )

    return {"ok": not issues, "issues": issues, "metrics": metrics, "walls": walls}


def in_room(wing: WingSpec, point, margin: float = 0.0) -> bool:
    """Whether ``point`` lies inside the wing with the given margin."""
    x, y = float(point[0]), float(point[1])
    return (
        wing.interior_x[0] + margin <= x <= wing.interior_x[1] - margin
        and wing.interior_y[0] + margin <= y <= wing.interior_y[1] - margin
    )


def object_room_report(episode: CorridorEpisode) -> dict:
    """Check every object sits inside the room it was assigned to."""
    spec = episode.spec
    issues: list[str] = []
    placement: dict[str, str | None] = {}
    for obj in episode.objects:
        owners = [
            wing.name
            for wing in spec.rooms
            if wing.object_name == obj.name and in_room(wing, obj.position, ROOM_OBJECT_MARGIN_M)
        ]
        placement[obj.name] = owners[0] if owners else None
        if not owners:
            issues.append(
                f"{obj.name!r} is not inside its assigned room with a {ROOM_OBJECT_MARGIN_M:.2f} m margin"
            )
    for wing in spec.rooms:
        if wing.object_name and wing.object_name not in {obj.name for obj in episode.objects}:
            issues.append(f"wing {wing.name!r} expects object {wing.object_name!r}, which is missing")
    rooms_used = [name for name in placement.values() if name]
    if len(set(rooms_used)) != len(rooms_used):
        issues.append(f"two objects share a room: {placement}")
    return {"ok": not issues, "issues": issues, "placement": placement}


def occlusion_report(episode: CorridorEpisode) -> dict:
    """Report which objects are hidden from the start pose.

    This is what makes S3 a search task: the robot cannot pick a target from
    the start pose, it has to choose a doorway.
    """
    walls = all_walls(episode.spec, strict=False)
    per_object = {
        obj.name: line_of_sight_blocked(episode.start_xy, obj.position, walls)
        for obj in episode.objects
    }
    return {
        "all_blocked": bool(per_object) and all(per_object.values()),
        "per_object": per_object,
    }


# ---------------------------------------------------------------------------
# Grid reachability / seal test
# ---------------------------------------------------------------------------


def _grid_setup(
    walls: list[ObstacleSpec], points: list[tuple[float, float]], resolution: float, pad: float
) -> tuple[np.ndarray, float, float, float, float]:
    """Rasterise dilated walls into a boolean blockage grid."""
    xs = [float(w.position[0]) for w in walls]
    ys = [float(w.position[1]) for w in walls]
    x_lo = min(min(xs), min(p[0] for p in points)) - pad
    x_hi = max(max(xs), max(p[0] for p in points)) + pad
    y_lo = min(min(ys), min(p[1] for p in points)) - pad
    y_hi = max(max(ys), max(p[1] for p in points)) + pad

    nx = int(np.ceil((x_hi - x_lo) / resolution)) + 1
    ny = int(np.ceil((y_hi - y_lo) / resolution)) + 1
    blocked = np.zeros((ny, nx), dtype=bool)

    for wall in walls:
        centre_x, centre_y = float(wall.position[0]), float(wall.position[1])
        half_x, half_y = float(wall.size[0]), float(wall.size[1])
        i0 = int(np.floor((centre_x - half_x - ROBOT_FOOTPRINT_RADIUS_M - x_lo) / resolution))
        i1 = int(np.ceil((centre_x + half_x + ROBOT_FOOTPRINT_RADIUS_M - x_lo) / resolution))
        j0 = int(np.floor((centre_y - half_y - ROBOT_FOOTPRINT_RADIUS_M - y_lo) / resolution))
        j1 = int(np.ceil((centre_y + half_y + ROBOT_FOOTPRINT_RADIUS_M - y_lo) / resolution))
        i0, i1 = max(0, i0), min(nx - 1, i1)
        j0, j1 = max(0, j0), min(ny - 1, j1)
        if i1 < i0 or j1 < j0:
            continue
        blocked[j0 : j1 + 1, i0 : i1 + 1] = True
    return blocked, x_lo, y_lo, float(resolution), float(nx)


def _bfs(blocked: np.ndarray, start_cell: tuple[int, int]):
    """4-connected flood fill; returns (visited, parent) arrays."""
    ny, nx = blocked.shape
    start_i, start_j = start_cell
    visited = np.zeros_like(blocked)
    parent = np.full((ny, nx, 2), -1, dtype=np.int32)
    if not (0 <= start_i < nx and 0 <= start_j < ny) or blocked[start_j, start_i]:
        return visited, parent, False
    queue = deque([(start_i, start_j)])
    visited[start_j, start_i] = True
    while queue:
        i, j = queue.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < nx and 0 <= nj < ny and not visited[nj, ni] and not blocked[nj, ni]:
                visited[nj, ni] = True
                parent[nj, ni] = (i, j)
                queue.append((ni, nj))
    return visited, parent, True


def _trace_path(parent: np.ndarray, start_cell, target_cell) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    current = (int(target_cell[0]), int(target_cell[1]))
    start = (int(start_cell[0]), int(start_cell[1]))
    for _ in range(200000):
        path.append(current)
        if current == start:
            break
        pi, pj = parent[current[1], current[0]]
        if pi < 0:
            return []
        current = (int(pi), int(pj))
    path.reverse()
    return path


def reachability_report(
    episode: CorridorEpisode, resolution: float = 0.05, keep_paths: bool = True
) -> dict:
    """Prove both halves of the doorway contract with a grid flood fill.

    Two runs over the same grid:

    * **open**   -- with the doorways as built. Every room must be reachable
      from the start pose, otherwise the scene is unsolvable.
    * **sealed** -- with every doorway (and the entrance) plugged. No room may
      be reachable, which is what proves the walls actually seal: if a corner
      leaked, the flood fill would walk into the room around the wall.

    The robot is modelled as a disc of ``ROBOT_FOOTPRINT_RADIUS_M``, so the
    passage test is the real one: a doorway too narrow for the robot fails here
    rather than in the collector.
    """
    spec = episode.spec
    walls = all_walls(spec, strict=False)
    points = [episode.start_xy] + [wing.interior_center for wing in spec.rooms]
    probes = {wing.name: wing.interior_center for wing in spec.rooms}

    def run(extra_walls: list[ObstacleSpec]) -> dict:
        blocked, x_lo, y_lo, res, nx = _grid_setup(walls + extra_walls, points, resolution, 0.60)
        start_cell = (
            int((episode.start_xy[0] - x_lo) / res),
            int((episode.start_xy[1] - y_lo) / res),
        )
        visited, parent, start_free = _bfs(blocked, start_cell)
        result: dict = {
            "start_cell_free": bool(start_free),
            "grid_shape": [int(blocked.shape[1]), int(blocked.shape[0])],
            "resolution_m": res,
            "rooms": {},
        }
        for name, probe in probes.items():
            cell = (int((probe[0] - x_lo) / res), int((probe[1] - y_lo) / res))
            reachable = bool(visited[cell[1], cell[0]])
            entry = {"reachable": reachable, "probe_xy": [float(probe[0]), float(probe[1])]}
            if reachable and keep_paths:
                path = _trace_path(parent, start_cell, cell)
                entry["path_length_cells"] = len(path)
                entry["path_xy"] = [
                    [x_lo + (i + 0.5) * res, y_lo + (j + 0.5) * res] for i, j in path
                ]
            result["rooms"][name] = entry
        result["all_rooms_reachable"] = all(v["reachable"] for v in result["rooms"].values())
        result["no_room_reachable"] = not any(v["reachable"] for v in result["rooms"].values())
        return result

    open_result = run([])
    sealed_result = run(door_plug_walls(spec))
    return {
        "resolution_m": float(resolution),
        "open": open_result,
        "sealed": sealed_result,
        "ok": bool(
            open_result["start_cell_free"]
            and open_result["all_rooms_reachable"]
            and sealed_result["no_room_reachable"]
        ),
    }


def scene_report(episode: CorridorEpisode, resolution: float = 0.05) -> dict:
    """Full contract report for one S3 episode."""
    validation = validate_multiroom(episode.spec)
    occlusion = occlusion_report(episode)
    placement = object_room_report(episode)
    reachability = reachability_report(episode, resolution=resolution)
    return {
        "ok": bool(
            validation["ok"]
            and occlusion["all_blocked"]
            and placement["ok"]
            and reachability["ok"]
        ),
        "validation": validation,
        "occlusion": occlusion,
        "placement": placement,
        "reachability": reachability,
    }


# ---------------------------------------------------------------------------
# Canonical episode + sampler
# ---------------------------------------------------------------------------


def default_objects() -> tuple[ObjectSpec, ...]:
    """The canonical S3 task objects, one per room."""
    table = {name: (label, kind, rgba) for name, label, kind, rgba in OBJECTS}
    objects: list[ObjectSpec] = []
    for wing in default_wings():
        label, kind, rgba = table[wing.object_name]
        position = S3_OBJECT_POSITIONS[wing.object_name]
        objects.append(
            ObjectSpec(
                name=wing.object_name,
                label=label,
                kind=kind,
                position=(float(position[0]), float(position[1])),
                rgba=rgba,
                size=OBJECT_HALF_SIZES[wing.object_name],
            )
        )
    return tuple(objects)


def default_episode() -> CorridorEpisode:
    """The canonical, hand-checked S3 episode used for review and tests."""
    return CorridorEpisode(
        spec=CorridorSpec(),
        objects=default_objects(),
        start_xy=S3_START_XY,
        start_yaw=S3_START_YAW,
    )


def _sample_objects(
    spec: CorridorSpec, rng: np.random.Generator
) -> tuple[ObjectSpec, ...] | None:
    table = {name: (label, kind, rgba) for name, label, kind, rgba in OBJECTS}
    objects: list[ObjectSpec] = []
    for wing in spec.rooms:
        x_lo, x_hi = wing.interior_x[0] + ROOM_OBJECT_MARGIN_M, wing.interior_x[1] - ROOM_OBJECT_MARGIN_M
        y_lo, y_hi = wing.interior_y[0] + ROOM_OBJECT_MARGIN_M, wing.interior_y[1] - ROOM_OBJECT_MARGIN_M
        if x_hi <= x_lo or y_hi <= y_lo:
            return None
        point = (float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi)))
        label, kind, rgba = table[wing.object_name]
        objects.append(
            ObjectSpec(
                name=wing.object_name,
                label=label,
                kind=kind,
                position=point,
                rgba=rgba,
                size=OBJECT_HALF_SIZES[wing.object_name],
            )
        )
    for index, obj in enumerate(objects):
        for other in objects[index + 1 :]:
            if float(np.hypot(obj.position[0] - other.position[0], obj.position[1] - other.position[1])) < MIN_OBJECT_SEPARATION_M:
                return None
    return tuple(objects)


ROOM_OBJECT_PERMUTATIONS = tuple(permutations(("green_cylinder", "yellow_box", "red_cube")))


def sample_episode(
    rng: np.random.Generator, attempts: int = 48, *,
    room_object_order: tuple[str, str, str] | None = None,
) -> CorridorEpisode | None:
    """Sample a jittered S3 episode that still satisfies every hard constraint.

    Rejection sampling keeps the family honest: nothing is silently relaxed.
    ``None`` means the sampler refused rather than emit an invalid building.
    """
    order = tuple(room_object_order) if room_object_order is not None else ROOM_OBJECT_PERMUTATIONS[0]
    if order not in ROOM_OBJECT_PERMUTATIONS:
        raise ValueError("room_object_order must place each of the three task objects in one room")
    t = WALL_THICKNESS_M
    for _ in range(max(1, attempts)):
        corridor_length = float(rng.uniform(5.00, 6.40))
        corridor_width = float(rng.uniform(CORRIDOR_WIDTH_M, 2.60))
        x0 = float(rng.uniform(0.80, 1.40))
        y0 = -0.5 * corridor_width
        corridor_x = (x0, x0 + corridor_length)
        corridor_y = (y0, y0 + corridor_width)

        door_north = float(rng.uniform(1.10, 1.45))
        door_south = float(rng.uniform(1.10, 1.45))
        door_end = float(rng.uniform(1.10, 1.45))
        door_entrance = float(rng.uniform(1.10, 1.45))

        # North wing: hangs off the corridor's north wall.
        north_w = float(rng.uniform(2.30, 2.90))
        north_h = float(rng.uniform(2.30, 2.90))
        north_x0 = corridor_x[0] + float(rng.uniform(0.60, 1.10))
        lo = north_x0 + DOOR_EDGE_MARGIN_M + 0.5 * door_north
        hi = north_x0 + north_w - DOOR_EDGE_MARGIN_M - 0.5 * door_north
        if hi <= lo:
            continue
        north = WingSpec(
            name="north_room",
            interior_x=(north_x0, north_x0 + north_w),
            interior_y=(corridor_y[1] + t, corridor_y[1] + t + north_h),
            door_center=float(rng.uniform(lo, hi)),
            door_width=door_north,
            object_name="green_cylinder",
        )

        # South wing: hangs off the corridor's south wall, flush with the east end.
        south_w = float(rng.uniform(2.30, 2.90))
        south_h = float(rng.uniform(2.30, 2.90))
        south_x1 = corridor_x[1] - float(rng.uniform(0.00, 0.40))
        south_x0 = south_x1 - south_w
        lo = south_x0 + DOOR_EDGE_MARGIN_M + 0.5 * door_south
        hi = south_x0 + south_w - DOOR_EDGE_MARGIN_M - 0.5 * door_south
        if hi <= lo:
            continue
        south = WingSpec(
            name="south_room",
            interior_x=(south_x0, south_x1),
            interior_y=(corridor_y[0] - t - south_h, corridor_y[0] - t),
            door_center=float(rng.uniform(lo, hi)),
            door_width=door_south,
            object_name="yellow_box",
        )

        # End room: hangs off the corridor's east end.
        end_w = float(rng.uniform(2.30, 3.00))
        end_h = float(rng.uniform(2.60, 3.40))
        end_y0 = -0.5 * end_h
        lo = end_y0 + DOOR_EDGE_MARGIN_M + 0.5 * door_end
        hi = end_y0 + end_h - DOOR_EDGE_MARGIN_M - 0.5 * door_end
        if hi <= lo:
            continue
        end = WingSpec(
            name="end_room",
            interior_x=(corridor_x[1] + t, corridor_x[1] + t + end_w),
            interior_y=(end_y0, end_y0 + end_h),
            door_center=float(rng.uniform(lo, hi)),
            door_width=door_end,
            object_name="red_cube",
        )

        spec = CorridorSpec(
            name="s3_corridor_rooms_sampled",
            corridor_x=corridor_x,
            corridor_y=corridor_y,
            entrance_side="west",
            entrance_center=0.0,
            entrance_width=door_entrance,
            rooms=tuple(dataclasses.replace(wing, object_name=name)
                        for wing, name in zip((north, south, end), order)),
        )
        if not validate_multiroom(spec)["ok"]:
            continue

        objects = _sample_objects(spec, rng)
        if objects is None:
            continue
        entrance = spec.entrance_gap()
        start_xy = (
            corridor_x[0] - float(rng.uniform(1.40, 2.00)),
            float(rng.uniform(entrance[0] + 0.15, entrance[1] - 0.15)),
        )
        episode = CorridorEpisode(
            spec=spec,
            objects=objects,
            start_xy=start_xy,
            start_yaw=float(rng.uniform(-0.25, 0.25)),
        )
        if not occlusion_report(episode)["all_blocked"]:
            continue
        if not object_room_report(episode)["ok"]:
            continue
        # Reachability is the expensive check, so it runs last: only episodes
        # that already satisfy every geometric rule pay for the flood fill.
        if not reachability_report(episode, resolution=0.06, keep_paths=False)["ok"]:
            continue
        return episode
    return None


__all__ = [
    "CORRIDOR_WIDTH_M",
    "MIN_ROOM_SIDE_M",
    "MIN_CORRIDOR_WIDTH_M",
    "OPPOSITE_SIDE",
    "ROOM_OBJECT_MARGIN_M",
    "ROOM_OBJECT_PERMUTATIONS",
    "CorridorEpisode",
    "CorridorSpec",
    "S3_OBJECT_POSITIONS",
    "S3_START_XY",
    "S3_START_YAW",
    "WingSpec",
    "all_walls",
    "boundary_intervals",
    "corridor_walls",
    "default_episode",
    "default_objects",
    "default_wings",
    "door_plug_walls",
    "in_room",
    "object_room_report",
    "occlusion_report",
    "reachability_report",
    "sample_episode",
    "scene_report",
    "shared_door_wall",
    "touching_side",
    "validate_multiroom",
    "wing_walls",
]
