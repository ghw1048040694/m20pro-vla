"""S2 room scene family: one walled room with a single doorway.

Why this exists
---------------
The S1 scene puts every task object in a narrow band around ``x ~ 3.2..3.95``
on an open plane, so "search" degenerates into "drive forward and pick a side".
A hidden evaluation drawn from the same template family only changes seeds, so
it cannot separate memorisation from generalisation.

S2 adds the smallest structure that actually changes the task: four walls, one
doorway, and task objects *inside* the room so they are occluded from the start
pose.  The robot has to locate the doorway before it can see any target.

Everything here lowers to plain :class:`~m20pro_vla.sim.mujoco.ObstacleSpec`
boxes, so the scene is materialised by the existing
:func:`~m20pro_vla.sim.mujoco.build_scene` pipeline: no new asset, no new
dynamics, and no change to ``mujoco.py``.

Scene tiers
-----------
S1  open plane, 3 objects, 1 screen wall      (current; regression baseline)
S2  single room, four walls, one doorway      (this module; training)
S3  multi-room / corridor, target out of view (planned; held-out OOD)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mujoco import OBJECTS, ObjectSpec, ObstacleSpec, obstacle_blocks_segment

# Same footprint radius the collector already uses for layout feasibility.
ROBOT_FOOTPRINT_RADIUS_M = 0.375
ROBOT_FOOTPRINT_DIAMETER_M = 2.0 * ROBOT_FOOTPRINT_RADIUS_M

WALL_THICKNESS_M = 0.12
WALL_HEIGHT_M = 1.00
WALL_RGBA = (0.74, 0.74, 0.76, 1.0)

DOOR_CLEAR_WIDTH_M = 1.20
DOOR_CLEARANCE_MARGIN_M = 0.25
DOOR_EDGE_MARGIN_M = 0.40
MIN_INTERIOR_SIDE_M = 2.00
OBJECT_EDGE_MARGIN_M = 0.45
MIN_OBJECT_SEPARATION_M = 0.55

# The planar LiDAR sits 0.20 m below the base and the front camera 0.15 m above
# it, so a wall has to clear both heights to occlude anything at all.
SENSOR_LOW_HEIGHT_M = 0.40
SENSOR_HIGH_HEIGHT_M = 0.75

WALL_SIDES = ("west", "east", "south", "north")

OBJECT_HALF_SIZES: dict[str, tuple[float, ...]] = {
    "red_cube": (0.16, 0.16, 0.16),
    "green_cylinder": (0.14, 0.20),
    "yellow_box": (0.16, 0.16, 0.16),
}

S2_OBJECT_POSITIONS: dict[str, tuple[float, float]] = {
    "red_cube": (4.00, 1.00),
    "green_cylinder": (4.80, 1.40),
    "yellow_box": (3.20, 1.60),
}

S2_START_XY = (0.00, 0.30)
S2_START_YAW = 0.0


@dataclass(frozen=True)
class RoomSpec:
    """Axis-aligned rectangular room with one doorway."""

    name: str = "s2_single_room"
    interior_x: tuple[float, float] = (2.0, 6.0)
    interior_y: tuple[float, float] = (-2.0, 2.0)
    door_wall: str = "west"
    door_center: float = -0.90
    door_width: float = DOOR_CLEAR_WIDTH_M
    wall_thickness: float = WALL_THICKNESS_M
    wall_height: float = WALL_HEIGHT_M
    rgba: tuple[float, float, float, float] = WALL_RGBA

    @property
    def door_gap(self) -> tuple[float, float]:
        half = 0.5 * float(self.door_width)
        return (float(self.door_center) - half, float(self.door_center) + half)

    @property
    def interior_center(self) -> tuple[float, float]:
        return (
            0.5 * (self.interior_x[0] + self.interior_x[1]),
            0.5 * (self.interior_y[0] + self.interior_y[1]),
        )

    def door_world_segment(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Endpoints of the door opening in world XY, on the wall centreline."""
        x0, x1 = self.interior_x
        y0, y1 = self.interior_y
        t = float(self.wall_thickness)
        lo, hi = self.door_gap
        if self.door_wall == "west":
            line = x0 - 0.5 * t
            return ((line, lo), (line, hi))
        if self.door_wall == "east":
            line = x1 + 0.5 * t
            return ((line, lo), (line, hi))
        if self.door_wall == "south":
            line = y0 - 0.5 * t
            return ((lo, line), (hi, line))
        line = y1 + 0.5 * t
        return ((lo, line), (hi, line))


@dataclass(frozen=True)
class RoomEpisode:
    """One S2 scene instance: room geometry + task objects + start pose."""

    spec: RoomSpec = field(default_factory=RoomSpec)
    objects: tuple[ObjectSpec, ...] = ()
    start_xy: tuple[float, float] = S2_START_XY
    start_yaw: float = S2_START_YAW


def _wall_centrelines(spec: RoomSpec) -> dict[str, tuple[str, float, float, float]]:
    """Map each side to (axis, centreline, span_lo, span_hi).

    The span is extended by one wall thickness past both interior corners so
    the four walls close the corners instead of leaving a diagonal leak.
    """
    x0, x1 = spec.interior_x
    y0, y1 = spec.interior_y
    t = float(spec.wall_thickness)
    half = 0.5 * t
    return {
        "west": ("y", x0 - half, y0 - t, y1 + t),
        "east": ("y", x1 + half, y0 - t, y1 + t),
        "south": ("x", y0 - half, x0 - t, x1 + t),
        "north": ("x", y1 + half, x0 - t, x1 + t),
    }


def room_obstacles(spec: RoomSpec) -> list[ObstacleSpec]:
    """Build the room walls as plain box obstacles for ``build_scene``.

    ``ObstacleSpec.size`` is MuJoCo half-extents, matching how
    :func:`obstacle_blocks_segment` reads it, so the geometric feasibility
    check and the rendered geometry cannot disagree.
    """
    half_t = 0.5 * float(spec.wall_thickness)
    half_h = 0.5 * float(spec.wall_height)
    gap_lo, gap_hi = spec.door_gap

    walls: list[ObstacleSpec] = []
    for side, (axis, line, lo, hi) in _wall_centrelines(spec).items():
        if side == spec.door_wall:
            segments = ((lo, gap_lo), (gap_hi, hi))
        else:
            segments = ((lo, hi),)
        for index, (segment_lo, segment_hi) in enumerate(segments):
            length = float(segment_hi - segment_lo)
            if length <= 1.0e-9:
                continue
            centre = 0.5 * (segment_lo + segment_hi)
            half_len = 0.5 * length
            if axis == "y":
                position = (line, centre, half_h)
                size = (half_t, half_len, half_h)
            else:
                position = (centre, line, half_h)
                size = (half_len, half_t, half_h)
            walls.append(
                ObstacleSpec(
                    name=f"room_wall_{side}_{index}",
                    kind="box",
                    position=(float(position[0]), float(position[1]), float(position[2])),
                    size=tuple(float(value) for value in size),
                    rgba=spec.rgba,
                )
            )
    return walls


def validate_room(spec: RoomSpec) -> dict:
    """Check the room is physically solvable before anything is collected.

    Every rule below is a hard constraint, not a preference: a room that fails
    one of them cannot yield a single valid demonstration episode.
    """
    issues: list[str] = []
    x0, x1 = spec.interior_x
    y0, y1 = spec.interior_y
    interior_w = float(x1 - x0)
    interior_h = float(y1 - y0)
    required_door = ROBOT_FOOTPRINT_DIAMETER_M + DOOR_CLEARANCE_MARGIN_M
    gap_lo, gap_hi = spec.door_gap

    metrics: dict[str, float | int | bool | str] = {
        "room": spec.name,
        "interior_width_m": interior_w,
        "interior_height_m": interior_h,
        "door_wall": spec.door_wall,
        "door_clear_width_m": float(spec.door_width),
        "door_gap_low_m": gap_lo,
        "door_gap_high_m": gap_hi,
        "robot_footprint_diameter_m": ROBOT_FOOTPRINT_DIAMETER_M,
        "door_required_width_m": required_door,
        "door_side_clearance_m": 0.5 * (float(spec.door_width) - ROBOT_FOOTPRINT_DIAMETER_M),
        "wall_height_m": float(spec.wall_height),
        "wall_thickness_m": float(spec.wall_thickness),
    }

    if spec.door_wall not in WALL_SIDES:
        issues.append(f"unknown door_wall {spec.door_wall!r}; expected one of {WALL_SIDES}")
    if interior_w <= 0.0 or interior_h <= 0.0:
        issues.append("interior extent must be positive")
    if min(interior_w, interior_h) < MIN_INTERIOR_SIDE_M:
        issues.append(
            f"interior too small to manoeuvre: {min(interior_w, interior_h):.2f} m "
            f"< {MIN_INTERIOR_SIDE_M:.2f} m"
        )
    if float(spec.door_width) < required_door:
        issues.append(
            f"doorway {float(spec.door_width):.2f} m is narrower than the required "
            f"{required_door:.2f} m (robot {ROBOT_FOOTPRINT_DIAMETER_M:.2f} m + "
            f"{DOOR_CLEARANCE_MARGIN_M:.2f} m margin)"
        )
    if float(spec.wall_thickness) <= 0.0:
        issues.append("wall thickness must be positive")

    if spec.door_wall in ("west", "east"):
        span_lo, span_hi = y0, y1
    else:
        span_lo, span_hi = x0, x1
    if spec.door_wall in WALL_SIDES:
        if gap_lo < span_lo + DOOR_EDGE_MARGIN_M or gap_hi > span_hi - DOOR_EDGE_MARGIN_M:
            issues.append(
                f"doorway span [{gap_lo:.2f}, {gap_hi:.2f}] must stay "
                f"{DOOR_EDGE_MARGIN_M:.2f} m inside the interior corners "
                f"[{span_lo:.2f}, {span_hi:.2f}]"
            )

    if float(spec.wall_height) < SENSOR_HIGH_HEIGHT_M:
        issues.append(
            f"wall height {float(spec.wall_height):.2f} m does not clear the camera "
            f"height {SENSOR_HIGH_HEIGHT_M:.2f} m, so the room would not occlude"
        )
    if float(spec.wall_height) < SENSOR_LOW_HEIGHT_M:
        issues.append(
            f"wall height {float(spec.wall_height):.2f} m does not clear the LiDAR "
            f"height {SENSOR_LOW_HEIGHT_M:.2f} m"
        )

    walls = room_obstacles(spec)
    metrics["wall_count"] = len(walls)
    expected_walls = len(WALL_SIDES) + 1
    if len(walls) != expected_walls:
        issues.append(
            f"expected {expected_walls} wall boxes (four sides, door side split in two), "
            f"built {len(walls)}"
        )

    return {"ok": not issues, "issues": issues, "metrics": metrics, "walls": walls}


def line_of_sight_blocked(
    start_xy: tuple[float, float] | np.ndarray,
    target_xy: tuple[float, float] | np.ndarray,
    walls: list[ObstacleSpec],
    padding: float = 0.0,
) -> bool:
    """Return whether any wall footprint crosses the start-to-target segment."""
    return any(
        obstacle_blocks_segment(start_xy, target_xy, wall, padding=padding) for wall in walls
    )


def occlusion_report(episode: RoomEpisode) -> dict:
    """Report which task objects are hidden from the start pose.

    This is what makes S2 a search task rather than a pointing task.  If a
    target is visible from the start, the episode is not measuring the
    doorway-localisation behaviour S2 exists to measure.
    """
    walls = room_obstacles(episode.spec)
    per_object = {
        obj.name: line_of_sight_blocked(episode.start_xy, obj.position, walls)
        for obj in episode.objects
    }
    return {
        "all_blocked": bool(per_object) and all(per_object.values()),
        "per_object": per_object,
    }


def default_objects() -> tuple[ObjectSpec, ...]:
    """The canonical S2 task objects, placed on the side away from the door."""
    table = {name: (label, kind, rgba) for name, label, kind, rgba in OBJECTS}
    objects: list[ObjectSpec] = []
    for name, position in S2_OBJECT_POSITIONS.items():
        label, kind, rgba = table[name]
        objects.append(
            ObjectSpec(
                name=name,
                label=label,
                kind=kind,
                position=(float(position[0]), float(position[1])),
                rgba=rgba,
                size=OBJECT_HALF_SIZES[name],
            )
        )
    return tuple(objects)


def default_episode() -> RoomEpisode:
    """The canonical, hand-checked S2 episode used for review and tests."""
    return RoomEpisode(
        spec=RoomSpec(),
        objects=default_objects(),
        start_xy=S2_START_XY,
        start_yaw=S2_START_YAW,
    )


def _object_region(spec: RoomSpec) -> tuple[tuple[float, float], tuple[float, float]]:
    """XY box the task objects may occupy: the half of the room away from the door."""
    x0, x1 = spec.interior_x
    y0, y1 = spec.interior_y
    gap_lo, gap_hi = spec.door_gap
    margin = OBJECT_EDGE_MARGIN_M
    if spec.door_wall in ("west", "east"):
        x_lo, x_hi = x0 + margin, x1 - margin
        if float(spec.door_center) < 0.5 * (y0 + y1):
            y_lo, y_hi = gap_hi + 0.35, y1 - margin
        else:
            y_lo, y_hi = y0 + margin, gap_lo - 0.35
    else:
        y_lo, y_hi = y0 + margin, y1 - margin
        if float(spec.door_center) < 0.5 * (x0 + x1):
            x_lo, x_hi = gap_hi + 0.35, x1 - margin
        else:
            x_lo, x_hi = x0 + margin, gap_lo - 0.35
    return (x_lo, x_hi), (y_lo, y_hi)


def _sample_objects(
    spec: RoomSpec,
    start_xy: tuple[float, float],
    rng: np.random.Generator,
    attempts: int = 24,
) -> tuple[ObjectSpec, ...] | None:
    """Sample three separated objects that are all occluded from the start."""
    (x_lo, x_hi), (y_lo, y_hi) = _object_region(spec)
    if x_hi <= x_lo or y_hi <= y_lo:
        return None
    names = list(S2_OBJECT_POSITIONS)
    table = {name: (label, kind, rgba) for name, label, kind, rgba in OBJECTS}
    walls = room_obstacles(spec)

    for _ in range(max(1, attempts)):
        points: list[tuple[float, float]] = []
        for _ in range(len(names)):
            for _ in range(32):
                candidate = (float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi)))
                if all(
                    float(np.hypot(candidate[0] - px, candidate[1] - py)) >= MIN_OBJECT_SEPARATION_M
                    for px, py in points
                ):
                    points.append(candidate)
                    break
            else:
                break
        if len(points) != len(names):
            continue
        objects = []
        for name, position in zip(names, points):
            label, kind, rgba = table[name]
            objects.append(
                ObjectSpec(
                    name=name,
                    label=label,
                    kind=kind,
                    position=position,
                    rgba=rgba,
                    size=OBJECT_HALF_SIZES[name],
                )
            )
        if all(
            line_of_sight_blocked(start_xy, obj.position, walls) for obj in objects
        ):
            return tuple(objects)
    return None


def sample_episode(rng: np.random.Generator, attempts: int = 64) -> RoomEpisode | None:
    """Sample a jittered S2 episode that still satisfies every hard constraint.

    Rejection sampling keeps the family honest: nothing is silently relaxed.
    ``None`` means the sampler refused rather than emit an invalid scene.
    """
    for _ in range(max(1, attempts)):
        width = float(rng.uniform(3.20, 4.80))
        height = float(rng.uniform(3.20, 4.80))
        x0 = float(rng.uniform(1.70, 2.20))
        y0 = -0.5 * height
        door_width = float(rng.uniform(1.05, 1.55))
        lo = y0 + DOOR_EDGE_MARGIN_M + 0.5 * door_width
        hi = y0 + height - DOOR_EDGE_MARGIN_M - 0.5 * door_width
        if hi <= lo:
            continue
        door_center = float(rng.uniform(lo, hi))
        spec = RoomSpec(
            name="s2_single_room_sampled",
            interior_x=(x0, x0 + width),
            interior_y=(y0, y0 + height),
            door_wall="west",
            door_center=door_center,
            door_width=door_width,
        )
        if not validate_room(spec)["ok"]:
            continue
        start_xy = (
            float(x0 - rng.uniform(1.60, 2.40)),
            float(door_center + rng.uniform(0.70, 1.70)),
        )
        y1 = y0 + height
        if not (y0 + 0.20 <= start_xy[1] <= y1 - 0.20):
            continue
        objects = _sample_objects(spec, start_xy, rng)
        if objects is None:
            continue
        episode = RoomEpisode(
            spec=spec,
            objects=objects,
            start_xy=start_xy,
            start_yaw=float(rng.uniform(-0.35, 0.35)),
        )
        if occlusion_report(episode)["all_blocked"]:
            return episode
    return None


__all__ = [
    "DOOR_CLEAR_WIDTH_M",
    "OBJECT_HALF_SIZES",
    "ROBOT_FOOTPRINT_DIAMETER_M",
    "ROBOT_FOOTPRINT_RADIUS_M",
    "RoomEpisode",
    "RoomSpec",
    "S2_OBJECT_POSITIONS",
    "S2_START_XY",
    "S2_START_YAW",
    "WALL_HEIGHT_M",
    "WALL_SIDES",
    "WALL_THICKNESS_M",
    "default_episode",
    "default_objects",
    "line_of_sight_blocked",
    "occlusion_report",
    "room_obstacles",
    "sample_episode",
    "validate_room",
]
