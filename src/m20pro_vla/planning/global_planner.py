"""Privileged grid global planner for the structured (S2/S3) curricula.

Why this exists
---------------
:class:`~m20pro_vla.planning.search_mpc.SearchMPCPlanner` is a *local*
predictive controller: it rolls primitive body commands out over an 8-step
horizon (8 x 0.02 s) against the MuJoCo dynamics and scores them. When the
straight line to the target is blocked it falls back to a hand-written
three-point detour around the single nearest blocking obstacle. That detour is
only correct for an isolated screen wall on an open plane (S1). As soon as the
scene has real structure -- a room corner, a doorway, a corridor -- the detour
points land outside the reachable free space and the teacher stalls against the
first wall.

This module supplies the missing capability: A* over an occupancy grid built
from the privileged layout.

Design contract
---------------
* **Privileged input only.** The planner consumes the same ``ObstacleSpec``
  table the simulator materialises. It never reads a policy observation, so
  wiring it into the offline teacher cannot leak anything into the learned
  policy.
* **Disc robot, square dilation.** Obstacles are dilated by
  ``footprint_radius + safety_margin`` in x and y, exactly the rasterisation
  ``corridor.reachability_report`` already uses for its doorway proof. "The
  planner says passable" and "the reachability proof says passable" therefore
  cannot disagree.
* **Hard margin vs soft margin.** ``footprint_radius`` is a hard constraint:
  the planner never routes through a passage narrower than the robot. The extra
  ``safety_margin`` is a preference and is relaxed along a configured ladder
  when the scene is tight (a 1.00 m doorway leaves only 0.25 m of slack), but
  it is never relaxed below zero.
* **Honest failure.** An unreachable goal returns ``reachable=False`` with a
  reason instead of a silently truncated path. Callers are expected to fall
  back, not to guess.

The output is a short waypoint list rather than a dense cell path: the teacher
steers towards the farthest waypoint it can still see, which keeps the
commanded yaw smooth.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np

from m20pro_vla.sim.mujoco import ObstacleSpec
from m20pro_vla.sim.rooms import ROBOT_FOOTPRINT_RADIUS_M

# (di, dj, step cost) for 8-connected movement.
_MOVES: tuple[tuple[int, int, float], ...] = (
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (-1, -1, math.sqrt(2.0)),
)


@dataclass(frozen=True)
class GlobalPlannerConfig:
    """Rasterisation and cost-shaping knobs for :class:`GlobalPlanner`."""

    resolution: float = 0.05
    footprint_radius: float = ROBOT_FOOTPRINT_RADIUS_M
    # Tried in order; the first margin that yields a path wins. The footprint
    # radius is *not* part of this ladder, so a margin of 0.0 still models the
    # real robot.
    safety_margins: tuple[float, ...] = (0.06, 0.03, 0.0)
    goal_tolerance: float = 0.35
    start_snap_radius: float = 0.30
    bounds_padding: float = 1.20
    clearance_weight: float = 0.80
    comfortable_clearance: float = 0.30
    max_expansions: int = 300_000
    simplify: bool = True


@dataclass(frozen=True)
class GlobalPlan:
    """One A* result, including enough provenance to audit a failure."""

    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    waypoints: tuple[tuple[float, float], ...]
    reachable: bool
    reason: str
    inflation_m: float
    cost_m: float
    path_cells: int
    expanded: int
    resolution_m: float
    grid_shape: tuple[int, int]
    start_snapped: bool
    goal_snapped: bool

    def as_dict(self) -> dict:
        return {
            "reachable": bool(self.reachable),
            "reason": self.reason,
            "inflation_m": round(float(self.inflation_m), 4),
            "cost_m": round(float(self.cost_m), 3),
            "path_cells": int(self.path_cells),
            "expanded": int(self.expanded),
            "waypoint_count": len(self.waypoints),
            "resolution_m": float(self.resolution_m),
            "grid_shape": [int(self.grid_shape[0]), int(self.grid_shape[1])],
            "start_snapped": bool(self.start_snapped),
            "goal_snapped": bool(self.goal_snapped),
        }


def _obstacle_half_extents_xy(obstacle: ObstacleSpec) -> tuple[float, float]:
    """XY half-extents of an obstacle footprint.

    ``ObstacleSpec.size`` is a MuJoCo half-extent vector. For a cylinder it is
    ``(radius, half_height)``, so the *y* half-extent must be the radius rather
    than ``size[1]``; reusing ``size[:2]`` verbatim would model a cylinder as a
    tall thin box and under-block it.
    """
    if str(obstacle.kind) == "cylinder":
        radius = float(obstacle.size[0])
        return radius, radius
    return float(obstacle.size[0]), float(obstacle.size[1])


class GlobalPlanner:
    """A* over a footprint-dilated occupancy grid of the privileged layout."""

    def __init__(
        self,
        obstacles: list[ObstacleSpec] | tuple[ObstacleSpec, ...] | None = None,
        config: GlobalPlannerConfig | None = None,
    ) -> None:
        self.config = config or GlobalPlannerConfig()
        if self.config.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if self.config.footprint_radius < 0.0:
            raise ValueError("footprint_radius must be non-negative")
        if not self.config.safety_margins:
            raise ValueError("safety_margins must not be empty")
        self.obstacles = list(obstacles or [])
        self._boxes: list[tuple[float, float, float, float]] = []
        for obstacle in self.obstacles:
            half_x, half_y = _obstacle_half_extents_xy(obstacle)
            if half_x <= 0.0 or half_y <= 0.0:
                continue
            self._boxes.append(
                (
                    float(obstacle.position[0]),
                    float(obstacle.position[1]),
                    half_x,
                    half_y,
                )
            )

    # ------------------------------------------------------------------ grid

    def _extents(
        self, points: list[tuple[float, float]]
    ) -> tuple[float, float, int, int]:
        res = float(self.config.resolution)
        if self._boxes:
            x_lo = min(box[0] - box[2] for box in self._boxes)
            x_hi = max(box[0] + box[2] for box in self._boxes)
            y_lo = min(box[1] - box[3] for box in self._boxes)
            y_hi = max(box[1] + box[3] for box in self._boxes)
        else:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_lo, x_hi = min(xs), max(xs)
            y_lo, y_hi = min(ys), max(ys)
        for px, py in points:
            x_lo = min(x_lo, px)
            x_hi = max(x_hi, px)
            y_lo = min(y_lo, py)
            y_hi = max(y_hi, py)
        pad = float(self.config.bounds_padding)
        x_lo -= pad
        x_hi += pad
        y_lo -= pad
        y_hi += pad
        nx = int(math.ceil((x_hi - x_lo) / res)) + 1
        ny = int(math.ceil((y_hi - y_lo) / res)) + 1
        return float(x_lo), float(y_lo), int(nx), int(ny)

    def _rasterize(
        self, x_lo: float, y_lo: float, nx: int, ny: int, inflate: float
    ) -> np.ndarray:
        """Mark every cell overlapping an obstacle dilated by ``inflate``."""
        res = float(self.config.resolution)
        blocked = np.zeros((ny, nx), dtype=bool)
        for centre_x, centre_y, half_x, half_y in self._boxes:
            i0 = int(math.floor((centre_x - half_x - inflate - x_lo) / res))
            i1 = int(math.ceil((centre_x + half_x + inflate - x_lo) / res))
            j0 = int(math.floor((centre_y - half_y - inflate - y_lo) / res))
            j1 = int(math.ceil((centre_y + half_y + inflate - y_lo) / res))
            i0, i1 = max(0, i0), min(nx - 1, i1)
            j0, j1 = max(0, j0), min(ny - 1, j1)
            if i1 < i0 or j1 < j0:
                continue
            blocked[j0 : j1 + 1, i0 : i1 + 1] = True
        return blocked

    def _clearance_field(
        self, x_lo: float, y_lo: float, nx: int, ny: int
    ) -> np.ndarray:
        """Distance from each cell centre to the nearest obstacle surface."""
        res = float(self.config.resolution)
        xs = x_lo + (np.arange(nx, dtype=np.float64) + 0.5) * res
        ys = y_lo + (np.arange(ny, dtype=np.float64) + 0.5) * res
        if not self._boxes:
            return np.full((ny, nx), 1.0e6, dtype=np.float64)
        clear = np.full((ny, nx), np.inf, dtype=np.float64)
        for centre_x, centre_y, half_x, half_y in self._boxes:
            dx = np.maximum(np.abs(xs[None, :] - centre_x) - half_x, 0.0)
            dy = np.maximum(np.abs(ys[:, None] - centre_y) - half_y, 0.0)
            np.minimum(clear, np.hypot(dx, dy), out=clear)
        return clear

    def _cell_of(self, point: tuple[float, float], x_lo: float, y_lo: float) -> tuple[int, int]:
        res = float(self.config.resolution)
        return (
            int(math.floor((float(point[0]) - x_lo) / res)),
            int(math.floor((float(point[1]) - y_lo) / res)),
        )

    def _world_of(self, i: int, j: int, x_lo: float, y_lo: float) -> tuple[float, float]:
        res = float(self.config.resolution)
        return (x_lo + (i + 0.5) * res, y_lo + (j + 0.5) * res)

    def _nearest_free_cell(
        self,
        blocked: np.ndarray,
        cell: tuple[int, int],
        radius_cells: int,
    ) -> tuple[int, int] | None:
        ny, nx = blocked.shape
        i0, j0 = cell
        best: tuple[int, int] | None = None
        best_distance = float("inf")
        for dj in range(-radius_cells, radius_cells + 1):
            for di in range(-radius_cells, radius_cells + 1):
                i, j = i0 + di, j0 + dj
                if not (0 <= i < nx and 0 <= j < ny):
                    continue
                if blocked[j, i]:
                    continue
                distance = float(math.hypot(di, dj))
                if distance < best_distance:
                    best_distance = distance
                    best = (i, j)
        return best

    def _segment_free(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        blocked: np.ndarray,
        x_lo: float,
        y_lo: float,
    ) -> bool:
        res = float(self.config.resolution)
        ny, nx = blocked.shape
        span = float(math.hypot(end[0] - start[0], end[1] - start[1]))
        steps = max(1, int(math.ceil(span / max(res * 0.5, 1.0e-6))))
        for k in range(steps + 1):
            t = k / steps
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            i = int(math.floor((x - x_lo) / res))
            j = int(math.floor((y - y_lo) / res))
            if not (0 <= i < nx and 0 <= j < ny):
                return False
            if blocked[j, i]:
                return False
        return True

    # ------------------------------------------------------------------- A*

    @staticmethod
    def _octile(di: int, dj: int) -> float:
        a, b = abs(di), abs(dj)
        lo, hi = (a, b) if a <= b else (b, a)
        return (hi - lo) + math.sqrt(2.0) * lo

    def _astar(
        self,
        blocked: np.ndarray,
        penalty: np.ndarray,
        start_cell: tuple[int, int],
        goal_cell: tuple[int, int],
    ) -> tuple[list[tuple[int, int]] | None, float, int, str]:
        ny, nx = blocked.shape
        si, sj = start_cell
        gi, gj = goal_cell
        cost = np.full((ny, nx), np.inf, dtype=np.float64)
        parent = np.full((ny, nx, 2), -1, dtype=np.int32)
        closed = np.zeros((ny, nx), dtype=bool)
        cost[sj, si] = 0.0
        heap: list[tuple[float, float, int, int]] = [
            (self._octile(gi - si, gj - sj), 0.0, si, sj)
        ]
        expanded = 0
        weight = float(self.config.clearance_weight)
        found = False
        while heap:
            _, current_cost, i, j = heapq.heappop(heap)
            if closed[j, i]:
                continue
            closed[j, i] = True
            expanded += 1
            if expanded > int(self.config.max_expansions):
                return None, float("inf"), expanded, "expansions_exhausted"
            if i == gi and j == gj:
                found = True
                break
            for di, dj, step in _MOVES:
                ni, nj = i + di, j + dj
                if not (0 <= ni < nx and 0 <= nj < ny):
                    continue
                if blocked[nj, ni] or closed[nj, ni]:
                    continue
                if di != 0 and dj != 0 and (blocked[j, ni] or blocked[nj, i]):
                    # No corner cutting: both orthogonal neighbours must be free.
                    continue
                move = step * (1.0 + weight * float(penalty[nj, ni]))
                candidate = current_cost + move
                if candidate < cost[nj, ni] - 1.0e-12:
                    cost[nj, ni] = candidate
                    parent[nj, ni] = (i, j)
                    heapq.heappush(
                        heap,
                        (candidate + self._octile(gi - ni, gj - nj), candidate, ni, nj),
                    )
        if not found:
            return None, float("inf"), expanded, "no_path"
        path: list[tuple[int, int]] = []
        i, j = gi, gj
        for _ in range(ny * nx + 1):
            path.append((i, j))
            if i == si and j == sj:
                break
            pi, pj = int(parent[j, i][0]), int(parent[j, i][1])
            if pi < 0 or pj < 0:
                return None, float("inf"), expanded, "broken_parent_chain"
            i, j = pi, pj
        path.reverse()
        return path, float(cost[gj, gi]), expanded, "ok"

    # ---------------------------------------------------------------- planning

    def plan(
        self, start_xy: np.ndarray | tuple[float, float], goal_xy: np.ndarray | tuple[float, float]
    ) -> GlobalPlan:
        cfg = self.config
        res = float(cfg.resolution)
        start = (float(start_xy[0]), float(start_xy[1]))
        goal = (float(goal_xy[0]), float(goal_xy[1]))
        x_lo, y_lo, nx, ny = self._extents([start, goal])
        clearance = self._clearance_field(x_lo, y_lo, nx, ny)
        comfortable = max(float(cfg.comfortable_clearance), 1.0e-9)
        penalty = np.clip((comfortable - clearance) / comfortable, 0.0, 1.0)

        goal_tol_cells = max(1, int(math.ceil(float(cfg.goal_tolerance) / res)))
        start_tol_cells = max(1, int(math.ceil(float(cfg.start_snap_radius) / res)))
        raw_start_cell = self._cell_of(start, x_lo, y_lo)
        raw_goal_cell = self._cell_of(goal, x_lo, y_lo)

        reasons: list[str] = []
        for margin in cfg.safety_margins:
            inflate = float(cfg.footprint_radius) + float(margin)
            blocked = self._rasterize(x_lo, y_lo, nx, ny, inflate)
            start_cell = self._nearest_free_cell(blocked, raw_start_cell, start_tol_cells)
            if start_cell is None:
                reasons.append(f"margin={margin:.3f}:start_in_obstacle")
                continue
            goal_cell = self._nearest_free_cell(blocked, raw_goal_cell, goal_tol_cells)
            if goal_cell is None:
                reasons.append(f"margin={margin:.3f}:goal_in_obstacle")
                continue
            path, cost, expanded, status = self._astar(blocked, penalty, start_cell, goal_cell)
            if path is None:
                reasons.append(f"margin={margin:.3f}:{status}")
                continue
            waypoints = self._waypoints(start, goal, path, blocked, x_lo, y_lo)
            return GlobalPlan(
                start_xy=start,
                goal_xy=goal,
                waypoints=tuple(waypoints),
                reachable=True,
                reason="ok",
                inflation_m=inflate,
                cost_m=cost * res,
                path_cells=len(path),
                expanded=expanded,
                resolution_m=res,
                grid_shape=(nx, ny),
                start_snapped=bool(start_cell != raw_start_cell),
                goal_snapped=bool(goal_cell != raw_goal_cell),
            )
        return GlobalPlan(
            start_xy=start,
            goal_xy=goal,
            waypoints=(start,),
            reachable=False,
            reason="; ".join(reasons) or "no_safety_margin_tried",
            inflation_m=float(cfg.footprint_radius),
            cost_m=float("inf"),
            path_cells=0,
            expanded=0,
            resolution_m=res,
            grid_shape=(nx, ny),
            start_snapped=False,
            goal_snapped=False,
        )

    def _waypoints(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        path: list[tuple[int, int]],
        blocked: np.ndarray,
        x_lo: float,
        y_lo: float,
    ) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = [start]
        for i, j in path[1:]:
            points.append(self._world_of(i, j, x_lo, y_lo))
        points.append(goal)
        if not self.config.simplify:
            return points
        pruned: list[tuple[float, float]] = [points[0]]
        index = 0
        last = len(points) - 1
        while index < last:
            target = last
            while target > index + 1 and not self._segment_free(
                points[index], points[target], blocked, x_lo, y_lo
            ):
                target -= 1
            pruned.append(points[target])
            if target == index:
                break
            index = target
        return pruned


__all__ = [
    "GlobalPlan",
    "GlobalPlanner",
    "GlobalPlannerConfig",
]
