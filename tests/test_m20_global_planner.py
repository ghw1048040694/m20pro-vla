"""Contract tests for the privileged grid global planner.

The planner exists to unblock the structured (S2/S3) curricula: the local
``SearchMPCPlanner`` detour cannot route around a room corner or through a
doorway. These tests guard the properties that make it usable and safe to wire
into the offline teacher:

1. it finds the direct route on an open plane (within one grid cell);
2. it routes *through* a gap instead of assuming the straight line is clear;
3. it reports an unreachable goal honestly instead of truncating a path;
4. the inflation ladder relaxes only the soft safety margin -- never the robot
   footprint -- and a passage narrower than the robot is refused outright.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from m20pro_vla.planning import (
    GlobalPlanner,
    GlobalPlannerConfig,
    SearchMPCConfig,
    SearchMPCPlanner,
)
from m20pro_vla.planning.global_planner import _obstacle_half_extents_xy
from m20pro_vla.sim.corridor import all_walls, default_episode as corridor_episode, door_plug_walls
from m20pro_vla.sim.mujoco import ObstacleSpec
from m20pro_vla.sim.rooms import (
    ROBOT_FOOTPRINT_RADIUS_M,
    default_episode as room_episode,
    room_obstacles,
    sample_episode as sample_room_episode,
)

WALL_HALF_THICKNESS = 0.06


def wall(
    name: str, centre_x: float, centre_y: float, half_x: float, half_y: float
) -> ObstacleSpec:
    return ObstacleSpec(
        name=name,
        kind="box",
        position=(centre_x, centre_y, 0.5),
        size=(half_x, half_y, 0.5),
    )


def wall_with_gap(centre_x: float, gap: float, span: float) -> list[ObstacleSpec]:
    """A wall on ``x = centre_x`` with a ``gap``-wide opening centred on y = 0."""
    segment_half = 0.5 * (span - 0.5 * gap)
    segment_centre = 0.5 * gap + segment_half
    return [
        wall("wall_lo", centre_x, -segment_centre, WALL_HALF_THICKNESS, segment_half),
        wall("wall_hi", centre_x, segment_centre, WALL_HALF_THICKNESS, segment_half),
    ]


def sealed_box(centre_x: float, centre_y: float, half: float) -> list[ObstacleSpec]:
    """Four walls forming a fully closed square around ``(centre_x, centre_y)``."""
    t = WALL_HALF_THICKNESS
    reach = half + 2.0 * t
    return [
        wall("box_w", centre_x - half - t, centre_y, t, reach),
        wall("box_e", centre_x + half + t, centre_y, t, reach),
        wall("box_s", centre_x, centre_y - half - t, reach, t),
        wall("box_n", centre_x, centre_y + half + t, reach, t),
    ]


def box_with_gap(
    centre_x: float, centre_y: float, half: float, gap: float
) -> list[ObstacleSpec]:
    """A sealed box whose *only* opening is a ``gap``-wide slit in the west wall.

    A finite free-standing wall can always be walked around its ends (the grid
    is padded past the obstacle bounds), so a doorway-reachability test has to
    enclose the goal or it would pass for the wrong reason.
    """
    t = WALL_HALF_THICKNESS
    reach = half + 2.0 * t
    segment_half = 0.5 * (reach - 0.5 * gap)
    segment_centre = 0.5 * gap + segment_half
    x_line = centre_x - half - t
    return [
        wall("box_w_lo", x_line, centre_y - segment_centre, t, segment_half),
        wall("box_w_hi", x_line, centre_y + segment_centre, t, segment_half),
        wall("box_e", centre_x + half + t, centre_y, t, reach),
        wall("box_s", centre_x, centre_y - half - t, reach, t),
        wall("box_n", centre_x, centre_y + half + t, reach, t),
    ]


def crossing_y(points, x_line: float) -> list[float]:
    """Y values where the polyline crosses the vertical line ``x = x_line``."""
    found: list[float] = []
    for start, end in zip(points, points[1:]):
        if (start[0] - x_line) * (end[0] - x_line) > 0.0:
            continue
        delta_x = end[0] - start[0]
        if abs(delta_x) < 1.0e-9:
            found.append(float(start[1]))
            continue
        t = (x_line - start[0]) / delta_x
        found.append(float(start[1] + t * (end[1] - start[1])))
    return found


class _Pose:
    """Minimal stand-in for ``mujoco.MjData``: only ``qpos[:7]`` is read."""

    def __init__(self, x: float, y: float, yaw: float = 0.0) -> None:
        self.qpos = np.zeros(7, dtype=np.float64)
        self.qpos[0] = x
        self.qpos[1] = y
        self.qpos[3] = math.cos(0.5 * yaw)
        self.qpos[6] = math.sin(0.5 * yaw)


class M20GlobalPlannerTests(unittest.TestCase):
    def test_footprint_default_matches_scene_constant(self) -> None:
        self.assertEqual(GlobalPlannerConfig().footprint_radius, ROBOT_FOOTPRINT_RADIUS_M)

    def test_cylinder_uses_radius_on_both_axes(self) -> None:
        # A cylinder's size is (radius, half_height); reading size[:2] verbatim
        # would model it as a tall thin box and under-block the y axis.
        cylinder = ObstacleSpec(
            name="post", kind="cylinder", position=(0.0, 0.0, 0.5), size=(0.20, 1.00)
        )
        self.assertEqual(_obstacle_half_extents_xy(cylinder), (0.20, 0.20))

    def test_open_plane_follows_the_direct_line(self) -> None:
        planner = GlobalPlanner([], GlobalPlannerConfig())
        plan = planner.plan((0.0, 0.0), (3.0, 0.0))
        self.assertTrue(plan.reachable, plan.reason)
        # Rasterisation costs at most one cell of slack per endpoint.
        self.assertAlmostEqual(plan.cost_m, 3.0, delta=0.10)
        self.assertEqual(len(plan.waypoints), 2)
        self.assertAlmostEqual(plan.waypoints[-1][0], 3.0, places=3)

    def test_planning_is_deterministic(self) -> None:
        obstacles = wall_with_gap(1.5, 1.20, 2.0)
        planner = GlobalPlanner(obstacles, GlobalPlannerConfig())
        first = planner.plan((0.0, 0.0), (3.0, 0.0))
        second = planner.plan((0.0, 0.0), (3.0, 0.0))
        self.assertEqual(first.waypoints, second.waypoints)
        self.assertEqual(first.cost_m, second.cost_m)

    def test_comfortable_gap_uses_the_preferred_margin(self) -> None:
        planner = GlobalPlanner(wall_with_gap(1.5, 1.20, 2.0), GlobalPlannerConfig())
        plan = planner.plan((0.0, 0.0), (3.0, 0.0))
        self.assertTrue(plan.reachable, plan.reason)
        preferred = GlobalPlannerConfig().footprint_radius + GlobalPlannerConfig().safety_margins[0]
        self.assertAlmostEqual(plan.inflation_m, preferred, places=6)
        for y in crossing_y(plan.waypoints, 1.5):
            self.assertLess(abs(y), 0.45, "the route must pass through the opening")
        self.assertGreater(plan.cost_m, 3.0, "a detour must cost more than the blocked line")

    def test_tight_gap_still_passable_but_never_below_footprint(self) -> None:
        # A 0.94 m opening leaves only 0.19 m of slack for a 0.75 m robot, so
        # the soft safety margin has to relax -- but not past the footprint.
        planner = GlobalPlanner(box_with_gap(3.0, 0.0, 0.90, 0.94), GlobalPlannerConfig())
        plan = planner.plan((0.0, 0.0), (3.0, 0.0))
        self.assertTrue(plan.reachable, plan.reason)
        self.assertGreaterEqual(plan.inflation_m, ROBOT_FOOTPRINT_RADIUS_M - 1.0e-9)

    def test_gap_narrower_than_the_robot_is_refused(self) -> None:
        # 0.70 m opening < 0.75 m robot: no safety margin can make this legal.
        planner = GlobalPlanner(box_with_gap(3.0, 0.0, 0.90, 0.70), GlobalPlannerConfig())
        plan = planner.plan((0.0, 0.0), (3.0, 0.0))
        self.assertFalse(plan.reachable, "a sub-footprint opening must not be planned")
        self.assertGreaterEqual(plan.inflation_m, ROBOT_FOOTPRINT_RADIUS_M - 1.0e-9)

    def test_sealed_box_reports_unreachable(self) -> None:
        planner = GlobalPlanner(sealed_box(3.0, 0.0, 0.90), GlobalPlannerConfig())
        plan = planner.plan((0.0, 0.0), (3.0, 0.0))
        self.assertFalse(plan.reachable)
        self.assertTrue(plan.reason)
        self.assertEqual(len(plan.waypoints), 1)

    def test_room_start_to_object_routes_through_the_doorway(self) -> None:
        episode = room_episode()
        walls = room_obstacles(episode.spec)
        door_lo, door_hi = episode.spec.door_gap
        door_line = episode.spec.interior_x[0] - 0.5 * episode.spec.wall_thickness
        planner = GlobalPlanner(walls, GlobalPlannerConfig())
        for obj in episode.objects:
            with self.subTest(obj=obj.name):
                plan = planner.plan(episode.start_xy, obj.position)
                self.assertTrue(plan.reachable, plan.reason)
                crossings = crossing_y(plan.waypoints, door_line)
                self.assertTrue(crossings, "every route must pass the wall line")
                for y in crossings:
                    self.assertGreater(y, door_lo - 0.05)
                    self.assertLess(y, door_hi + 0.05)

    def test_sampled_rooms_remain_plannable(self) -> None:
        rng = np.random.default_rng(20261004)
        checked = 0
        for _ in range(40):
            episode = sample_room_episode(rng)
            if episode is None:
                continue
            planner = GlobalPlanner(room_obstacles(episode.spec), GlobalPlannerConfig())
            for obj in episode.objects:
                plan = planner.plan(episode.start_xy, obj.position)
                self.assertTrue(plan.reachable, f"{obj.name}: {plan.reason}")
            checked += 1
            if checked >= 5:
                break
        self.assertGreaterEqual(checked, 5, "the sampler refused every attempt")

    def test_corridor_reaches_every_room(self) -> None:
        episode = corridor_episode()
        walls = all_walls(episode.spec, strict=False)
        planner = GlobalPlanner(walls, GlobalPlannerConfig())
        for obj in episode.objects:
            with self.subTest(obj=obj.name):
                plan = planner.plan(episode.start_xy, obj.position)
                self.assertTrue(plan.reachable, plan.reason)

    def test_sealed_corridor_rooms_are_unreachable(self) -> None:
        episode = corridor_episode()
        walls = all_walls(episode.spec, strict=False) + door_plug_walls(episode.spec)
        planner = GlobalPlanner(walls, GlobalPlannerConfig())
        for obj in episode.objects:
            with self.subTest(obj=obj.name):
                plan = planner.plan(episode.start_xy, obj.position)
                self.assertFalse(plan.reachable, "a plugged doorway must not leak")

    # ------------------------------------------------------- teacher wiring

    def test_teacher_uses_the_global_plan_when_a_wall_blocks(self) -> None:
        episode = room_episode()
        walls = room_obstacles(episode.spec)
        target = episode.objects[0].position
        planner = SearchMPCPlanner(
            None,  # the global route never touches the MuJoCo model
            target,
            walls,
            SearchMPCConfig(),
            global_planner=GlobalPlanner(walls, GlobalPlannerConfig()),
        )
        command, info = planner.recommend(_Pose(*episode.start_xy, episode.start_yaw), None)
        self.assertEqual(info["global_planner"], 1.0)
        self.assertEqual(info["global_plan_reachable"], 1.0)
        self.assertGreater(info["global_waypoint_count"], 2)
        self.assertFalse(info["goal_is_target"], "the doorway waypoint is not the target")
        self.assertEqual(len(command), 4)
        self.assertTrue(np.isfinite(command).all())

    def test_teacher_ignores_the_grid_once_the_target_is_visible(self) -> None:
        # Inside the room with a clear line to the object, the planner must
        # collapse to the legacy straight-approach behaviour.
        episode = room_episode()
        walls = room_obstacles(episode.spec)
        target = np.asarray(episode.objects[0].position, dtype=np.float64)
        planner = SearchMPCPlanner(
            None,
            target,
            walls,
            SearchMPCConfig(),
            global_planner=GlobalPlanner(walls, GlobalPlannerConfig()),
        )
        inside = target - np.array((1.20, 0.0))
        _, info = planner.recommend(_Pose(float(inside[0]), float(inside[1]), 0.0), None)
        self.assertEqual(info["global_planner"], 1.0)
        self.assertTrue(info["goal_is_target"])
        self.assertEqual(info["global_waypoint_count"], 0)
        self.assertGreater(float(info["best_forward"]), 0.0)

    def test_legacy_detour_is_untouched_without_a_global_planner(self) -> None:
        # No planner supplied -> the S1 hand-written detour must remain in charge.
        obstacles = [wall("screen", 1.5, 0.0, WALL_HALF_THICKNESS, 1.2)]
        planner = SearchMPCPlanner(None, (3.0, 0.0), obstacles, SearchMPCConfig())
        _, info = planner.recommend(_Pose(0.0, 0.0, 0.0), None)
        self.assertNotIn("global_planner", info)
        self.assertIn("route_phase", info)


if __name__ == "__main__":
    unittest.main()
