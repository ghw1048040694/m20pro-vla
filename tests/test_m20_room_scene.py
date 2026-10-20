"""Contract tests for the S2 room scene family.

The room exists to make the search task structurally harder, so these tests
guard the two properties that give it meaning: the doorway is physically
passable, and the task objects are not visible from the start pose.
"""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from m20pro_vla.sim.rooms import (
    DOOR_CLEARANCE_MARGIN_M,
    ROBOT_FOOTPRINT_DIAMETER_M,
    default_episode,
    occlusion_report,
    room_obstacles,
    sample_episode,
    validate_room,
)


class M20RoomSceneTests(unittest.TestCase):
    def test_default_room_passes_validation(self) -> None:
        report = validate_room(default_episode().spec)
        self.assertTrue(report["ok"], report["issues"])

    def test_door_clears_robot_footprint_with_margin(self) -> None:
        metrics = validate_room(default_episode().spec)["metrics"]
        self.assertGreaterEqual(
            metrics["door_clear_width_m"],
            ROBOT_FOOTPRINT_DIAMETER_M + DOOR_CLEARANCE_MARGIN_M,
        )
        self.assertGreater(metrics["door_side_clearance_m"], 0.0)

    def test_wall_height_clears_both_sensors(self) -> None:
        metrics = validate_room(default_episode().spec)["metrics"]
        self.assertGreaterEqual(metrics["wall_height_m"], 0.75)

    def test_doorway_is_an_actual_gap_in_the_wall(self) -> None:
        spec = default_episode().spec
        lo, hi = spec.door_gap
        door_walls = [
            wall for wall in room_obstacles(spec) if wall.name.startswith(f"room_wall_{spec.door_wall}")
        ]
        self.assertEqual(len(door_walls), 2)
        for wall in door_walls:
            wall_lo = float(wall.position[1]) - float(wall.size[1])
            wall_hi = float(wall.position[1]) + float(wall.size[1])
            self.assertTrue(
                wall_hi <= lo + 1.0e-9 or wall_lo >= hi - 1.0e-9,
                f"{wall.name} overlaps the doorway span [{lo:.2f}, {hi:.2f}]",
            )

    def test_four_sides_close_into_five_wall_boxes(self) -> None:
        self.assertEqual(len(room_obstacles(default_episode().spec)), 5)

    def test_every_object_is_occluded_from_the_start_pose(self) -> None:
        report = occlusion_report(default_episode())
        self.assertTrue(report["all_blocked"], report["per_object"])

    def test_narrow_door_is_rejected(self) -> None:
        spec = dataclasses.replace(default_episode().spec, door_width=ROBOT_FOOTPRINT_DIAMETER_M + 0.05)
        report = validate_room(spec)
        self.assertFalse(report["ok"])
        self.assertTrue(any("narrower than the required" in issue for issue in report["issues"]))

    def test_short_wall_is_rejected(self) -> None:
        spec = dataclasses.replace(default_episode().spec, wall_height=0.30)
        report = validate_room(spec)
        self.assertFalse(report["ok"])
        self.assertTrue(any("does not clear" in issue for issue in report["issues"]))

    def test_door_at_interior_corner_is_rejected(self) -> None:
        spec = dataclasses.replace(default_episode().spec, door_center=-1.95)
        report = validate_room(spec)
        self.assertFalse(report["ok"])
        self.assertTrue(any("interior corners" in issue for issue in report["issues"]))

    def test_sampled_episodes_stay_valid_across_seeds(self) -> None:
        rng = np.random.default_rng(20261003)
        accepted = 0
        for _ in range(12):
            episode = sample_episode(rng)
            if episode is None:
                continue
            accepted += 1
            self.assertTrue(validate_room(episode.spec)["ok"])
            self.assertTrue(occlusion_report(episode)["all_blocked"])
        self.assertGreaterEqual(accepted, 10, f"sampler accepted only {accepted}/12 attempts")


if __name__ == "__main__":
    unittest.main()
