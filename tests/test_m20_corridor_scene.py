"""Contract tests for the S3 corridor scene family.

S3 exists to make the search task structural: several rooms off one corridor,
one object per room, every object hidden from the start, and every room sealed
except for its own doorway.  These tests guard exactly those properties, plus
the construction invariant that the corridor wall lines are covered end to end
except at the openings we intend.
"""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from m20pro_vla.sim.corridor import (
    CORRIDOR_WIDTH_M,
    ROOM_OBJECT_MARGIN_M,
    all_walls,
    boundary_intervals,
    corridor_walls,
    default_episode,
    in_room,
    object_room_report,
    occlusion_report,
    reachability_report,
    sample_episode,
    scene_report,
    touching_side,
    validate_multiroom,
    wing_walls,
)
from m20pro_vla.sim.rooms import (
    DOOR_CLEARANCE_MARGIN_M,
    ROBOT_FOOTPRINT_DIAMETER_M,
)


def _replace_wing(spec, index: int, **changes):
    rooms = list(spec.rooms)
    rooms[index] = dataclasses.replace(rooms[index], **changes)
    return dataclasses.replace(spec, rooms=tuple(rooms))


class M20CorridorSceneTests(unittest.TestCase):
    # --- geometry ---------------------------------------------------------

    def test_default_building_passes_validation(self) -> None:
        report = validate_multiroom(default_episode().spec)
        self.assertTrue(report["ok"], report["issues"])

    def test_every_wing_attaches_to_the_side_its_geometry_implies(self) -> None:
        spec = default_episode().spec
        self.assertEqual(touching_side(spec.rooms[0], spec), "north")
        self.assertEqual(touching_side(spec.rooms[1], spec), "south")
        self.assertEqual(touching_side(spec.rooms[2], spec), "east")

    def test_corridor_is_wide_enough_for_a_centred_doorway(self) -> None:
        spec = default_episode().spec
        self.assertGreaterEqual(spec.corridor_width, CORRIDOR_WIDTH_M)
        self.assertGreaterEqual(
            spec.corridor_width,
            spec.rooms[0].door_width + 2.0 * 0.40,
        )

    def test_every_doorway_clears_the_robot_footprint(self) -> None:
        spec = default_episode().spec
        required = ROBOT_FOOTPRINT_DIAMETER_M + DOOR_CLEARANCE_MARGIN_M
        self.assertGreaterEqual(spec.entrance_width, required)
        for wing in spec.rooms:
            self.assertGreaterEqual(wing.door_width, required, wing.name)

    def test_corridor_wall_lines_are_sealed_except_at_the_intended_openings(self) -> None:
        spec = default_episode().spec
        boundary = boundary_intervals(spec)
        for wing in spec.rooms:
            side = touching_side(wing, spec)
            gaps = boundary[side]["gaps"]
            self.assertEqual(len(gaps), 1, f"{side} has {gaps}")
            self.assertAlmostEqual(gaps[0][0], wing.door_gap()[0], places=6)
            self.assertAlmostEqual(gaps[0][1], wing.door_gap()[1], places=6)
        entrance_gaps = boundary[spec.entrance_side]["gaps"]
        self.assertEqual(len(entrance_gaps), 1)
        self.assertAlmostEqual(entrance_gaps[0][0], spec.entrance_gap()[0], places=6)
        self.assertAlmostEqual(entrance_gaps[0][1], spec.entrance_gap()[1], places=6)

    def test_wall_boxes_match_the_construction_rule(self) -> None:
        spec = default_episode().spec
        # Each wing contributes three non-shared walls plus two door flanks.
        expected = len(corridor_walls(spec)) + 5 * len(spec.rooms)
        self.assertEqual(len(all_walls(spec)), expected)
        for wing in spec.rooms:
            self.assertEqual(len(wing_walls(wing, spec)), 3, wing.name)

    # --- solvability ------------------------------------------------------

    def test_objects_are_occluded_from_the_start_pose(self) -> None:
        report = occlusion_report(default_episode())
        self.assertTrue(report["all_blocked"], report["per_object"])

    def test_every_room_is_reachable_with_the_doorways_open(self) -> None:
        report = reachability_report(default_episode(), resolution=0.06, keep_paths=False)
        self.assertTrue(report["open"]["start_cell_free"])
        self.assertTrue(report["open"]["all_rooms_reachable"], report["open"]["rooms"])

    def test_no_room_is_reachable_once_every_opening_is_plugged(self) -> None:
        """Seal test: the rooms must be closed boxes whose only access is a door."""
        report = reachability_report(default_episode(), resolution=0.06, keep_paths=False)
        self.assertTrue(
            report["sealed"]["no_room_reachable"],
            {name: entry["reachable"] for name, entry in report["sealed"]["rooms"].items()},
        )

    def test_each_object_sits_in_a_different_room(self) -> None:
        episode = default_episode()
        report = object_room_report(episode)
        self.assertTrue(report["ok"], report["issues"])
        rooms = list(report["placement"].values())
        self.assertEqual(len(set(rooms)), len(rooms))
        for obj in episode.objects:
            wing = episode.spec.wing_named(report["placement"][obj.name])
            self.assertIsNotNone(wing)
            self.assertTrue(in_room(wing, obj.position, ROOM_OBJECT_MARGIN_M), obj.name)

    def test_full_scene_report_passes(self) -> None:
        report = scene_report(default_episode(), resolution=0.06)
        self.assertTrue(report["ok"])

    # --- rejection --------------------------------------------------------

    def test_narrow_doorway_is_rejected(self) -> None:
        spec = _replace_wing(
            default_episode().spec, 0, door_width=ROBOT_FOOTPRINT_DIAMETER_M + 0.05
        )
        report = validate_multiroom(spec)
        self.assertFalse(report["ok"])
        self.assertTrue(any("narrower than the required" in issue for issue in report["issues"]))

    def test_detached_wing_is_rejected(self) -> None:
        spec = default_episode().spec
        wing = spec.rooms[0]
        shifted = dataclasses.replace(
            wing, interior_y=(wing.interior_y[0] + 0.10, wing.interior_y[1] + 0.10)
        )
        spec = dataclasses.replace(spec, rooms=(shifted,) + spec.rooms[1:])
        report = validate_multiroom(spec)
        self.assertFalse(report["ok"])
        self.assertTrue(any("not attached to the corridor" in issue for issue in report["issues"]))

    def test_narrow_corridor_is_rejected(self) -> None:
        spec = dataclasses.replace(default_episode().spec, corridor_y=(-0.50, 0.50))
        report = validate_multiroom(spec)
        self.assertFalse(report["ok"])
        self.assertTrue(any("corridor width" in issue for issue in report["issues"]))

    def test_doorway_opening_past_the_corridor_end_is_rejected(self) -> None:
        spec = _replace_wing(default_episode().spec, 1, door_center=0.10)
        report = validate_multiroom(spec)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("not inside the corridor span" in issue for issue in report["issues"]),
            report["issues"],
        )

    def test_entrance_sharing_a_side_with_a_wing_is_rejected(self) -> None:
        spec = dataclasses.replace(default_episode().spec, entrance_side="north")
        report = validate_multiroom(spec)
        self.assertFalse(report["ok"])
        self.assertTrue(any("also carries a wing" in issue for issue in report["issues"]))

    def test_overlapping_rooms_are_rejected(self) -> None:
        spec = default_episode().spec
        # Drop the south wing onto the north wing's footprint.
        south = dataclasses.replace(
            spec.rooms[1],
            interior_x=spec.rooms[0].interior_x,
            interior_y=spec.rooms[0].interior_y,
        )
        spec = dataclasses.replace(spec, rooms=(spec.rooms[0], south, spec.rooms[2]))
        report = validate_multiroom(spec)
        self.assertFalse(report["ok"])
        self.assertTrue(any("overlap" in issue for issue in report["issues"]))

    def test_room_expecting_a_missing_object_is_rejected(self) -> None:
        episode = dataclasses.replace(default_episode(), objects=default_episode().objects[:2])
        report = object_room_report(episode)
        self.assertFalse(report["ok"])
        self.assertTrue(any("which is missing" in issue for issue in report["issues"]))

    def test_object_placed_outside_its_room_is_rejected(self) -> None:
        episode = default_episode()
        moved = dataclasses.replace(
            episode.objects[0], position=(episode.objects[0].position[0], 0.0)
        )
        episode = dataclasses.replace(
            episode, objects=(moved,) + episode.objects[1:]
        )
        report = object_room_report(episode)
        self.assertFalse(report["ok"])

    # --- sampler ----------------------------------------------------------

    def test_sampled_episodes_stay_valid_across_seeds(self) -> None:
        rng = np.random.default_rng(20261003)
        accepted = 0
        for _ in range(10):
            episode = sample_episode(rng)
            if episode is None:
                continue
            accepted += 1
            self.assertTrue(validate_multiroom(episode.spec)["ok"])
            self.assertTrue(occlusion_report(episode)["all_blocked"])
            self.assertTrue(object_room_report(episode)["ok"])
            reach = reachability_report(episode, resolution=0.08, keep_paths=False)
            self.assertTrue(reach["ok"], reach)
        self.assertGreaterEqual(accepted, 8, f"sampler accepted only {accepted}/10 attempts")

    def test_sampled_episodes_keep_one_object_per_room(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(3):
            episode = sample_episode(rng)
            if episode is None:
                continue
            self.assertEqual(len(episode.objects), len(episode.spec.rooms))
            rooms = {entry for entry in object_room_report(episode)["placement"].values()}
            self.assertEqual(len(rooms), len(episode.spec.rooms))


if __name__ == "__main__":
    unittest.main()
