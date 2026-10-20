from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from m20pro_vla.low_level.shield import (
    DEFAULT_SAFETY_STOP_DISTANCE_M,
    DEFAULT_SUCCESS_RADIUS_M,
    LIDAR_TO_BASE_OFFSET_M,
    approach_margin,
    lidar_safety_shield,
    validate_stop_geometry,
)

try:  # The player only imports cleanly inside the lerobot environment.
    from scripts.mujoco.play_m20_smolvla import apply_evaluation_config

    HAVE_PLAYER = True
except Exception:  # pragma: no cover - exercised only where lerobot is absent
    HAVE_PLAYER = False

EXPERIMENT_CONFIG = Path(__file__).resolve().parents[1] / "configs/experiment.json"


class SmolVLAClosedLoopSafetyTest(unittest.TestCase):
    def test_emergency_lidar_response_is_a_hard_stop(self) -> None:
        lidar = np.full(72, 2.0, dtype=np.float64)
        lidar[32:41] = 0.4

        command, reason = lidar_safety_shield(
            np.asarray((0.3, 0.0, 0.15, 0.0)),
            lidar,
            stop_distance=0.62,
            slow_distance=1.25,
        )

        np.testing.assert_array_equal(command, np.asarray((0.0, 0.0, 0.0, 1.0)))
        self.assertEqual(reason, "emergency_stop")

    def test_slow_zone_does_not_inject_a_turn(self) -> None:
        lidar = np.full(72, 2.0, dtype=np.float64)
        lidar[32:41] = 0.9

        command, reason = lidar_safety_shield(
            np.asarray((0.3, 0.0, 0.02, 0.0)),
            lidar,
            stop_distance=0.62,
            slow_distance=1.25,
        )

        self.assertEqual(reason, "slow")
        self.assertLessEqual(command[0], 0.12)
        self.assertEqual(command[2], 0.02)
        self.assertEqual(command[3], 0.0)


class StopGeometryTest(unittest.TestCase):
    """The safety stop must sit inside the ring the episode is graded on."""

    def test_shipped_defaults_satisfy_the_invariant(self) -> None:
        validate_stop_geometry(
            success_radius=DEFAULT_SUCCESS_RADIUS_M,
            stop_distance=DEFAULT_SAFETY_STOP_DISTANCE_M,
            slow_distance=1.25,
        )

    def test_stop_outside_the_success_radius_is_rejected(self) -> None:
        # The 2026-10-03 regression: 0.62 m stop vs 0.45 m radius braked the
        # robot 0.17 m short of the ring and scored three arrivals as failures.
        with self.assertRaises(ValueError):
            validate_stop_geometry(success_radius=0.45, stop_distance=0.62, slow_distance=1.25)

    def test_stop_beyond_the_slow_zone_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_stop_geometry(success_radius=0.70, stop_distance=1.30, slow_distance=1.25)

    def test_non_positive_ranges_are_rejected(self) -> None:
        for kwargs in (
            {"success_radius": 0.0, "stop_distance": 0.50, "slow_distance": 1.25},
            {"success_radius": 0.70, "stop_distance": 0.0, "slow_distance": 1.25},
            {"success_radius": 0.70, "stop_distance": float("nan"), "slow_distance": 1.25},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    validate_stop_geometry(**kwargs)

    def test_shipped_pair_leaves_room_for_the_base_to_enter(self) -> None:
        # The shipped pair moved from (0.50 / 0.70) to (0.50 / 0.95) on
        # 2026-10-04. The old pair sat at 0.70 - (0.50 + 0.202) = -2 mm, i.e.
        # nominally unreachable; the new one leaves +0.248 m. The LiDAR/base frame
        # gap and the stop radius are both load-bearing, so pin the arithmetic
        # here - a future change to either number has to face the coupling instead
        # of rediscovering it.
        margin = approach_margin(DEFAULT_SUCCESS_RADIUS_M, DEFAULT_SAFETY_STOP_DISTANCE_M)
        self.assertAlmostEqual(margin, 0.95 - 0.50 - LIDAR_TO_BASE_OFFSET_M, places=9)
        self.assertGreater(margin, 0.20)

    def test_the_retired_0_70_pair_was_nominally_unreachable(self) -> None:
        # Kept as the documented counter-example: at a 0.70 m ring the frame gap
        # alone ate the whole 0.20 m of nominal clearance.
        self.assertLess(approach_margin(0.70, 0.50), 0.0)

    def test_stop_radius_stays_inside_the_visible_band(self) -> None:
        # M20-STOP-DIAG-02 + diag/stop_rule_feasibility.py: on the 47
        # m20_hidden_search_v2 episodes the target is best framed at ~0.98 m and
        # leaves the camera through the bottom edge closer in. At a 0.70 m stop,
        # 20 of the 29 teacher arrivals carried *zero* target pixels, so the stop
        # label could not be decoded from the observation the policy gets. A stop
        # at 0.95 m carries a median of 812 pixels with none below the 80-pixel
        # gate. Keep the ring inside that visible band.
        self.assertGreaterEqual(DEFAULT_SUCCESS_RADIUS_M, 0.90)
        self.assertLessEqual(DEFAULT_SUCCESS_RADIUS_M, 1.10)

    def test_a_negative_margin_means_the_ring_is_unreachable(self) -> None:
        # The retired pairing: a 0.62 m stop could never let a base enter a
        # 0.45 m ring, so every arrival was scored as a failure.
        self.assertLess(approach_margin(0.45, 0.62), 0.0)


class ExperimentConfigGeometryTest(unittest.TestCase):
    """Guard the checked-in evaluation geometry, not just the code path."""

    def test_experiment_config_geometry_respects_the_invariant(self) -> None:
        config = json.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
        evaluation = config["smolvla_evaluation"]
        validate_stop_geometry(
            success_radius=float(evaluation["success_radius"]),
            stop_distance=float(evaluation["safety_stop_distance"]),
            slow_distance=1.25,
        )


@unittest.skipUnless(HAVE_PLAYER, "play_m20_smolvla needs the lerobot stack")
class EvaluationConfigWiringTest(unittest.TestCase):
    def _args(self) -> object:
        from m20pro_vla.eval.acceptance import EVALUATION_KNOBS
        return type("Args", (), {attr: None for attr, *_ in EVALUATION_KNOBS})()

    def test_config_supplies_the_geometry(self) -> None:
        args = self._args()
        apply_evaluation_config(
            args, {"smolvla_evaluation": {"success_radius": 0.8, "safety_stop_distance": 0.4}}
        )
        self.assertEqual(args.success_radius, 0.8)
        self.assertEqual(args.safety_stop_distance, 0.4)

    def test_explicit_cli_values_win(self) -> None:
        args = self._args()
        args.success_radius = 0.55
        args.safety_stop_distance = 0.30
        apply_evaluation_config(
            args, {"smolvla_evaluation": {"success_radius": 0.8, "safety_stop_distance": 0.4}}
        )
        self.assertEqual(args.success_radius, 0.55)
        self.assertEqual(args.safety_stop_distance, 0.30)

    def test_missing_config_falls_back_to_the_shipped_defaults(self) -> None:
        args = self._args()
        apply_evaluation_config(args, {})
        self.assertEqual(args.success_radius, DEFAULT_SUCCESS_RADIUS_M)
        self.assertEqual(args.safety_stop_distance, DEFAULT_SAFETY_STOP_DISTANCE_M)


if __name__ == "__main__":
    unittest.main()
