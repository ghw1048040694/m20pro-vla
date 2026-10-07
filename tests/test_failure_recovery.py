import unittest

import numpy as np

from m20pro_vla.data.failure_recovery import build_recovery_planner, restore_controller_state, select_states
from m20pro_vla.low_level.policy_v5 import M20V5PolicyController, M20V5PolicyControllerState


class FailureRecoveryTest(unittest.TestCase):
    def test_restores_real_policy_snapshot_dataclass(self):
        # Snapshot/restore require no inference session; exercise the actual API.
        controller = object.__new__(M20V5PolicyController)
        controller.last_action = np.zeros(16)
        controller.safety_recovery_active = False
        state = M20V5PolicyControllerState(np.arange(16), True)
        restore_controller_state(controller, state, backend="v5", step=25)
        np.testing.assert_array_equal(controller.last_action, state.last_action)
        self.assertTrue(controller.safety_recovery_active)
        with self.assertRaisesRegex(ValueError, "matching backend"):
            restore_controller_state(controller, {"last_action": np.zeros(16)}, backend="analytic", step=25)
        np.testing.assert_array_equal(controller.last_action, state.last_action)

    def test_room_recovery_uses_doorway_aware_teacher(self):
        for kind in ("s2", "s3"):
            with self.subTest(kind=kind):
                planner = build_recovery_planner(None, (1.0, 0.0), [], {"scene_kind": kind}, 0.95)
                self.assertIsNotNone(planner.global_planner)
                self.assertEqual(planner.config.target_radius, 0.95)
        self.assertIsNone(build_recovery_planner(None, (1.0, 0.0), [], {}, 0.95).global_planner)

    def test_selects_distinct_near_goal_and_near_obstacle_states(self):
        states = [
            {"step": 0, "target_distance": 3.0, "obstacle_clearance": 2.0},
            {"step": 25, "target_distance": 2.0, "obstacle_clearance": 0.6},
            {"step": 50, "target_distance": 1.0, "obstacle_clearance": 0.2},
            {"step": 75, "target_distance": 0.8, "obstacle_clearance": 0.4},
        ]
        self.assertEqual([state["step"] for state in select_states(states, 2)], [75, 25])

    def test_empty_states_do_not_generate_examples(self):
        self.assertEqual(select_states([], 2), [])
        with self.assertRaises(ValueError):
            select_states([], 0)


if __name__ == "__main__":
    unittest.main()
