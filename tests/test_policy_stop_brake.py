import unittest
import numpy as np
from test_policy_body_velocity import PolicyBodyVelocityTest
from m20pro_vla.low_level.policy_v5 import WHEEL_ACTION_ACTUATORS, LEG_ACTION_ACTUATORS

class PolicyStopBrakeTest(PolicyBodyVelocityTest):
    def setUp(self):
        super().setUp()
        self.controller.reset(self.data)
        self.controller._infer = lambda observation: np.full(16, .1)

    def test_stop_holds_attained_pose_and_wheel_targets_decay(self):
        self.controller.apply(self.data, [.2, 0, 0, 0])
        pose = self.data.qpos[self.controller._leg_qpos].copy()
        previous = self.controller._ctrl_target[12:].copy()
        diagnostic = self.controller.apply(self.data, [0, 0, 0, 1])
        np.testing.assert_array_equal(self.data.ctrl[:12], pose)
        np.testing.assert_allclose(self.data.ctrl[12:], previous - .2)
        np.testing.assert_array_equal(diagnostic.wheel_target, self.data.ctrl[12:])
        for _ in range(5): self.controller.apply(self.data, [0, 0, 0, 1])
        np.testing.assert_array_equal(self.data.ctrl[12:], np.zeros(4))
        np.testing.assert_array_equal(self.controller.last_action[12:], np.zeros(4))

    def test_stop_snapshot_restore_reproduces_next_actuator_targets(self):
        self.controller.apply(self.data, [.2, 0, 0, 0])
        self.controller.apply(self.data, [0, 0, 0, 1])
        state = self.controller.snapshot()
        self.controller.apply(self.data, [0, 0, 0, 1]); expected = self.data.ctrl.copy()
        self.controller.apply(self.data, [.2, 0, 0, 0])
        self.controller.restore(state)
        self.controller.apply(self.data, [0, 0, 0, 1])
        np.testing.assert_array_equal(self.data.ctrl, expected)
        np.testing.assert_array_equal(state.stop_pose, self.controller._stop_pose)

    def test_resume_and_reset_clear_stop_hold(self):
        self.controller.apply(self.data, [0, 0, 0, 1])
        self.controller.apply(self.data, [-.2, 0, 0, 0])
        self.assertIsNone(self.controller._stop_pose)
        np.testing.assert_array_equal(self.controller.last_action, np.full(16, .1))
        self.controller.reset(self.data)
        self.assertIsNone(self.controller._stop_pose)
        np.testing.assert_array_equal(self.controller._brake_wheels, np.zeros(4))

if __name__ == '__main__': unittest.main()
