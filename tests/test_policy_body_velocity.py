import unittest

import mujoco
import numpy as np

from m20pro_vla.low_level.controller import yaw_quaternion
from m20pro_vla.low_level.policy_v5 import (
    ACTION_JOINT_NAMES,
    ANGULAR_VELOCITY_SCALE,
    M20V5PolicyController,
)


class PolicyBodyVelocityTest(unittest.TestCase):
    def setUp(self):
        # Exercise the actual MuJoCo free-joint and controller observation API;
        # weights and the ignored robot mesh assets are unnecessary here.
        bodies = ''.join(
            f'<body name="{name.removesuffix("_joint")}"><joint name="{name}"/>'
            '<geom type="sphere" size="0.02" mass="0.1"/></body>'
            for name in ACTION_JOINT_NAMES
        )
        actuators = ''.join(f'<motor joint="{name}"/>' for name in ACTION_JOINT_NAMES)
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="base_link"><freejoint/>'
            '<inertial pos="0 0 0" mass="1" diaginertia="0.1 0.2 0.3" '
            'quat="0.70710678 0.70710678 0 0"/>'
            f'{bodies}</body></worldbody><actuator>{actuators}</actuator></mujoco>'
        )
        self.data = mujoco.MjData(self.model)
        self.controller = M20V5PolicyController(self.model)

    def test_angular_feedback_matches_local_body_api_at_every_heading(self):
        for yaw in (0., .76, 2.324, 2.58, -2.4):
            with self.subTest(yaw=yaw):
                self.controller.reset(self.data, yaw=yaw)
                self.data.qvel[3:6] = (.3, -.2, .1)
                mujoco.mj_forward(self.model, self.data)
                local = np.zeros(6)
                # BODY would select the rotated inertial frame, not base_link.
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
                                        self.controller._base_body_id, local, 1)
                observation = self.controller._observation(self.data, np.zeros(3))[0]
                self.assertEqual(observation.shape, (53,))
                np.testing.assert_allclose(observation[:3], local[:3] * ANGULAR_VELOCITY_SCALE,
                                           rtol=1e-6, atol=1e-8)

    def test_gravity_projection_still_uses_body_attitude(self):
        self.controller.reset(self.data, yaw=2.324)
        roll = np.array((np.cos(.2), np.sin(.2), 0., 0.))
        mujoco.mju_mulQuat(self.data.qpos[3:7], yaw_quaternion(2.324), roll)
        mujoco.mj_forward(self.model, self.data)
        observation = self.controller._observation(self.data, np.zeros(3))[0]
        np.testing.assert_allclose(observation[3:6], (0., -np.sin(.4), -np.cos(.4)), atol=1e-7)


if __name__ == '__main__':
    unittest.main()
