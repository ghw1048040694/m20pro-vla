"""Official M20 locomotion policy adapter for the MuJoCo-first VLA stack.

The adapter is deliberately limited to body-state plus a 3D velocity command.
It does not consume RGB, language, target pose, or simulator semantic labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import onnxruntime as ort

from m20pro_vla.sim.scene import ensure_official_policy


ROBOT_ORDER = (
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fl_wheel_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint", "fr_wheel_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint", "hl_wheel_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint", "hr_wheel_joint",
)
POLICY_ORDER = (
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
)
DEFAULT_ROBOT_POSITION = np.array(
    [0.0, -0.6, 1.0, 0.0, 0.0, -0.6, 1.0, 0.0, 0.0, 0.6, -1.0, 0.0, 0.0, 0.6, -1.0, 0.0],
    dtype=np.float64,
)
DEFAULT_POLICY_POSITION = np.array(
    [0.0, -0.6, 1.0, 0.0, -0.6, 1.0, 0.0, 0.6, -1.0, 0.0, 0.6, -1.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
DEFAULT_BASE_POSITION = np.array((0.0, 0.0, 0.54), dtype=np.float64)
DEFAULT_BASE_QUATERNION = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
ACTION_SCALE_ROBOT = np.array(
    [0.125, 0.25, 0.25, 5.0] * 4,
    dtype=np.float64,
)
KP = np.array([80.0, 80.0, 80.0, 0.0] * 4, dtype=np.float64)
KD = np.array([2.0, 2.0, 2.0, 0.6] * 4, dtype=np.float64)


def _rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class LowLevelStep:
    observation: np.ndarray
    policy_action: np.ndarray
    motor_torque: np.ndarray


class M20OfficialLocomotion:
    """Map a velocity command to stable M20 torque controls through the official ONNX policy."""

    policy_period = 0.02

    def __init__(self, model: mujoco.MjModel) -> None:
        if model.nu != len(ROBOT_ORDER):
            raise ValueError(f"Expected {len(ROBOT_ORDER)} M20 actuators, got {model.nu}")
        self.model = model
        self.robot_joint_ids = self._named_ids(mujoco.mjtObj.mjOBJ_JOINT, ROBOT_ORDER)
        self.policy_joint_ids = self._named_ids(mujoco.mjtObj.mjOBJ_JOINT, POLICY_ORDER)
        self.robot_actuator_ids = self._named_ids(mujoco.mjtObj.mjOBJ_ACTUATOR, ROBOT_ORDER)
        self.base_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
        if self.base_joint_id < 0:
            raise RuntimeError("M20 model does not expose the floating base joint")
        self.last_policy_action = np.zeros(len(POLICY_ORDER), dtype=np.float64)
        self.session = ort.InferenceSession(
            str(ensure_official_policy()), providers=["CPUExecutionProvider"]
        )
        input_spec = self.session.get_inputs()[0]
        output_spec = self.session.get_outputs()[0]
        if input_spec.name != "obs" or output_spec.name != "actions":
            raise RuntimeError(f"Unexpected official ONNX I/O: {input_spec.name} -> {output_spec.name}")

    def _named_ids(self, object_type: mujoco.mjtObj, names: tuple[str, ...]) -> np.ndarray:
        ids = np.array([mujoco.mj_name2id(self.model, object_type, name) for name in names], dtype=np.int32)
        if np.any(ids < 0):
            missing = [name for name, identifier in zip(names, ids) if identifier < 0]
            raise RuntimeError(f"M20 model is missing named controls: {missing}")
        return ids

    def reset(self, data: mujoco.MjData) -> None:
        mujoco.mj_resetData(self.model, data)
        # MuJoCo compiles the free joint at a different default height than
        # the controller's trained standing pose.  Set the floating base
        # explicitly before the first policy observation.
        base_qpos = self.model.jnt_qposadr[self.base_joint_id]
        base_dof = self.model.jnt_dofadr[self.base_joint_id]
        data.qpos[base_qpos : base_qpos + 3] = DEFAULT_BASE_POSITION
        data.qpos[base_qpos + 3 : base_qpos + 7] = DEFAULT_BASE_QUATERNION
        data.qvel[base_dof : base_dof + 6] = 0.0
        for joint_id, position in zip(self.robot_joint_ids, DEFAULT_ROBOT_POSITION):
            data.qpos[self.model.jnt_qposadr[joint_id]] = position
        self.last_policy_action.fill(0.0)
        mujoco.mj_forward(self.model, data)

    def observation(self, data: mujoco.MjData, command: np.ndarray) -> np.ndarray:
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("Low-level command must be three finite values: forward, lateral, yaw")

        base_qpos = self.model.jnt_qposadr[self.base_joint_id]
        base_dof = self.model.jnt_dofadr[self.base_joint_id]
        rotation = _rotation_matrix(data.qpos[base_qpos + 3: base_qpos + 7])
        base_omega = rotation.T @ data.qvel[base_dof + 3: base_dof + 6]
        projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0])

        policy_position = np.array(
            [data.qpos[self.model.jnt_qposadr[joint_id]] for joint_id in self.policy_joint_ids], dtype=np.float64
        )
        policy_velocity = np.array(
            [data.qvel[self.model.jnt_dofadr[joint_id]] for joint_id in self.policy_joint_ids], dtype=np.float64
        )
        policy_position[12:] = 0.0
        observation = np.concatenate(
            (
                0.25 * base_omega,
                projected_gravity,
                command,
                policy_position - DEFAULT_POLICY_POSITION,
                0.05 * policy_velocity,
                self.last_policy_action,
            )
        ).astype(np.float32)
        if observation.shape != (57,):
            raise RuntimeError(f"Unexpected official policy observation shape: {observation.shape}")
        return observation

    def apply(self, data: mujoco.MjData, command: np.ndarray) -> LowLevelStep:
        observation = self.observation(data, command)
        policy_action = self.session.run(["actions"], {"obs": observation[None]})[0].reshape(-1).astype(np.float64)
        if policy_action.shape != (16,) or not np.isfinite(policy_action).all():
            raise RuntimeError("Official locomotion policy returned an invalid action")
        self.last_policy_action = policy_action.copy()

        policy_index = {name: index for index, name in enumerate(POLICY_ORDER)}
        action_robot = np.array([policy_action[policy_index[name]] for name in ROBOT_ORDER], dtype=np.float64)
        goal_robot = DEFAULT_ROBOT_POSITION + ACTION_SCALE_ROBOT * action_robot
        torque = np.empty(len(ROBOT_ORDER), dtype=np.float64)
        for index, joint_id in enumerate(self.robot_joint_ids):
            position = data.qpos[self.model.jnt_qposadr[joint_id]]
            velocity = data.qvel[self.model.jnt_dofadr[joint_id]]
            if index % 4 == 3:
                torque[index] = KD[index] * (goal_robot[index] - velocity)
            else:
                torque[index] = KP[index] * (goal_robot[index] - position) - KD[index] * velocity
        control_range = self.model.actuator_ctrlrange[self.robot_actuator_ids]
        torque = np.clip(torque, control_range[:, 0], control_range[:, 1])
        data.ctrl[self.robot_actuator_ids] = torque
        return LowLevelStep(observation=observation, policy_action=policy_action, motor_torque=torque)

    def step(self, data: mujoco.MjData, command: np.ndarray) -> LowLevelStep:
        physics_steps = round(self.policy_period / self.model.opt.timestep)
        if not np.isclose(physics_steps * self.model.opt.timestep, self.policy_period):
            raise RuntimeError("MuJoCo timestep must divide the official policy period exactly")
        result = self.apply(data, command)
        for _ in range(physics_steps):
            mujoco.mj_step(self.model, data)
        return result
