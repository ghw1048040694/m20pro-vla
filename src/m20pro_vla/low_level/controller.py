"""Canonical M20 low-level locomotion controller.

Every M20 VLA in this workspace must use this module as its execution layer.
The high-level policy emits a body command; this controller owns joint-level
stance, wheel velocity tracking, attitude feedback, acceleration limiting,
braking, yaw-rate regulation, and diagnostics. It intentionally contains no target or language
logic so it can be reused by SmolVLA, Pi0.5, world-model MPC, and evaluators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np


LOW_LEVEL_CONTRACT = "m20_low_level_v1"
CONTROL_DT = 0.02
PHYSICS_STEPS = 8
WHEEL_RADIUS = 0.09
WHEEL_COMMAND_SCALE = 2.0
WHEEL_YAW_SCALE = 5.0
WHEEL_YAW_RATE_FEEDBACK = 0.20
WHEEL_TARGET_SLEW = 0.16
WHEEL_YAW_TARGET_SLEW = 0.03
WHEEL_NEGATIVE_YAW_TARGET_SLEW = 0.025
WHEEL_STOP_TARGET_SLEW = 0.55
STANCE_HEIGHT = 0.54
STANCE_HEIGHT_KP = 0.25
STANCE_HEIGHT_KD = 0.03
STANCE_ATTITUDE_KP = 0.80
STANCE_RATE_KD = 0.08
MAX_LEG_STANCE_OFFSET = 0.16
TURN_STANCE_ATTITUDE_KP = 1.35
TURN_STANCE_RATE_KD = 0.20
TURN_MAX_LEG_STANCE_OFFSET = 0.24
# Short contact-triggered lift for low obstacle edges. The latch avoids
# chattering while a wheel crosses an obstacle.
TERRAIN_LIFT_HIP_OFFSET = 0.28
TERRAIN_LIFT_KNEE_OFFSET = 0.35
TERRAIN_LIFT_HOLD_STEPS = 18
TERRAIN_LIFT_RAMP = 0.25
TERRAIN_LIFT_DECAY = 0.85
TERRAIN_DRIVE_SCALE = 1.0
TERRAIN_TRACTION_SCALE = 2.5
TERRAIN_TARGET_SLEW = 1.20
TERRAIN_RECOVERY_TILT = math.radians(10.0)
TERRAIN_RECOVERY_CLEAR_TILT = math.radians(4.0)
SAFETY_RECOVERY_HEIGHT = 0.48
SAFETY_RECOVERY_CLEAR_HEIGHT = 0.515

JOINT_NAMES = (
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fl_wheel_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint", "fr_wheel_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint", "hl_wheel_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint", "hr_wheel_joint",
)
LEG_JOINT_NAMES = tuple(name for name in JOINT_NAMES if "wheel" not in name)
WHEEL_JOINT_NAMES = tuple(name for name in JOINT_NAMES if "wheel" in name)
LEG_POSE = np.array(
    [0.0, -0.6, 1.0, 0.0, 0.0, -0.6, 1.0, 0.0,
     0.0, 0.6, -1.0, 0.0, 0.0, 0.6, -1.0, 0.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class M20BodyCommand:
    """Versioned high-level command consumed by the bottom controller."""

    forward: float = 0.0
    lateral: float = 0.0
    yaw: float = 0.0
    stop: bool = False

    @classmethod
    def from_array(cls, action: np.ndarray | list[float] | tuple[float, ...]) -> "M20BodyCommand":
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("M20 body command must be four finite values: forward, lateral, yaw, stop")
        return cls(float(values[0]), float(values[1]), float(values[2]), bool(values[3] >= 0.5))

    def clipped(self) -> "M20BodyCommand":
        return M20BodyCommand(
            forward=float(np.clip(self.forward, -0.40, 0.40)),
            lateral=float(np.clip(self.lateral, -0.20, 0.20)),
            yaw=float(np.clip(self.yaw, -0.15, 0.15)),
            stop=bool(self.stop),
        )

    def as_array(self) -> np.ndarray:
        return np.array((self.forward, self.lateral, self.yaw, float(self.stop)), dtype=np.float64)


@dataclass(frozen=True)
class M20LowLevelControllerState:
    wheel_target: np.ndarray
    last_stance_pose: np.ndarray
    terrain_lift_steps: np.ndarray
    terrain_lift_level: np.ndarray
    safety_recovery_active: bool


@dataclass(frozen=True)
class M20LowLevelDiagnostics:
    contract: str
    command: np.ndarray
    wheel_target: np.ndarray
    base_height: float
    roll: float
    pitch: float
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    lateral_command: float
    lateral_supported: bool
    terrain_lift_active: bool
    safety_recovery_active: bool
    finite: bool


def yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)), dtype=np.float64)


class M20LowLevelController:
    """Reusable M20 stance and wheel locomotion controller.

    This is the single execution implementation for the M20 VLA stack. A
    future learned rough-terrain policy can replace the internal stance law
    while preserving ``M20BodyCommand`` and the ``step`` interface.
    """

    contract = LOW_LEVEL_CONTRACT
    policy_period = CONTROL_DT
    lateral_supported = False

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        self.leg_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in LEG_JOINT_NAMES]
        self.wheel_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in WHEEL_JOINT_NAMES]
        self.leg_actuator_ids = list(range(12))
        self.wheel_actuator_ids = list(range(12, 16))
        if any(identifier < 0 for identifier in self.joint_ids) or model.nu != 16:
            raise RuntimeError("Unexpected M20 joint/actuator contract")
        self.wheel_target = np.zeros(4, dtype=np.float64)
        self.last_stance_pose = LEG_POSE.copy()
        self.wheel_body_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
             for name in ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")],
            dtype=np.int32,
        )
        self.terrain_lift_steps = np.zeros(4, dtype=np.int32)
        self.terrain_lift_level = np.zeros(4, dtype=np.float64)
        self.safety_recovery_active = False

    def reset(
        self,
        data: mujoco.MjData,
        yaw: float = 0.0,
        base_xy: tuple[float, float] | np.ndarray | None = None,
    ) -> None:
        mujoco.mj_resetData(self.model, data)
        if base_xy is None:
            base_xy = (0.0, 0.0)
        base_xy_array = np.asarray(base_xy, dtype=np.float64)
        if base_xy_array.shape != (2,) or not np.isfinite(base_xy_array).all():
            raise ValueError("base_xy must be two finite values")
        data.qpos[0:3] = (float(base_xy_array[0]), float(base_xy_array[1]), STANCE_HEIGHT)
        data.qpos[3:7] = yaw_quaternion(float(yaw))
        for joint_id, position in zip(self.joint_ids, LEG_POSE):
            data.qpos[self.model.jnt_qposadr[joint_id]] = position
        data.qvel[:] = 0.0
        self.wheel_target.fill(0.0)
        self.last_stance_pose = LEG_POSE.copy()
        self.terrain_lift_steps.fill(0)
        self.terrain_lift_level.fill(0.0)
        self.safety_recovery_active = False
        mujoco.mj_forward(self.model, data)

    def snapshot(self) -> M20LowLevelControllerState:
        return M20LowLevelControllerState(
            wheel_target=self.wheel_target.copy(),
            last_stance_pose=self.last_stance_pose.copy(),
            terrain_lift_steps=self.terrain_lift_steps.copy(),
            terrain_lift_level=self.terrain_lift_level.copy(),
            safety_recovery_active=bool(self.safety_recovery_active),
        )

    def restore(self, state: M20LowLevelControllerState) -> None:
        self.wheel_target[:] = state.wheel_target
        self.last_stance_pose[:] = state.last_stance_pose
        self.terrain_lift_steps[:] = state.terrain_lift_steps
        self.terrain_lift_level[:] = state.terrain_lift_level
        self.safety_recovery_active = bool(state.safety_recovery_active)

    def _terrain_contact_flags(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return edge, traction, and all terrain-contact wheel flags."""
        edge_flags = np.zeros(4, dtype=bool)
        traction_flags = np.zeros(4, dtype=bool)
        surface_flags = np.zeros(4, dtype=bool)
        for index in range(data.ncon):
            contact = data.contact[index]
            for geom_id, other_geom_id in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                other_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(other_geom_id)) or ""
                if not other_name.startswith("terrain_"):
                    continue
                body_id = int(self.model.geom_bodyid[int(geom_id)])
                matches = np.flatnonzero(self.wheel_body_ids == body_id)
                if matches.size:
                    wheel_index = int(matches[0])
                    surface_flags[wheel_index] = True
                    if float(np.hypot(contact.frame[0], contact.frame[1])) > 0.20:
                        traction_flags[wheel_index] = True
                    if abs(float(contact.frame[2])) > 0.75:
                        continue
                    edge_flags[wheel_index] = True
        return edge_flags, traction_flags, surface_flags

    def _attitude(self, data: mujoco.MjData) -> tuple[float, float]:
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        rotation = data.xmat[base_id].reshape(3, 3)
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        pitch = math.atan2(float(-rotation[2, 0]), float(np.hypot(rotation[2, 1], rotation[2, 2])))
        return roll, pitch

    def stance_pose(
        self,
        data: mujoco.MjData,
        turning: bool = False,
        *,
        height_offset: float = 0.0,
    ) -> np.ndarray:
        roll, pitch = self._attitude(data)
        angular_velocity = data.qvel[3:6]
        height_error = (STANCE_HEIGHT + float(height_offset)) - float(data.qpos[2])
        attitude_kp = TURN_STANCE_ATTITUDE_KP if turning else STANCE_ATTITUDE_KP
        rate_kd = TURN_STANCE_RATE_KD if turning else STANCE_RATE_KD
        max_offset = TURN_MAX_LEG_STANCE_OFFSET if turning else MAX_LEG_STANCE_OFFSET
        roll_command = np.clip(
            -attitude_kp * roll - rate_kd * angular_velocity[0],
            -max_offset,
            max_offset,
        )
        pitch_command = np.clip(
            -attitude_kp * pitch - rate_kd * angular_velocity[1],
            -max_offset,
            max_offset,
        )
        height_command = np.clip(
            STANCE_HEIGHT_KP * height_error - STANCE_HEIGHT_KD * float(data.qvel[2]),
            -0.08,
            0.08,
        )
        pose = LEG_POSE.copy()
        for leg_index, (front_sign, side_sign) in enumerate(((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))):
            offset = front_sign * pitch_command + side_sign * roll_command
            hipy = leg_index * 4 + 1
            knee = leg_index * 4 + 2
            pose[hipy] = LEG_POSE[hipy] + float(np.clip(offset, -max_offset, max_offset))
            pose[knee] = LEG_POSE[knee] + float(np.clip(-0.35 * height_command, -0.04, 0.04))
            if self.terrain_lift_steps[leg_index] > 0:
                # Front hips move backward and rear hips forward in their
                # respective joint conventions, lifting a contacted wheel.
                level = float(self.terrain_lift_level[leg_index])
                pose[hipy] += -front_sign * TERRAIN_LIFT_HIP_OFFSET * level
                pose[knee] += front_sign * TERRAIN_LIFT_KNEE_OFFSET * level
        self.last_stance_pose = pose
        return pose

    def _command(self, command: M20BodyCommand | np.ndarray | list[float] | tuple[float, ...]) -> M20BodyCommand:
        if isinstance(command, M20BodyCommand):
            result = command
        else:
            result = M20BodyCommand.from_array(command)
        return result.clipped()

    def apply(self, data: mujoco.MjData, command: M20BodyCommand | np.ndarray | list[float] | tuple[float, ...]) -> M20LowLevelDiagnostics:
        command = self._command(command)
        edge_flags, traction_flags, surface_flags = self._terrain_contact_flags(data)
        terrain_contact_active = bool(np.any(surface_flags))
        self.terrain_lift_steps[edge_flags] = TERRAIN_LIFT_HOLD_STEPS
        self.terrain_lift_steps[~edge_flags] = np.maximum(self.terrain_lift_steps[~edge_flags] - 1, 0)
        self.terrain_lift_level[edge_flags] = np.minimum(
            1.0, self.terrain_lift_level[edge_flags] + TERRAIN_LIFT_RAMP
        )
        inactive = ~edge_flags & (self.terrain_lift_steps == 0)
        self.terrain_lift_level[inactive] *= TERRAIN_LIFT_DECAY
        terrain_lift_active = bool(np.any(self.terrain_lift_steps > 0))
        roll, pitch = self._attitude(data)
        tilt = max(abs(roll), abs(pitch))
        recovery_triggered = (
            tilt >= TERRAIN_RECOVERY_TILT
            or float(data.qpos[2]) <= SAFETY_RECOVERY_HEIGHT
        )
        recovery_cleared = (
            tilt <= TERRAIN_RECOVERY_CLEAR_TILT
            and float(data.qpos[2]) >= SAFETY_RECOVERY_CLEAR_HEIGHT
        )
        if recovery_triggered:
            self.safety_recovery_active = True
        elif self.safety_recovery_active and recovery_cleared:
            self.safety_recovery_active = False
        safety_stop = bool(command.stop or self.safety_recovery_active)
        forward = 0.0 if safety_stop else command.forward
        yaw = 0.0 if safety_stop else command.yaw
        if safety_stop:
            # Velocity actuators already supply damping toward the target.
            # Driving the target past zero introduces a reverse impulse that
            # drops the chassis during the stop latch, so brake by coasting to
            # zero instead of actively counter-spinning.
            desired = np.zeros(4, dtype=np.float64)
        else:
            drive_scale = TERRAIN_DRIVE_SCALE if terrain_lift_active else 1.0
            desired = np.full(4, -WHEEL_COMMAND_SCALE * drive_scale * forward / WHEEL_RADIUS, dtype=np.float64)
            yaw_rate = float(data.qvel[5])
            yaw_command = float(np.clip(yaw - WHEEL_YAW_RATE_FEEDBACK * yaw_rate, -0.15, 0.15))
            desired += WHEEL_YAW_SCALE * np.array((yaw_command, -yaw_command, yaw_command, -yaw_command), dtype=np.float64) / WHEEL_RADIUS
            if np.any(traction_flags):
                # Any wheel touching rough terrain needs bounded extra
                # traction; otherwise contact reaction can reverse its spin.
                boost_level = np.maximum(self.terrain_lift_level, 0.75)
                boost = 1.0 + (TERRAIN_TRACTION_SCALE - 1.0) * boost_level
                desired[traction_flags] *= boost[traction_flags]
        yaw_slew = WHEEL_YAW_TARGET_SLEW if yaw >= 0.0 else WHEEL_NEGATIVE_YAW_TARGET_SLEW
        if terrain_lift_active and not safety_stop:
            slew = TERRAIN_TARGET_SLEW
        elif command.stop:
            slew = WHEEL_STOP_TARGET_SLEW
        elif abs(yaw) > 1e-4:
            slew = yaw_slew
        else:
            slew = WHEEL_TARGET_SLEW
        self.wheel_target += np.clip(desired - self.wheel_target, -slew, slew)
        stance_pose = self.stance_pose(
            data,
            turning=(
                abs(yaw) > 1e-4
                or terrain_lift_active
                or terrain_contact_active
                or self.safety_recovery_active
                or safety_stop
            ),
            height_offset=0.05 if safety_stop else 0.0,
        )
        for index, actuator in enumerate(self.leg_actuator_ids):
            data.ctrl[actuator] = stance_pose[index + (index // 3)]
        for index, actuator in enumerate(self.wheel_actuator_ids):
            data.ctrl[actuator] = float(np.clip(self.wheel_target[index], -20.0, 20.0))
        roll, pitch = self._attitude(data)
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        return M20LowLevelDiagnostics(
            contract=self.contract, command=command.as_array(), wheel_target=self.wheel_target.copy(),
            base_height=float(data.qpos[2]), roll=roll, pitch=pitch,
            linear_velocity=data.qvel[:3].copy(), angular_velocity=data.qvel[3:6].copy(),
            lateral_command=command.lateral, lateral_supported=self.lateral_supported,
            terrain_lift_active=terrain_lift_active,
            safety_recovery_active=self.safety_recovery_active,
            finite=finite,
        )

    def step(self, data: mujoco.MjData, command: M20BodyCommand | np.ndarray | list[float] | tuple[float, ...]) -> M20LowLevelDiagnostics:
        diagnostics = self.apply(data, command)
        for _ in range(PHYSICS_STEPS):
            mujoco.mj_step(self.model, data)
        return diagnostics


# Compatibility name for old datasets/replayers. New code must import the
# canonical class directly from this module.
M20NativeController = M20LowLevelController
