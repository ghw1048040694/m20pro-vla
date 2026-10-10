"""Learned M20 low-level locomotion policy (v5 ONNX expert).

``M20LowLevelController`` documents that a learned policy may replace the
internal stance law while preserving ``M20BodyCommand`` and the ``step``
interface. This module is that replacement: a 53 -> 16 ONNX policy trained in
Isaac Lab on the flat velocity task, wrapped so callers cannot tell the
difference apart from the execution quality.

Background
----------
The expert was trained against a different simulator, so two conventions differ
from what a naive port would assume. Both were measured, not guessed:

* The policy's **action** vector is joint-grouped (``fl_hipx, fl_hipy,
  fl_knee, fr_hipx, ...``), and that is also this model's MJCF actuator order.
* Its **observation** joint slots are *not* joint-grouped. ``joint_pos_rel``
  and ``joint_vel_rel`` are emitted in ascending articulation-index order
  (``fl_hipx, fr_hipx, hl_hipx, hr_hipx, fl_hipy, ...``), while ``last_action``
  stays in action order. Feeding action order into the observation slots
  scrambles 28 of the 53 inputs.

Observation (53)::

    [ 0, 3)  base angular velocity, body frame, x0.25
    [ 3, 6)  projected gravity, body frame
    [ 6, 9)  velocity command (forward, lateral, yaw)
    [ 9,21)  joint position relative to the default stance, 12 legs
    [21,37)  joint velocity, 16 joints, x0.05
    [37,53)  previous action, 16

Action (16)::

    leg   q_des = STANCE + LEG_ACTION_SCALE * a
    wheel qd_des = WHEEL_ACTION_SCALE * a

MuJoCo-specific conventions used here:

* Free-joint ``qvel[3:6]`` already holds angular velocity in the local body
  frame. Rotating it again makes the policy feedback depend on world heading.
* ``M20LowLevelDiagnostics`` reports that same local angular velocity, as does
  the analytic controller. Projected gravity still requires ``R.T``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from .controller import (
    CONTROL_DT,
    LEG_JOINT_NAMES,
    M20BodyCommand,
    M20LowLevelController,
    M20LowLevelDiagnostics,
    WHEEL_JOINT_NAMES,
    yaw_quaternion,
)

POLICY_CONTRACT = "m20_low_level_v5_policy"

# The weights are a large binary artifact, so they live in the ignored runtime
# tree rather than in the package. Point at another file with the constructor
# argument or the environment variable.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ONNX_PATH = _REPOSITORY_ROOT / ".runtime/policies/m20_flat_v5/policy_single.onnx"
ONNX_PATH_ENV = "M20_V5_POLICY_ONNX"

# The policy's own default stance. This is NOT ``LEG_POSE``; the analytic
# controller and the expert use different nominal leg angles.
STANCE = np.array(
    [0.0, -0.3, 0.6] * 2 + [0.0, 0.3, -0.6] * 2,
    dtype=np.float64,
)
LEG_ACTION_SCALE = np.array([0.125, 0.25, 0.25] * 4, dtype=np.float64)
WHEEL_ACTION_SCALE = 5.0

ANGULAR_VELOCITY_SCALE = 0.25
JOINT_VELOCITY_SCALE = 0.05

SPAWN_HEIGHT = 0.58
OBSERVATION_SIZE = 53
ACTION_SIZE = 16

# Observation slot order: ascending articulation index, i.e. joint type major.
OBSERVATION_JOINT_NAMES = (
    "fl_hipx_joint", "fr_hipx_joint", "hl_hipx_joint", "hr_hipx_joint",
    "fl_hipy_joint", "fr_hipy_joint", "hl_hipy_joint", "hr_hipy_joint",
    "fl_knee_joint", "fr_knee_joint", "hl_knee_joint", "hr_knee_joint",
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
)
OBSERVATION_LEG_JOINT_NAMES = tuple(
    name for name in OBSERVATION_JOINT_NAMES if "wheel" not in name
)
STANCE_BY_NAME = dict(zip(LEG_JOINT_NAMES, STANCE))

# Actuator slots in this MJCF are legs first (grouped by leg), then the four
# wheels - which is exactly the policy's action order. That is NOT the order of
# ``controller.JOINT_NAMES``, which interleaves each wheel with its leg and
# only describes the nominal joint list used to read LEG_POSE.
ACTION_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES
LEG_ACTION_ACTUATORS = tuple(range(len(LEG_JOINT_NAMES)))
WHEEL_ACTION_ACTUATORS = tuple(
    range(len(LEG_JOINT_NAMES), len(ACTION_JOINT_NAMES))
)

RECOVERY_TRIGGER_TILT_RAD = np.radians(10.0)
RECOVERY_CLEAR_TILT_RAD = np.radians(4.0)
RECOVERY_TRIGGER_HEIGHT = 0.48
RECOVERY_CLEAR_HEIGHT = 0.49


@dataclass(frozen=True)
class M20V5PolicyControllerState:
    """Rollout state that a planner must carry across a hypothetical step."""

    last_action: np.ndarray
    safety_recovery_active: bool
    stop_pose: np.ndarray | None = None
    brake_wheels: np.ndarray | None = None


def resolve_onnx_path(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    from_env = os.environ.get(ONNX_PATH_ENV)
    if from_env:
        return Path(from_env)
    return DEFAULT_ONNX_PATH


class M20V5PolicyController(M20LowLevelController):
    """Body-command execution layer backed by the learned v5 expert.

    Drop-in for :class:`M20LowLevelController`: same constructor signature,
    same ``reset`` / ``step`` / ``snapshot`` / ``restore`` surface, same
    diagnostics type and the same four-element body command.
    """

    contract = POLICY_CONTRACT
    policy_period = CONTROL_DT
    # The learned expert does support a lateral command, but the fixed-wheel
    # hardware bridge does not, so the published capability stays unchanged.
    lateral_supported = False

    def __init__(self, model: mujoco.MjModel, onnx_path: str | Path | None = None,
                 providers: tuple[str, ...] = ("CPUExecutionProvider",),
                 teacher_motion_limits: tuple[float, float] | None = None,
                 inference_threads: int | None = None):
        super().__init__(model)
        if teacher_motion_limits is not None:
            if tuple(teacher_motion_limits) != (.50, .40):
                raise ValueError('Only the explicitly validated teacher .50/.40 profile is supported')
        self.teacher_motion_limits = teacher_motion_limits
        if inference_threads is not None and inference_threads < 1:
            raise ValueError("inference_threads must be positive")
        self.inference_threads = inference_threads
        self._joint_id = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ACTION_JOINT_NAMES
        }
        self._verify_actuator_order(model)
        if model.nu != ACTION_SIZE:
            raise RuntimeError(f"Expected {ACTION_SIZE} M20 actuators, found {model.nu}")

        self._leg_observation_qpos = np.array(
            [model.jnt_qposadr[self._joint_id[name]] for name in OBSERVATION_LEG_JOINT_NAMES],
            dtype=np.intp,
        )
        self._observation_dof = np.array(
            [model.jnt_dofadr[self._joint_id[name]] for name in OBSERVATION_JOINT_NAMES],
            dtype=np.intp,
        )
        self._leg_observation_default = np.array(
            [STANCE_BY_NAME[name] for name in OBSERVATION_LEG_JOINT_NAMES],
            dtype=np.float64,
        )
        self._leg_qpos = np.array(
            [model.jnt_qposadr[self._joint_id[name]] for name in LEG_JOINT_NAMES],
            dtype=np.intp,
        )
        self._base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        self.onnx_path = Path(onnx_path) if onnx_path is not None else None
        self._session = None
        self._input_name = ""
        self.last_action = np.zeros(ACTION_SIZE, dtype=np.float64)
        self._ctrl_target = np.zeros(model.nu, dtype=np.float64)
        self.safety_recovery_active = False
        # Resolved lazily so importing the module never requires onnxruntime.
        self._providers = tuple(providers)
        self._stop_pose = None
        self._brake_wheels = np.zeros(4, dtype=np.float64)

    def _command(self, command):
        if getattr(self, 'teacher_motion_limits', None) is None:
            return super()._command(command)
        c = command if isinstance(command, M20BodyCommand) else M20BodyCommand.from_array(command)
        return M20BodyCommand(float(np.clip(c.forward, -.50, .50)), 0.,
                              float(np.clip(c.yaw, -.40, .40)), c.stop)

    @staticmethod
    def _verify_actuator_order(model: mujoco.MjModel) -> None:
        actual = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0]))
            for i in range(model.nu)
        )
        if actual != ACTION_JOINT_NAMES:
            raise RuntimeError(
                "M20 MJCF actuator order does not match the policy action order; "
                f"expected {ACTION_JOINT_NAMES}, found {actual}"
            )

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - dependency guidance
            raise RuntimeError(
                "The learned M20 low-level policy needs onnxruntime; install the "
                "'m20pro-vla[vla]' extra or `pip install onnxruntime`."
            ) from error
        path = resolve_onnx_path(self.onnx_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"v5 policy weights not found at {path}. Pass onnx_path= or set "
                f"{ONNX_PATH_ENV}."
            )
        self.onnx_path = path
        options = None
        if self.inference_threads is not None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.inference_threads
            options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(path), sess_options=options, providers=list(self._providers))
        inputs = self._session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape[-1] != OBSERVATION_SIZE:
            raise RuntimeError(
                f"Unexpected v5 policy signature: {[(i.name, i.shape) for i in inputs]}"
            )
        self._input_name = inputs[0].name
        return self._session

    def reset(self, data: mujoco.MjData, yaw: float = 0.0,
              base_xy: tuple[float, float] | np.ndarray | None = None) -> None:
        mujoco.mj_resetData(self.model, data)
        if base_xy is None:
            base_xy = (0.0, 0.0)
        origin = np.asarray(base_xy, dtype=np.float64)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("base_xy must be two finite values")
        data.qpos[0:3] = (float(origin[0]), float(origin[1]), SPAWN_HEIGHT)
        data.qpos[3:7] = yaw_quaternion(float(yaw))
        data.qpos[self._leg_qpos] = STANCE
        for name in WHEEL_JOINT_NAMES:
            data.qpos[self.model.jnt_qposadr[self._joint_id[name]]] = 0.0
        data.qvel[:] = 0.0
        self.last_action.fill(0.0)
        self._ctrl_target.fill(0.0)
        self._stop_pose = None
        self._brake_wheels.fill(0.0)
        self.safety_recovery_active = False
        mujoco.mj_forward(self.model, data)

    def snapshot(self) -> M20V5PolicyControllerState:
        return M20V5PolicyControllerState(
            last_action=self.last_action.copy(),
            safety_recovery_active=bool(self.safety_recovery_active),
            stop_pose=None if self._stop_pose is None else self._stop_pose.copy(),
            brake_wheels=self._brake_wheels.copy(),
        )

    def restore(self, state: M20V5PolicyControllerState) -> None:
        self.last_action[:] = state.last_action
        self.safety_recovery_active = bool(state.safety_recovery_active)
        self._stop_pose = None if state.stop_pose is None else state.stop_pose.copy()
        self._brake_wheels = np.zeros(4) if state.brake_wheels is None else state.brake_wheels.copy()
        self._ctrl_target[:12] = STANCE + LEG_ACTION_SCALE * self.last_action[:12]
        self._ctrl_target[12:] = WHEEL_ACTION_SCALE * self.last_action[12:]

    def _attitude(self, data: mujoco.MjData) -> tuple[float, float]:
        rotation = data.xmat[self._base_body_id].reshape(3, 3)
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        pitch = float(np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2])))
        return roll, pitch

    def _tracked_command(self, command: M20BodyCommand) -> np.ndarray:
        if command.stop or self.safety_recovery_active:
            return np.zeros(3, dtype=np.float64)
        return np.array((command.forward, command.lateral, command.yaw), dtype=np.float64)

    def _observation(self, data: mujoco.MjData, command: np.ndarray) -> np.ndarray:
        rotation = data.xmat[self._base_body_id].reshape(3, 3)
        angular_velocity = data.qvel[3:6]
        gravity = rotation.T @ np.array((0.0, 0.0, -1.0), dtype=np.float64)
        leg_position = data.qpos[self._leg_observation_qpos] - self._leg_observation_default
        joint_velocity = data.qvel[self._observation_dof]
        return np.concatenate((
            angular_velocity * ANGULAR_VELOCITY_SCALE,
            gravity,
            command,
            leg_position,
            joint_velocity * JOINT_VELOCITY_SCALE,
            self.last_action,
        )).astype(np.float32).reshape(1, OBSERVATION_SIZE)

    def _infer(self, observation: np.ndarray) -> np.ndarray:
        session = self._ensure_session()
        action = session.run(None, {self._input_name: observation})[0]
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != ACTION_SIZE or not np.isfinite(action).all():
            raise RuntimeError(f"v5 policy produced an unusable action: {action!r}")
        return action

    def apply(self, data: mujoco.MjData,
              command: M20BodyCommand | np.ndarray | list[float] | tuple[float, ...],
              ) -> M20LowLevelDiagnostics:
        command = self._command(command)
        if command.stop and self._stop_pose is None:
            self._stop_pose = data.qpos[self._leg_qpos].copy()
            self._brake_wheels = self._ctrl_target[list(WHEEL_ACTION_ACTUATORS)].copy()
        action = self._infer(self._observation(data, self._tracked_command(command)))
        self.last_action[:] = action
        self._ctrl_target[list(LEG_ACTION_ACTUATORS)] = STANCE + LEG_ACTION_SCALE * action[:12]
        self._ctrl_target[list(WHEEL_ACTION_ACTUATORS)] = WHEEL_ACTION_SCALE * action[12:]
        if command.stop:
            # A zero input to the locomotion network still generates wheel and
            # leg motion. Hold the attained stance and ramp wheel targets down
            # together; abruptly clamping only wheels destabilizes the body.
            self._brake_wheels += np.clip(-self._brake_wheels, -0.2, 0.2)
            self._ctrl_target[list(LEG_ACTION_ACTUATORS)] = self._stop_pose
            self._ctrl_target[list(WHEEL_ACTION_ACTUATORS)] = self._brake_wheels
            self.last_action[:12] = (self._stop_pose - STANCE) / LEG_ACTION_SCALE
            self.last_action[12:] = self._brake_wheels / WHEEL_ACTION_SCALE
        else:
            self._stop_pose = None
        data.ctrl[:] = self._ctrl_target

        roll, pitch = self._attitude(data)
        tilt = max(abs(roll), abs(pitch))
        if tilt >= RECOVERY_TRIGGER_TILT_RAD or float(data.qpos[2]) <= RECOVERY_TRIGGER_HEIGHT:
            self.safety_recovery_active = True
        elif (self.safety_recovery_active
              and tilt <= RECOVERY_CLEAR_TILT_RAD
              and float(data.qpos[2]) >= RECOVERY_CLEAR_HEIGHT):
            self.safety_recovery_active = False

        return M20LowLevelDiagnostics(
            contract=self.contract,
            command=command.as_array(),
            wheel_target=self._ctrl_target[list(WHEEL_ACTION_ACTUATORS)].copy(),
            base_height=float(data.qpos[2]),
            roll=roll,
            pitch=pitch,
            linear_velocity=data.qvel[:3].copy(),
            angular_velocity=data.qvel[3:6].copy(),
            lateral_command=command.lateral,
            lateral_supported=self.lateral_supported,
            terrain_lift_active=False,
            safety_recovery_active=self.safety_recovery_active,
            finite=bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
        )


__all__ = [
    "DEFAULT_ONNX_PATH",
    "M20V5PolicyController",
    "M20V5PolicyControllerState",
    "ONNX_PATH_ENV",
    "POLICY_CONTRACT",
    "STANCE",
    "resolve_onnx_path",
]
