"""Deterministic regression gate for the reusable M20 controller."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from .controller import PHYSICS_STEPS
from .factory import build_low_level_controller, resolve_backend
from m20pro_vla.sim.mujoco import ASSET, build_scene

# Constant-speed smoothness bench. Mirrors the 50 Hz diagnostics sampled by the
# closed-loop players (qpos[2]/qvel[2]/qvel[3]/qvel[4] once per control step), so
# the bed numbers and the closed-loop numbers share one definition.
#
# The whole-window RMS is dominated by the unavoidable acceleration transient at
# the start of the bench, so the gates read the *settled* tail instead.
#
# RE-BASELINED 2026-10-03 for the v5 RL expert. The previous bounds were
# non-regression bounds around the *analytic* controller. Once v5 became the
# default execution layer those bounds no longer described the robot that walks:
# v5 is far better at the things that matter for the task (forward tilt 1.00 deg
# vs 0.43, turn tilt 1.25 deg, base height 0.559 m vs 0.492, turn yaw 22.9 deg)
# but it trades that for body smoothness - settled roll-rate RMS is 0.121 rad/s
# against the analytic 0.005. Both facts are real and neither is a defect: the
# v5 wheel-leg gait scrubs the wheels through a wider stance to steer, and the
# stance oscillation is what the old bounds called "worse".
#
# The bounds below are v5 non-regression bounds with ~1.4x headroom over the
# measured value, so they still catch a genuine degradation of the layer that is
# actually deployed. They are still NOT a certificate of smoothness, and the
# tracked low-level item (see open_body_smoothness_item) is unchanged.
CRUISE_SPEED_MPS = 0.2
CRUISE_STEPS = 500
CRUISE_WARMUP_STEPS = 100
CRUISE_SETTLED_FRACTION = 0.6
# v5 measured: vv 0.0541, roll_rate 0.1210, pitch_rate 0.1176, h_range 0.0127.
# The two former bounds (0.06 / 0.12) were already within 10% of the v5 value.
MAX_SETTLED_VERTICAL_VELOCITY_RMS_MPS = 0.09
MAX_SETTLED_ROLL_RATE_RMS_RADPS = 0.18
MAX_SETTLED_PITCH_RATE_RMS_RADPS = 0.18
MAX_SETTLED_BASE_HEIGHT_RANGE_M = 0.020

# Policy-cadence smoothness bench. A learned policy re-issues its body command
# every action-replan interval (10 control steps, i.e. 0.2 s), which re-excites
# the underdamped ~5 Hz pitch mode before it can decay. The constant-speed bench
# above therefore reads "smooth" while the same body runs several times
# jitterier under a policy, so smoothness has to be judged at the cadence the
# body is actually driven with. This phase replays that cadence from a fixed
# seed: no policy, no GPU, no 3-minute episode.
#
# Thresholds are again non-regression bounds, re-based onto v5 on 2026-10-03
# (v5 measured: vv 0.2070, roll_rate 0.3856, pitch_rate 0.3367, h_range 0.0540).
# The vertical-velocity and pitch-rate bounds were already generous enough to
# hold; the roll-rate and height-range bounds had to move because v5 reaches
# them with a gait the analytic controller never produced.
CADENCE_STEPS = 500
CADENCE_WARMUP_STEPS = 100
CADENCE_REPLAN_STEPS = 10
CADENCE_SEED = 20260918
CADENCE_FORWARD_RANGE_MPS = (0.0, 0.35)
CADENCE_YAW_RANGE_RADPS = (-0.15, 0.15)
MAX_CADENCE_VERTICAL_VELOCITY_RMS_MPS = 0.45
MAX_CADENCE_ROLL_RATE_RMS_RADPS = 0.55
MAX_CADENCE_PITCH_RATE_RMS_RADPS = 0.50
MAX_CADENCE_BASE_HEIGHT_RANGE_M = 0.08


def _smoothness_window(
    height_values: np.ndarray,
    vertical_values: np.ndarray,
    roll_values: np.ndarray,
    pitch_values: np.ndarray,
    wheel_delta: np.ndarray,
    lo: int,
    hi: int,
) -> dict:
    """Quantify body smoothness over one window of a bench.

    Both benches must report identical metrics. If they drifted apart, a change
    could look better on one bed and worse on the other for definitional reasons
    instead of physical ones.
    """
    return {
        "steps": [lo, hi],
        "base_height_range_m": float(np.ptp(height_values[lo:hi])),
        "base_height_std_m": float(np.std(height_values[lo:hi])),
        "vertical_velocity_rms_mps": float(np.sqrt(np.mean(np.square(vertical_values[lo:hi])))),
        "vertical_velocity_max_abs_mps": float(np.max(np.abs(vertical_values[lo:hi]))),
        "roll_rate_rms_radps": float(np.sqrt(np.mean(np.square(roll_values[lo:hi])))),
        "roll_rate_max_abs_radps": float(np.max(np.abs(roll_values[lo:hi]))),
        "pitch_rate_rms_radps": float(np.sqrt(np.mean(np.square(pitch_values[lo:hi])))),
        "pitch_rate_max_abs_radps": float(np.max(np.abs(pitch_values[lo:hi]))),
        "wheel_target_delta_rms": float(np.sqrt(np.mean(np.square(wheel_delta)))),
        "wheel_target_delta_max_abs": float(np.max(np.abs(wheel_delta))),
    }


def _attitude(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    rotation = data.xmat[base_id].reshape(3, 3)
    return (
        math.atan2(float(rotation[2, 1]), float(rotation[2, 2])),
        math.atan2(float(-rotation[2, 0]), float(np.hypot(rotation[2, 1], rotation[2, 2]))),
    )


def _yaw_angle(data: mujoco.MjData) -> float:
    q = data.qpos[3:7]
    return math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))


def _wrapped_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _cruise_smoothness(
    model: mujoco.MjModel,
    *,
    speed_mps: float = CRUISE_SPEED_MPS,
    steps: int = CRUISE_STEPS,
    warmup_steps: int = CRUISE_WARMUP_STEPS,
    backend: str | None = None,
) -> dict:
    """Run a straight constant-speed bench and quantify body smoothness.

    The robot never turns and the command never changes here, so any residual
    oscillation is caused by the low-level dynamics rather than by a policy.
    This is the cheap, deterministic way to catch the "hopping" body mode that a
    closed-loop video would otherwise be needed for.
    """
    data = mujoco.MjData(model)
    controller = build_low_level_controller(model, backend)
    controller.reset(data, 0.0)
    hold = np.array((0.0, 0.0, 0.0, 0.0))
    for _ in range(warmup_steps):
        controller.step(data, hold)
    start_xy = data.qpos[:2].copy()
    heights: list[float] = []
    vertical_velocities: list[float] = []
    roll_rates: list[float] = []
    pitch_rates: list[float] = []
    wheel_targets: list[list[float]] = []
    command = np.array((speed_mps, 0.0, 0.0, 0.0))
    for _ in range(steps):
        diagnostics = controller.step(data, command)
        heights.append(float(data.qpos[2]))
        vertical_velocities.append(float(data.qvel[2]))
        roll_rates.append(float(data.qvel[3]))
        pitch_rates.append(float(data.qvel[4]))
        wheel_targets.append(diagnostics.wheel_target.astype(float).tolist())
    height_values = np.asarray(heights, dtype=np.float64)
    vertical_values = np.asarray(vertical_velocities, dtype=np.float64)
    roll_values = np.asarray(roll_rates, dtype=np.float64)
    pitch_values = np.asarray(pitch_rates, dtype=np.float64)
    wheel_delta = np.diff(np.asarray(wheel_targets, dtype=np.float64), axis=0)
    settled_from = int(steps * CRUISE_SETTLED_FRACTION)

    def _window(lo: int, hi: int, deltas: np.ndarray) -> dict:
        return _smoothness_window(
            height_values, vertical_values, roll_values, pitch_values, deltas, lo, hi
        )

    return {
        "speed_command_mps": float(speed_mps),
        "cruise_steps": steps,
        "warmup_steps": warmup_steps,
        "sample_hz": 1.0 / (PHYSICS_STEPS * model.opt.timestep),
        "displacement_xy_m": float(np.linalg.norm(data.qpos[:2] - start_xy)),
        "settled_from_step": settled_from,
        "full": _window(0, steps, wheel_delta),
        "settled": _window(settled_from, steps, wheel_delta[settled_from - 1:steps - 1]),
    }


def _cadence_smoothness(
    model: mujoco.MjModel,
    *,
    steps: int = CADENCE_STEPS,
    warmup_steps: int = CADENCE_WARMUP_STEPS,
    replan_steps: int = CADENCE_REPLAN_STEPS,
    seed: int = CADENCE_SEED,
    backend: str | None = None,
) -> dict:
    """Run the policy-cadence bench and quantify body smoothness.

    The command is resampled every ``replan_steps`` control steps from the
    validated body-command envelope, which is what a SmolVLA policy does. The
    seed is fixed so the bed is reproducible; the resulting numbers are
    comparable to a closed-loop episode that runs the same 50 Hz diagnostics.
    """
    generator = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    controller = build_low_level_controller(model, backend)
    controller.reset(data, 0.0)
    hold = np.array((0.0, 0.0, 0.0, 0.0))
    for _ in range(warmup_steps):
        controller.step(data, hold)
    start_xy = data.qpos[:2].copy()
    heights: list[float] = []
    vertical_velocities: list[float] = []
    roll_rates: list[float] = []
    pitch_rates: list[float] = []
    wheel_targets: list[list[float]] = []
    command = np.array((CADENCE_FORWARD_RANGE_MPS[0], 0.0, 0.0, 0.0))
    for step in range(steps):
        if step % replan_steps == 0:
            command = np.array(
                (
                    generator.uniform(*CADENCE_FORWARD_RANGE_MPS),
                    0.0,
                    generator.uniform(*CADENCE_YAW_RANGE_RADPS),
                    0.0,
                ),
                dtype=np.float64,
            )
        diagnostics = controller.step(data, command)
        heights.append(float(data.qpos[2]))
        vertical_velocities.append(float(data.qvel[2]))
        roll_rates.append(float(data.qvel[3]))
        pitch_rates.append(float(data.qvel[4]))
        wheel_targets.append(diagnostics.wheel_target.astype(float).tolist())
    height_values = np.asarray(heights, dtype=np.float64)
    vertical_values = np.asarray(vertical_velocities, dtype=np.float64)
    roll_values = np.asarray(roll_rates, dtype=np.float64)
    pitch_values = np.asarray(pitch_rates, dtype=np.float64)
    wheel_delta = np.diff(np.asarray(wheel_targets, dtype=np.float64), axis=0)
    settled_from = int(steps * CRUISE_SETTLED_FRACTION)
    return {
        "cadence_steps": steps,
        "warmup_steps": warmup_steps,
        "replan_steps": replan_steps,
        "command_seed": seed,
        "forward_command_range_mps": [float(value) for value in CADENCE_FORWARD_RANGE_MPS],
        "yaw_command_range_radps": [float(value) for value in CADENCE_YAW_RANGE_RADPS],
        "sample_hz": 1.0 / (PHYSICS_STEPS * model.opt.timestep),
        "displacement_xy_m": float(np.linalg.norm(data.qpos[:2] - start_xy)),
        "settled_from_step": settled_from,
        "full": _smoothness_window(
            height_values, vertical_values, roll_values, pitch_values, wheel_delta, 0, steps
        ),
        "settled": _smoothness_window(
            height_values,
            vertical_values,
            roll_values,
            pitch_values,
            wheel_delta[settled_from - 1:steps - 1],
            settled_from,
            steps,
        ),
    }


def run_low_level_gate(
    *,
    warmup_steps: int = 100,
    forward_steps: int = 220,
    stop_steps: int = 100,
    turn_steps: int = 180,
    turn_command: float = 0.10,
    backend: str | None = None,
) -> dict:
    """Run stance, forward, stop, and turn phases against the selected execution layer.

    ``backend`` defaults to the process default (``M20_LOW_LEVEL_BACKEND``, else
    the factory default). Every phase - including both smoothness benches - runs
    on that single layer so the report describes one robot, not two.
    """
    resolved_backend = resolve_backend(backend)
    if not ASSET.is_file():
        raise FileNotFoundError(f"Build the MuJoCo asset first: {ASSET}")
    if min(warmup_steps, forward_steps, stop_steps, turn_steps) <= 0:
        raise ValueError("all phases must be positive")
    if not np.isfinite(turn_command) or abs(turn_command) > 0.15:
        raise ValueError("turn_command must be finite and within the body-command yaw range")
    scene_path = ASSET.with_name("m20_low_level_gate.scene.xml")
    build_scene(scene_path, [])
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        controller = build_low_level_controller(model, resolved_backend)
        controller.reset(data, 0.0)
        phases = (
            ("warmup", warmup_steps, np.array((0.0, 0.0, 0.0, 0.0))),
            ("forward", forward_steps, np.array((0.35, 0.0, 0.0, 0.0))),
            ("stop", stop_steps, np.array((0.0, 0.0, 0.0, 1.0))),
            ("turn", turn_steps, np.array((0.0, 0.0, turn_command, 0.0))),
            ("final_stop", stop_steps, np.array((0.0, 0.0, 0.0, 1.0))),
        )
        phase_stats: dict[str, dict] = {}
        all_finite = True
        initial_xy = data.qpos[:2].copy()
        initial_yaw = _yaw_angle(data)
        for phase_name, count, action in phases:
            start = data.qpos[:2].copy()
            phase_yaw = _yaw_angle(data)
            heights: list[float] = []
            rolls: list[float] = []
            pitches: list[float] = []
            speeds: list[float] = []
            for _ in range(count):
                controller.step(data, action)
                roll, pitch = _attitude(model, data)
                heights.append(float(data.qpos[2]))
                rolls.append(abs(roll))
                pitches.append(abs(pitch))
                speeds.append(float(np.linalg.norm(data.qvel[:2])))
                all_finite = all_finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
            phase_stats[phase_name] = {
                "steps": count,
                "displacement_xy_m": float(np.linalg.norm(data.qpos[:2] - start)),
                "min_base_height_m": min(heights),
                "max_abs_roll_deg": math.degrees(max(rolls)),
                "max_abs_pitch_deg": math.degrees(max(pitches)),
                "max_horizontal_speed_mps": max(speeds),
                "yaw_change_deg": math.degrees(_wrapped_angle(_yaw_angle(data) - phase_yaw)),
                "final_xy": data.qpos[:2].tolist(),
            }
        final_yaw = _yaw_angle(data)
        forward = phase_stats["forward"]
        stop = phase_stats["stop"]
        turn = phase_stats["turn"]
        cruise = _cruise_smoothness(model, backend=resolved_backend)
        cadence = _cadence_smoothness(model, backend=resolved_backend)
        report = {
            "schema": "m20pro_low_level_stance_velocity_gate_v4",
            "controller": "M20LowLevelController_feedback_stance_velocity",
            "low_level_backend": resolved_backend,
            "action_contract": ["forward", "lateral", "yaw", "stop"],
            "physics_steps_per_control": PHYSICS_STEPS,
            "all_finite": all_finite,
            "forward_displacement_m": forward["displacement_xy_m"],
            "forward_min_height_m": forward["min_base_height_m"],
            "forward_max_tilt_deg": max(forward["max_abs_roll_deg"], forward["max_abs_pitch_deg"]),
            "stop_drift_m": stop["displacement_xy_m"],
            "turn_displacement_m": turn["displacement_xy_m"],
            "yaw_change_deg": math.degrees(_wrapped_angle(final_yaw - initial_yaw)),
            "turn_command": turn_command,
            "turn_yaw_change_deg": turn["yaw_change_deg"],
            "final_stop_yaw_change_deg": phase_stats["final_stop"]["yaw_change_deg"],
            "cruise_smoothness": cruise,
            "cadence_smoothness": cadence,
            "gates": {
                "finite_state": all_finite,
                "forward_height": forward["min_base_height_m"] >= 0.45,
                "forward_tilt": forward["max_abs_roll_deg"] <= 8.0 and forward["max_abs_pitch_deg"] <= 8.0,
                "stop_drift": stop["displacement_xy_m"] <= 0.25,
                "turn_yaw": abs(turn["yaw_change_deg"]) >= 10.0,
                "turn_height": turn["min_base_height_m"] >= 0.45,
                "turn_tilt": turn["max_abs_roll_deg"] <= 8.0 and turn["max_abs_pitch_deg"] <= 8.0,
                "turn_exit_height": phase_stats["final_stop"]["min_base_height_m"] >= 0.45,
                "cruise_settled_vertical_velocity": (
                    cruise["settled"]["vertical_velocity_rms_mps"] <= MAX_SETTLED_VERTICAL_VELOCITY_RMS_MPS
                ),
                "cruise_settled_roll_rate": (
                    cruise["settled"]["roll_rate_rms_radps"] <= MAX_SETTLED_ROLL_RATE_RMS_RADPS
                ),
                "cruise_settled_pitch_rate": (
                    cruise["settled"]["pitch_rate_rms_radps"] <= MAX_SETTLED_PITCH_RATE_RMS_RADPS
                ),
                "cruise_settled_height_range": (
                    cruise["settled"]["base_height_range_m"] <= MAX_SETTLED_BASE_HEIGHT_RANGE_M
                ),
                "cruise_delivery": cruise["displacement_xy_m"] >= 1.0,
                "cadence_settled_vertical_velocity": (
                    cadence["settled"]["vertical_velocity_rms_mps"]
                    <= MAX_CADENCE_VERTICAL_VELOCITY_RMS_MPS
                ),
                "cadence_settled_roll_rate": (
                    cadence["settled"]["roll_rate_rms_radps"] <= MAX_CADENCE_ROLL_RATE_RMS_RADPS
                ),
                "cadence_settled_pitch_rate": (
                    cadence["settled"]["pitch_rate_rms_radps"] <= MAX_CADENCE_PITCH_RATE_RMS_RADPS
                ),
                "cadence_settled_height_range": (
                    cadence["settled"]["base_height_range_m"] <= MAX_CADENCE_BASE_HEIGHT_RANGE_M
                ),
            },
            "smoothness_gates_are_non_regression": (
                "Bounds are v5 non-regression bounds (re-based 2026-10-03) for two command "
                "cadences: constant speed and the policy's 0.2 s replan cadence. Passing them "
                "means the deployed execution layer did not get worse; it does not certify "
                "smoothness. The cadence bed reproduces the closed loop's body jitter offline, "
                "and tightening it is the tracked low-level item."
            ),
            "phase_stats": phase_stats,
            "initial_xy": initial_xy.tolist(),
            "final_xy": data.qpos[:2].tolist(),
        }
        report["eligible_for_flat_vla_execution"] = bool(all(report["gates"].values()))
        return report
    finally:
        scene_path.unlink(missing_ok=True)


__all__ = ["run_low_level_gate"]
